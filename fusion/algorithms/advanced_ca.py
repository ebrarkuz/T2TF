"""
füzyon2.py
==========
Gorevi: radar_sensor_tracks.csv uzerinden Covariance Intersection (CI)
fuzyon algoritmasini (Sabit Ivme Modeli ile) calistirip fused_tracks.csv cikisi olusturmaktir.
"""

import math
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar

warnings.filterwarnings("ignore")

# ===========================================================================
# KONFIGURASYON & SABITLER
# ===========================================================================
PROCESS_NOISE_INTENSITY = 1.5  # CA modelinde "Jerk (İvme değişimi)" varyansını temsil eder
ADAPTIVE_Q_ENABLED = True
Q_SCALE_MIN = 1.0
Q_SCALE_MAX = 12.0
NIS_Q_LOW_THRESHOLD = 6.0
NIS_Q_HIGH_THRESHOLD = 18.0
Q_SCALE_UP_FACTOR = 1.8
Q_SCALE_DOWN_FACTOR = 0.90
XY_MANEUVER_Q_GAIN = 2.0
Z_MANEUVER_Q_GAIN = 1.0

BASE_GATE_CHI2_6DOF = 22.5
GATE_SCALE_MIN = 1.0
GATE_SCALE_MAX = 1.8
MAX_ASSOC_VEL_DIFF_MPS = 120.0
MAX_ASSOC_ACCEL_MPS2 = 30.0
MAX_POSITION_INNOVATION_M = 750.0
POSITION_GATE_SIGMA = 3.0
VELOCITY_GATE_SIGMA = 3.0
MIN_ACCEL_DT_S = 1.0
COAST_TIME_LIMIT = 30.0
CONFIRMED_OUTPUT_COAST_S = 2.0
CONFIRM_HITS = 6               # minimum number of updates before a track can be CONFIRMED
MIN_CONFIRM_AGE_S = 8.0
MIN_CONFIRM_SENSORS = 2
DUPLICATE_DIST_M = 150.0       # IMM kodundaki gibi genişletildi
DUPLICATE_VEL_MPS = 30.0       # YENI: Paralel track'lerin maks hız farkı (m/s)
DUPLICATE_TIME_S = 5.0         # time window for duplicate appearance check (seconds)

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5
SENSOR_CSV = "radar_sensor_tracks.csv"
OUTPUT_FUSED_CSV = "fused_tracks.csv"

# ===========================================================================
# YARDIMCI FONKSIYONLAR
# ===========================================================================
def tq_to_sigma_pos(tq) -> float:
    tq = float(np.clip(tq, TQ_MIN, TQ_MAX))
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac

def tq_to_sigma_vel(tq) -> float:
    tq = float(np.clip(tq, TQ_MIN, TQ_MAX))
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac

def tq_to_cov(tq) -> np.ndarray:
    sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
    return np.diag([sp**2, sv**2, sp**2, sv**2, sp**2, sv**2])

def sigma_pos_to_tq(sigma_pos: float) -> int:
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))

def measurement_cov_from_row(row) -> np.ndarray:
    if "sigma_pos_m" in row and "sigma_vel_mps" in row:
        sp = float(row["sigma_pos_m"])
        sv = float(row["sigma_vel_mps"])
        return np.diag([sp**2, sv**2, sp**2, sv**2, sp**2, sv**2])
    return tq_to_cov(row["track_quality"])


def _passes_physical_association_gates(gt, diff, innovation_cov) -> bool:
    """Belirsizliği hesaba katan konum, hız ve ivme tutarlılık kontrolü."""
    pos_indices = [0, 2, 4]
    vel_indices = [1, 3, 5]
    diagonal = np.maximum(np.diag(innovation_cov), 0.0)

    pos_diff = float(np.linalg.norm(diff[pos_indices, 0]))
    velocity_diff = float(np.linalg.norm(diff[vel_indices, 0]))
    pos_sigma = math.sqrt(float(np.sum(diagonal[pos_indices])))
    velocity_sigma = math.sqrt(float(np.sum(diagonal[vel_indices])))

    # Sabit 750 m kapı, düşük TQ ve uzak menzilde geçerli ölçümleri kesiyordu.
    # Gate taban değerin altına düşmez, belirsizlik büyüdüğünde adaptif genişler.
    position_limit = max(
        MAX_POSITION_INNOVATION_M,
        POSITION_GATE_SIGMA * pos_sigma,
    )

    # Ham hız farkının tamamını ivme saymak, yakın zamanlı iki radar ölçümünde
    # gürültüyü devasa bir ivmeye dönüştürüyordu. Önce 3-sigma ölçüm payını düş.
    unexplained_velocity = max(
        0.0,
        velocity_diff - VELOCITY_GATE_SIGMA * velocity_sigma,
    )
    update_dt = max(gt.time - gt.last_update, MIN_ACCEL_DT_S)
    implied_accel = unexplained_velocity / update_dt

    return (
        velocity_diff <= MAX_ASSOC_VEL_DIFF_MPS
        and implied_accel <= MAX_ASSOC_ACCEL_MPS2
        and pos_diff <= position_limit
    )

def _pad_6d_to_9d(state_6d, cov_6d):
    """6D ölçümü 9D (3B sabit ivme) duruma genişletir. İvme varyansı devasa bırakılır."""
    state_9d = np.zeros((9, 1))
    state_9d[0, 0] = state_6d[0, 0]  # x
    state_9d[1, 0] = state_6d[1, 0]  # vx
    state_9d[3, 0] = state_6d[2, 0]  # y
    state_9d[4, 0] = state_6d[3, 0]  # vy
    state_9d[6, 0] = state_6d[4, 0]  # z
    state_9d[7, 0] = state_6d[5, 0]  # vz

    cov_9d = np.eye(9) * 1e5
    cov_9d[0:2, 0:2] = cov_6d[0:2, 0:2]
    cov_9d[3:5, 3:5] = cov_6d[2:4, 2:4]
    cov_9d[6:8, 6:8] = cov_6d[4:6, 4:6]

    return state_9d, cov_9d

# ===========================================================================
# FUZYON ALGORITMASI SINIFLARI
# ===========================================================================
def _ci_fuse(x1, P1, x2, P2):
    P1i, P2i = np.linalg.inv(P1), np.linalg.inv(P2)
    def trace_f(omega):
        try:
            return float(np.trace(np.linalg.inv(omega * P1i + (1 - omega) * P2i)))
        except np.linalg.LinAlgError:
            return np.inf

    res = minimize_scalar(trace_f, bounds=(1e-3, 1 - 1e-3), method="bounded")
    omega = res.x
    Pf = np.linalg.inv(omega * P1i + (1 - omega) * P2i)
    Pf = 0.5 * (Pf + Pf.T)
    xf = Pf @ (omega * P1i @ x1 + (1 - omega) * P2i @ x2)
    return xf, Pf

def _standard_fuse(x1, P1, x2, P2):
    """Standart Kalman (LMMSE) Güncellemesi. Hataların bağımsız olduğunu varsayar."""
    try:
        S = P1 + P2
        S_inv = np.linalg.inv(S)
        K = P1 @ S_inv
        xf = x1 + K @ (x2 - x1)
        Pf = P1 - K @ P1
        Pf = 0.5 * (Pf + Pf.T) # Simetriyi koru
        return xf, Pf
    except np.linalg.LinAlgError:
        return x1, P1

class GlobalTrack:
    _cnt = 0
    def __init__(
        self, t, state, cov, tq, src, use_ci=True,
        adaptive_q_enabled=ADAPTIVE_Q_ENABLED,
    ):
        GlobalTrack._cnt += 1
        self.id = f"GT-{GlobalTrack._cnt:04d}"
        self.time = t
        
        # 6D veriyi 9D CA modeline genişlet
        state_9d, cov_9d = _pad_6d_to_9d(state.reshape(6, 1), cov)
        self.state = state_9d
        self.cov = cov_9d
        
        self.last_update = t
        self.existence_prob = 0.1 + 0.7 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.status = "TENTATIVE"
        self.sources: set = {src}
        self.source_radar_names: set = {src[0] if isinstance(src, tuple) else src}
        self.source_measurement_details: set = {f"{src[0] if isinstance(src, tuple) else src}@{t:.2f}"}
        
        self.hits_count = 1
        self.creation_time = t
        self.position_history = [(float(self.state[0,0]), float(self.state[3,0]), float(self.state[6,0]), float(t))]
        self.use_ci = use_ci
        self.adaptive_q_enabled = adaptive_q_enabled
        self.q_scale = 1.0
        self.last_nis = 0.0
        self.maneuver_score = 0.0

    def update_maneuver_adaptation(self, nis):
        """Update per-track process-noise scale after an accepted association."""
        if not np.isfinite(nis) or nis < 0.0:
            return

        self.last_nis = float(nis)
        denominator = NIS_Q_HIGH_THRESHOLD - NIS_Q_LOW_THRESHOLD
        normalized = (self.last_nis - NIS_Q_LOW_THRESHOLD) / denominator
        new_score = float(np.clip(normalized, 0.0, 1.0))
        self.maneuver_score = 0.8 * self.maneuver_score + 0.2 * new_score

        if not self.adaptive_q_enabled:
            return

        if self.maneuver_score > 0.7:
            self.q_scale *= Q_SCALE_UP_FACTOR
        elif self.maneuver_score < 0.2:
            self.q_scale *= Q_SCALE_DOWN_FACTOR

        self.q_scale = float(np.clip(self.q_scale, Q_SCALE_MIN, Q_SCALE_MAX))

    def get_effective_gate(self):
        if self.maneuver_score <= 0.6:
            return BASE_GATE_CHI2_6DOF
        gate_scale = 1.0 + 0.8 * self.maneuver_score
        gate_scale = float(np.clip(gate_scale, GATE_SCALE_MIN, GATE_SCALE_MAX))
        return BASE_GATE_CHI2_6DOF * gate_scale

    def propagate(self, t):
        dt = t - self.time
        if dt <= 0: return
        
        # 9D (x,y,z eksenlerinde CA) durum geçiş matrisi
        F = np.array([
            [1, dt, 0.5 * dt**2, 0,  0,          0, 0,  0,          0],
            [0,  1,          dt, 0,  0,          0, 0,  0,          0],
            [0,  0,           1, 0,  0,          0, 0,  0,          0],
            [0,  0,           0, 1, dt, 0.5 * dt**2, 0,  0,          0],
            [0,  0,           0, 0,  1,         dt, 0,  0,          0],
            [0,  0,           0, 0,  0,          1, 0,  0,          0],
            [0,  0,           0, 0,  0,          0, 1, dt, 0.5 * dt**2],
            [0,  0,           0, 0,  0,          0, 0,  1,         dt],
            [0,  0,           0, 0,  0,          0, 0,  0,          1],
        ])
        
        # 9D continuous-white-jerk süreç gürültüsü matrisi.
        q = PROCESS_NOISE_INTENSITY ** 2
        adaptive_q = q * self.q_scale if self.adaptive_q_enabled else q
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4; dt5 = dt**5
        base_q_block = np.array([
            [dt5/20, dt4/8, dt3/6],
            [dt4/8, dt3/3, dt2/2],
            [dt3/6, dt2/2, dt]
        ])

        q_xy = adaptive_q * base_q_block
        if self.q_scale > 2.0:
            q_xy *= XY_MANEUVER_Q_GAIN
        q_z = adaptive_q * base_q_block * Z_MANEUVER_Q_GAIN
        
        Q = np.zeros((9, 9))
        Q[np.ix_([0,1,2],[0,1,2])] = q_xy
        Q[np.ix_([3,4,5],[3,4,5])] = q_xy
        Q[np.ix_([6,7,8],[6,7,8])] = q_z
        
        self.state = F @ self.state
        self.cov = 0.5 * ((F @ self.cov @ F.T + Q) + (F @ self.cov @ F.T + Q).T)
        self.time = t
        # Olasılığı radar mesajı sayısına göre değil geçen gerçek süreye göre azalt.
        self.existence_prob *= 0.97 ** dt
        self._update_status()

    def update(self, meas_state, meas_cov, tq, src):
        m_state_9d, m_cov_9d = _pad_6d_to_9d(meas_state, meas_cov)
        
        if self.use_ci:
            xf, Pf = _ci_fuse(self.state, self.cov, m_state_9d, m_cov_9d)
        else:
            xf, Pf = _standard_fuse(self.state, self.cov, m_state_9d, m_cov_9d)
            
        self.state, self.cov = xf, Pf
        self.last_update = self.time
        mp = 0.5 + 0.45 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.existence_prob = self.existence_prob + (1 - self.existence_prob) * mp
        self.sources.add(src)
        self.source_radar_names.add(src[0] if isinstance(src, tuple) else src)
        self.source_measurement_details.add(f"{src[0] if isinstance(src, tuple) else src}@{self.time:.2f}")
        
        self.hits_count += 1
        self.position_history.append((float(self.state[0,0]), float(self.state[3,0]), float(self.state[6,0]), float(self.time)))
        self._update_status()

    def _update_status(self):
        if (self.time - self.last_update) > COAST_TIME_LIMIT or self.existence_prob < 0.2:
            self.status = "DELETED"
        elif self.status == "CONFIRMED":
            # Doğrulanmış bir track, birkaç ilişkisiz radar olayı yüzünden tekrar
            # TENTATIVE yapılmaz; yalnızca coast/olasılık silme koşulu sonlandırır.
            return
        elif (
            self.existence_prob > 0.90
            and self.hits_count >= CONFIRM_HITS
            and (self.time - self.creation_time) >= MIN_CONFIRM_AGE_S
            and len(self.source_radar_names) >= MIN_CONFIRM_SENSORS
        ):
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"

    def should_emit(self):
        """Yalnızca yakın zamanda ölçümle desteklenmiş confirmed track'i yayınla."""
        return (
            self.status == "CONFIRMED"
            and (self.time - self.last_update) <= CONFIRMED_OUTPUT_COAST_S
        )

class FusionCenter:
    def __init__(self, use_ci=True, adaptive_q_enabled=ADAPTIVE_Q_ENABLED):
        self.tracks: List[GlobalTrack] = []
        self.src_map: Dict[Tuple, str] = {}  
        self.use_ci = use_ci
        self.adaptive_q_enabled = adaptive_q_enabled

    def _tracks_are_duplicate(self, t1, t2, chi2_thresh=16.0):
        """
        İki track'in konumlarını Mahalanobis, hızlarını Öklid ile kıyaslar.
        chi2_thresh=16.0 yaklaşık %99+ güven aralığına denk gelir (2 serbestlik derecesi için).
        """
        # 1. Konum için Mahalanobis Mesafesi (0: x, 3: y, 6: z)
        dx = np.array([
            [float(t1.state[0,0]) - float(t2.state[0,0])],
            [float(t1.state[3,0]) - float(t2.state[3,0])],
            [float(t1.state[6,0]) - float(t2.state[6,0])],
        ])
        
        # Inovasyon Kovaryansı (İki track'in pozisyon belirsizliklerinin toplamı)
        P_sum = t1.cov[np.ix_([0,3,6],[0,3,6])] + t2.cov[np.ix_([0,3,6],[0,3,6])]
        
        try:
            d2 = float((dx.T @ np.linalg.inv(P_sum) @ dx).item())
        except np.linalg.LinAlgError:
            return False

        # 2. Hız için basit Öklid Mesafesi (1: vx, 4: vy, 7: vz)
        dvx = float(t1.state[1,0]) - float(t2.state[1,0])
        dvy = float(t1.state[4,0]) - float(t2.state[4,0])
        dvz = float(t1.state[7,0]) - float(t2.state[7,0])
        vel_diff = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

        return d2 < chi2_thresh and vel_diff <= DUPLICATE_VEL_MPS

    def _merge_duplicates(self, current_time):
        """Birbirine çok yakın (konum ve hız) ve paralel ilerleyen track'leri birleştirir."""
        confirmed_tracks = [gt for gt in self.tracks if gt.status == "CONFIRMED"]
        tentative_tracks = [gt for gt in self.tracks if gt.status == "TENTATIVE"]
        to_delete = set()
        
        # ==========================================
        # 1. Aşama: CONFIRMED <-> CONFIRMED Kontrolü
        # ==========================================
        for i in range(len(confirmed_tracks)):
            for j in range(i + 1, len(confirmed_tracks)):
                t1, t2 = confirmed_tracks[i], confirmed_tracks[j]
                
                if t1.id in to_delete or t2.id in to_delete:
                    continue
                    
                if self._tracks_are_duplicate(t1, t2, chi2_thresh=16.0):
                    # Hangisini tutacağımızı seç (hit sayısı veya olasılığa göre)
                    if t1.hits_count > t2.hits_count or (t1.hits_count == t2.hits_count and t1.existence_prob >= t2.existence_prob):
                        keeper, weaker = t1, t2
                    else:
                        keeper, weaker = t2, t1
                        
                    # Zayıfın verilerini güçlüye devret
                    keeper.sources.update(weaker.sources)
                    keeper.source_radar_names.update(weaker.source_radar_names)
                    keeper.source_measurement_details.update(weaker.source_measurement_details)
                    
                    for src_key, track_id in list(self.src_map.items()):
                        if track_id == weaker.id:
                            self.src_map[src_key] = keeper.id
                            
                    
                    to_delete.add(weaker.id)

        # ==========================================
        # 2. Aşama: CONFIRMED <-> TENTATIVE Kontrolü (Erken Temizlik)
        # ==========================================
        for c in confirmed_tracks:
            if c.id in to_delete:
                continue
            for t in tentative_tracks:
                if t.id in to_delete:
                    continue

                if self._tracks_are_duplicate(c, t, chi2_thresh=16.0):
                    # TENTATIVE her zaman silinir, CONFIRMED tutulur
                    c.sources.update(t.sources)
                    c.source_radar_names.update(t.source_radar_names)
                    c.source_measurement_details.update(t.source_measurement_details)

                    for src_key, track_id in list(self.src_map.items()):
                        if track_id == t.id:
                            self.src_map[src_key] = c.id

                    
                    to_delete.add(t.id)

        # Silinecekleri ana listeden çıkar
        self.tracks = [gt for gt in self.tracks if gt.id not in to_delete]

    def process_batch(self, t, measurements):
        for gt in self.tracks: gt.propagate(t)
        deleted = {gt.id for gt in self.tracks if gt.status == "DELETED"}
        self.tracks = [gt for gt in self.tracks if gt.id not in deleted]
        self.src_map = {k: v for k, v in self.src_map.items() if v not in deleted}

        gt_by_id = {gt.id: i for i, gt in enumerate(self.tracks)}
        matched_gt, matched_m = set(), set()
        
        # Gözlem (Observation) Matrisi H: 9D Durumu 6D ölçüme iz düşürür
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # vx
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # z
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # vz
        ])

        for mi, m in enumerate(measurements):
            gid = self.src_map.get(m["src"])
            if gid is None or gid not in gt_by_id: continue
            gi = gt_by_id[gid]
            if gi in matched_gt: continue
            gt = self.tracks[gi]
            
            # H ile iz düşüm hesaplaması
            S = H @ gt.cov @ H.T + m["cov"]
            diff = H @ gt.state - m["state"]
            
            try:
                try:
                    solved = np.linalg.solve(S, diff)
                except np.linalg.LinAlgError:
                    solved = np.linalg.pinv(S) @ diff
                nis = float((diff.T @ solved).item())
                if (
                    np.isfinite(nis)
                    and nis >= 0.0
                    and nis < gt.get_effective_gate()
                    and _passes_physical_association_gates(gt, diff, S)
                ):
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    gt.update_maneuver_adaptation(nis)
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi); matched_m.add(mi)
            except np.linalg.LinAlgError:
                continue

        r_gt = [i for i in range(len(self.tracks)) if i not in matched_gt]
        r_m  = [i for i in range(len(measurements)) if i not in matched_m]
        if r_gt and r_m:
            cost = np.full((len(r_gt), len(r_m)), 1e9) 
            accepted_nis = np.full((len(r_gt), len(r_m)), np.nan)
            for ri, gi in enumerate(r_gt):
                gt = self.tracks[gi]
                for ci, mi in enumerate(r_m):
                    m = measurements[mi]
                    S = H @ gt.cov @ H.T + m["cov"]
                    diff = H @ gt.state - m["state"]
                    try:
                        _, ld = np.linalg.slogdet(S)
                        try:
                            solved = np.linalg.solve(S, diff)
                        except np.linalg.LinAlgError:
                            solved = np.linalg.pinv(S) @ diff
                        d2 = float((diff.T @ solved).item())
                        if (
                            np.isfinite(d2)
                            and d2 >= 0.0
                            and d2 < gt.get_effective_gate()
                            and _passes_physical_association_gates(gt, diff, S)
                        ):
                            cost[ri, ci] = d2 + max(0, ld)
                            accepted_nis[ri, ci] = d2
                    except np.linalg.LinAlgError:
                        pass

            if not np.all(cost == 1e9):
                for ri, ci in zip(*linear_sum_assignment(cost)):
                    if cost[ri, ci] >= 1e9: continue 
                    
                    gi, mi = r_gt[ri], r_m[ci]
                    gt = self.tracks[gi]
                    m = measurements[mi]
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    gt.update_maneuver_adaptation(accepted_nis[ri, ci])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi); matched_m.add(mi)
                    
        for mi, m in enumerate(measurements):
            if mi in matched_m: continue
            ng = GlobalTrack(
                t, m["state"], m["cov"], m["tq"], m["src"],
                use_ci=self.use_ci,
                adaptive_q_enabled=self.adaptive_q_enabled,
            )
            self.tracks.append(ng)
            self.src_map[m["src"]] = ng.id

        # --- YENI ADIM: Döngü sonunda duplicate track'leri temizle/birleştir ---
        self._merge_duplicates(t)

# ===========================================================================
# ANA YURUTME
# ===========================================================================
def run_advanced_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
    adaptive_q_enabled=ADAPTIVE_Q_ENABLED,
) -> pd.DataFrame:
    if verbose:
        mode_str = "Covariance Intersection (CI)" if use_ci else "Standard LMMSE (No-CI)"
        q_mode = "Adaptive Q" if adaptive_q_enabled else "Fixed Q"
        print(f"{mode_str} + {q_mode} füzyon çalıştırılıyor: {sensor_csv}")

    sensor_df = pd.read_csv(sensor_csv)
    fusion_input = sensor_df.copy()
    GlobalTrack._cnt = 0
    fc = FusionCenter(
        use_ci=use_ci,
        adaptive_q_enabled=adaptive_q_enabled,
    )
    output_records = []

    for t_val, group in fusion_input.groupby("time", sort=True):
        measurements = []
        for _, row in group.iterrows():
            state = np.array([
                row["x"],
                row["vx"],
                row["y"],
                row["vy"],
                row.get("z", 0.0),
                row.get("vz", 0.0),
            ]).reshape(6, 1)
            cov = measurement_cov_from_row(row)
            tq = row.get("track_quality", TQ_MAX)
            measurements.append({
                "state": state,
                "cov": cov,
                "tq": tq,
                "src": (row["sensor"], row["local_track_id"]),
            })
        fc.process_batch(t_val, measurements)

        for gt in fc.tracks:
            if gt.should_emit():
                
                # Eski hatalı ve sınırlı olan duplicate check (mesafe/history kontrolü) 
                # buradan silinmiştir. Bu işlem artık FusionCenter içindeki
                # _merge_duplicates fonksiyonunda çok daha güvenli yapılmaktadır.

                # 9D endeksler: x,vx,ax,y,vy,ay,z,vz,az
                sigma_x = math.sqrt(max(float(gt.cov[0, 0]), 1e-6))
                sigma_vx = math.sqrt(max(float(gt.cov[1, 1]), 1e-6))
                sigma_y = math.sqrt(max(float(gt.cov[3, 3]), 1e-6))
                sigma_vy = math.sqrt(max(float(gt.cov[4, 4]), 1e-6))
                sigma_z = math.sqrt(max(float(gt.cov[6, 6]), 1e-6))
                sigma_vz = math.sqrt(max(float(gt.cov[7, 7]), 1e-6))
                source_names = ", ".join(sorted(gt.source_radar_names))
                
                output_records.append({
                    "time": t_val,
                    "global_track_id": gt.id,
                    "x": float(gt.state[0, 0]),
                    "y": float(gt.state[3, 0]),
                    "z": float(gt.state[6, 0]),
                    "vx": float(gt.state[1, 0]),
                    "vy": float(gt.state[4, 0]),
                    "vz": float(gt.state[7, 0]),
                    "ax": float(gt.state[2, 0]),
                    "ay": float(gt.state[5, 0]),
                    "az": float(gt.state[8, 0]),
                    "pos_sigma_m": sigma_x,
                    "vel_sigma_mps": sigma_vx,
                    "sigma_x_m": sigma_x,
                    "sigma_vx_mps": sigma_vx,
                    "sigma_y_m": sigma_y,
                    "sigma_vy_mps": sigma_vy,
                    "sigma_z_m": sigma_z,
                    "sigma_vz_mps": sigma_vz,
                    "fused_tq": sigma_pos_to_tq(sigma_x),
                    "prob": round(gt.existence_prob, 3),
                    "n_sources": len(gt.source_radar_names),
                    "source_radars": source_names,
                    "source_measurement_details": "; ".join(sorted(gt.source_measurement_details)),
                })

    fused_df = pd.DataFrame(output_records)
    fused_df.to_csv(output_csv, index=False)
    
    if verbose:
        if not fused_df.empty:
            print(f"Füzyon tamamlandı. {len(fused_df)} CONFIRMED kayıt bulundu.")
        else:
            print(f"[!] Hiç CONFIRMED track oluşamadı. Boş CSV oluşturuldu: {output_csv}")
        print(f"Çıktı dosyası: {output_csv}")

    return fused_df


if __name__ == "__main__":
    run_advanced_fusion()
