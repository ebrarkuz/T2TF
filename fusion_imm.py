"""
fusion_imm3.py
==============
Gorevi: radar_sensor_tracks.csv uzerinden UC MODELLI IMM (Interacting Multiple
Model) tabanli track-to-track fuzyon calistirip fused_tracks_imm3.csv
cikisi olusturmaktir.

Bu dosya fusion_imm.py'nin (2 modelli: CV+CA) UZERINE, rapordaki 5.4.5
bolumunde tarif edilen "Uc Modelli Genisleme" ile insa edilmistir:

    CV       : Duz ucus / dogrusal hareket. Ivme ~ 0.
    CA_LOW   : Yumusak, kademeli donusler. DUSUK surec gurultusu (jerk).
    CA_HIGH  : Ani, keskin/asabi manevralar. YUKSEK surec gurultusu (jerk).
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
CV_PROCESS_NOISE_INTENSITY = 0.4
CV_ACCEL_LEAK_Q = 1e-4

CA_LOW_PROCESS_NOISE_INTENSITY = 0.8
CA_HIGH_PROCESS_NOISE_INTENSITY = 15.0

GATE_CHI2_4DOF = 25.0 
COAST_TIME_LIMIT = 30.0
CONFIRM_HITS = 3
DUPLICATE_DIST_M = 400.0
DUPLICATE_VEL_MPS = 150.0
DUPLICATE_TIME_S = 5.0

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5

MODEL_NAMES = ["CV", "CA_LOW", "CA_HIGH"]
TRANS_PROB = np.array([
    # ->CV    ->CA_LOW  ->CA_HIGH
    [0.80,    0.15,     0.05],   # CV
    [0.10,    0.75,     0.15],   # CA_LOW
    [0.05,    0.15,     0.80],   # CA_HIGH
])
INIT_MODE_PROB = np.array([0.85, 0.12, 0.03]) 

SENSOR_CSV = "radar_sensor_tracks.csv"
OUTPUT_FUSED_CSV = "fused_tracks_imm3.csv"

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
    return np.diag([sp**2, sv**2, sp**2, sv**2])

def sigma_pos_to_tq(sigma_pos: float) -> int:
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))

def measurement_cov_from_row(row) -> np.ndarray:
    if "sigma_pos_m" in row and "sigma_vel_mps" in row:
        sp = float(row["sigma_pos_m"])
        sv = float(row["sigma_vel_mps"])
        return np.diag([sp**2, sv**2, sp**2, sv**2])
    return tq_to_cov(row["track_quality"])

def _pad_4d_to_6d(state_4d, cov_4d):
    state_6d = np.zeros((6, 1))
    state_6d[0, 0] = state_4d[0, 0]  # x
    state_6d[1, 0] = state_4d[1, 0]  # vx
    state_6d[3, 0] = state_4d[2, 0]  # y
    state_6d[4, 0] = state_4d[3, 0]  # vy

    cov_6d = np.eye(6) * 1e5
    cov_6d[0:2, 0:2] = cov_4d[0:2, 0:2]
    cov_6d[3:5, 3:5] = cov_4d[2:4, 2:4]
    return state_6d, cov_6d

H_MEAS = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
])

# ===========================================================================
# MODEL DINAMIKLERI: F ve Q URETICILERI
# ===========================================================================
def _F_Q_ca(dt: float, process_noise_intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, dt, 0.5 * dt**2, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5 * dt**2],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1],
    ])
    q = process_noise_intensity ** 2
    dt2, dt3, dt4 = dt**2, dt**3, dt**4
    qb = q * np.array([
        [dt4 / 4, dt3 / 2, dt2 / 2],
        [dt3 / 2, dt2, dt],
        [dt2 / 2, dt, 1],
    ])
    Q = np.zeros((6, 6))
    Q[np.ix_([0, 1, 2], [0, 1, 2])] = qb
    Q[np.ix_([3, 4, 5], [3, 4, 5])] = qb
    return F, Q

def _F_Q_ca_low(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    return _F_Q_ca(dt, CA_LOW_PROCESS_NOISE_INTENSITY)

def _F_Q_ca_high(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    return _F_Q_ca(dt, CA_HIGH_PROCESS_NOISE_INTENSITY)

def _F_Q_cv(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, dt, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    q = CV_PROCESS_NOISE_INTENSITY ** 2
    dt2, dt3 = dt**2, dt**3
    qb_posvel = q * np.array([
        [dt3 / 3, dt2 / 2],
        [dt2 / 2, dt],
    ])
    Q = np.zeros((6, 6))
    Q[np.ix_([0, 1], [0, 1])] = qb_posvel
    Q[2, 2] = CV_ACCEL_LEAK_Q
    Q[np.ix_([3, 4], [3, 4])] = qb_posvel
    Q[5, 5] = CV_ACCEL_LEAK_Q
    return F, Q

MODEL_FQ = {"CV": _F_Q_cv, "CA_LOW": _F_Q_ca_low, "CA_HIGH": _F_Q_ca_high}

# ===========================================================================
# FUZYON (OLCUM BIRLESTIRME) FONKSIYONLARI
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
    try:
        S = P1 + P2
        S_inv = np.linalg.inv(S)
        K = P1 @ S_inv
        xf = x1 + K @ (x2 - x1)
        Pf = P1 - K @ P1
        Pf = 0.5 * (Pf + Pf.T)
        return xf, Pf
    except np.linalg.LinAlgError:
        return x1, P1

def _gaussian_likelihood(diff: np.ndarray, S: np.ndarray) -> float:
    try:
        S_inv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e-12
        d2 = float((diff.T @ S_inv @ diff).item())
        k = diff.shape[0]
        log_lik = -0.5 * (d2 + logdet + k * math.log(2 * math.pi))
        return math.exp(max(log_lik, -700.0))
    except np.linalg.LinAlgError:
        return 1e-12

# ===========================================================================
# IMM GLOBAL TRACK 
# ===========================================================================
class GlobalTrackIMM3:
    _cnt = 0
    def __init__(self, t, state_4d, cov_4d, tq, src, use_ci=True):
        GlobalTrackIMM3._cnt += 1
        self.id = f"GT-{GlobalTrackIMM3._cnt:04d}"
        self.time = t
        self.use_ci = use_ci
        state_6d, cov_6d = _pad_4d_to_6d(state_4d, cov_4d)
        self.model_state: Dict[str, np.ndarray] = {m: state_6d.copy() for m in MODEL_NAMES}
        self.model_cov: Dict[str, np.ndarray] = {m: cov_6d.copy() for m in MODEL_NAMES}
        self.mode_prob: Dict[str, float] = dict(zip(MODEL_NAMES, INIT_MODE_PROB))
        self.state, self.cov = self._combine()
        self.last_update = t
        self.existence_prob = 0.1 + 0.7 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.status = "TENTATIVE"
        self.sources: set = {src}
        self.source_radar_names: set = {src[0] if isinstance(src, tuple) else src}
        self.source_measurement_details: set = {f"{src[0] if isinstance(src, tuple) else src}@{t:.2f}"}
        self.hits_count = 1
        self.creation_time = t
        self.position_history = [(float(self.state[0, 0]), float(self.state[3, 0]), float(t))]

    def _combine(self):
        x_comb = np.zeros((6, 1))
        for m in MODEL_NAMES:
            x_comb += self.mode_prob[m] * self.model_state[m]
        P_comb = np.zeros((6, 6))
        for m in MODEL_NAMES:
            dx = self.model_state[m] - x_comb
            P_comb += self.mode_prob[m] * (self.model_cov[m] + dx @ dx.T)
        return x_comb, 0.5 * (P_comb + P_comb.T)

    def propagate(self, t):
        dt = t - self.time
        if dt <= 0:
            return
        mu = np.array([self.mode_prob[m] for m in MODEL_NAMES])
        c_bar = TRANS_PROB.T @ mu
        c_bar = np.clip(c_bar, 1e-12, None)
        mix_w = (TRANS_PROB * mu[:, None]) / c_bar[None, :] 
        mixed_state, mixed_cov = {}, {}
        for j, mj in enumerate(MODEL_NAMES):
            x0j = np.zeros((6, 1))
            for i, mi in enumerate(MODEL_NAMES):
                x0j += mix_w[i, j] * self.model_state[mi]
            P0j = np.zeros((6, 6))
            for i, mi in enumerate(MODEL_NAMES):
                dx = self.model_state[mi] - x0j
                P0j += mix_w[i, j] * (self.model_cov[mi] + dx @ dx.T)
            mixed_state[mj] = x0j
            mixed_cov[mj] = 0.5 * (P0j + P0j.T)
        for m in MODEL_NAMES:
            F, Q = MODEL_FQ[m](dt)
            self.model_state[m] = F @ mixed_state[m]
            self.model_cov[m] = 0.5 * ((F @ mixed_cov[m] @ F.T + Q) + (F @ mixed_cov[m] @ F.T + Q).T)
        self.mode_prob = dict(zip(MODEL_NAMES, c_bar))
        self.state, self.cov = self._combine()
        self.time = t
        self.existence_prob *= 0.99
        self._update_status()

    def update(self, meas_state_4d, meas_cov_4d, tq, src):
        m_state_6d, m_cov_6d = _pad_4d_to_6d(meas_state_4d, meas_cov_4d)
        likelihoods = {}
        for m in MODEL_NAMES:
            x_pred, P_pred = self.model_state[m], self.model_cov[m]
            S = H_MEAS @ P_pred @ H_MEAS.T + meas_cov_4d
            diff = meas_state_4d - (H_MEAS @ x_pred)
            likelihoods[m] = _gaussian_likelihood(diff, S)
            if self.use_ci:
                xf, Pf = _ci_fuse(x_pred, P_pred, m_state_6d, m_cov_6d)
            else:
                xf, Pf = _standard_fuse(x_pred, P_pred, m_state_6d, m_cov_6d)
            self.model_state[m], self.model_cov[m] = xf, Pf
        c_bar = np.array([self.mode_prob[m] for m in MODEL_NAMES])
        raw = c_bar * np.array([likelihoods[m] for m in MODEL_NAMES])
        total = raw.sum()
        if total <= 0 or not np.isfinite(total):
            new_mu = c_bar
        else:
            new_mu = raw / total
        floor = 1e-3
        new_mu = np.clip(new_mu, floor, 1 - floor)
        new_mu = new_mu / new_mu.sum()
        self.mode_prob = dict(zip(MODEL_NAMES, new_mu))
        self.state, self.cov = self._combine()
        self.last_update = self.time
        mp = 0.5 + 0.45 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.existence_prob = self.existence_prob + (1 - self.existence_prob) * mp
        self.sources.add(src)
        self.source_radar_names.add(src[0] if isinstance(src, tuple) else src)
        self.source_measurement_details.add(f"{src[0] if isinstance(src, tuple) else src}@{self.time:.2f}")
        self.hits_count += 1
        self.position_history.append((float(self.state[0, 0]), float(self.state[3, 0]), float(self.time)))
        self._update_status()

    def _update_status(self):
        if (self.time - self.last_update) > COAST_TIME_LIMIT or self.existence_prob < 0.2:
            self.status = "DELETED"
        elif self.existence_prob > 0.85 and self.hits_count >= CONFIRM_HITS:
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"

    def dominant_model(self) -> str:
        return max(MODEL_NAMES, key=lambda m: self.mode_prob[m])


# ===========================================================================
# FUZYON MERKEZI 
# ===========================================================================
class FusionCenterIMM3:
    def __init__(self, use_ci=True, verbose=True):
        self.tracks: List[GlobalTrackIMM3] = []
        self.src_map: Dict[Tuple, str] = {}
        self.use_ci = use_ci
        self.verbose = verbose
        # [YENI]: Kopan veya silinen hedeflerin belirli bir süre tutulduğu hafıza
        self.recently_deleted: List[dict] = [] 

    def _tracks_are_duplicate(self, t1, t2, chi2_thresh=16.0):
        dx = np.array([[float(t1.state[0, 0]) - float(t2.state[0, 0])],
                       [float(t1.state[3, 0]) - float(t2.state[3, 0])]])
        P_sum = t1.cov[np.ix_([0, 3], [0, 3])] + t2.cov[np.ix_([0, 3], [0, 3])]
        try:
            d2 = float((dx.T @ np.linalg.inv(P_sum) @ dx).item())
        except np.linalg.LinAlgError:
            return False
        dvx = float(t1.state[1, 0]) - float(t2.state[1, 0])
        dvy = float(t1.state[4, 0]) - float(t2.state[4, 0])
        vel_diff = math.hypot(dvx, dvy)
        return d2 < chi2_thresh and vel_diff <= DUPLICATE_VEL_MPS

    def _merge_duplicates(self, current_time):
        confirmed_tracks = [gt for gt in self.tracks if gt.status == "CONFIRMED"]
        tentative_tracks = [gt for gt in self.tracks if gt.status == "TENTATIVE"]
        to_delete = set()
        for i in range(len(confirmed_tracks)):
            for j in range(i + 1, len(confirmed_tracks)):
                t1, t2 = confirmed_tracks[i], confirmed_tracks[j]
                if t1.id in to_delete or t2.id in to_delete:
                    continue
                if self._tracks_are_duplicate(t1, t2, chi2_thresh=36.0):
                    if t1.hits_count > t2.hits_count or (t1.hits_count == t2.hits_count and t1.existence_prob >= t2.existence_prob):
                        keeper, weaker = t1, t2
                    else:
                        keeper, weaker = t2, t1
                    keeper.sources.update(weaker.sources)
                    keeper.source_radar_names.update(weaker.source_radar_names)
                    keeper.source_measurement_details.update(weaker.source_measurement_details)
                    for src_key, track_id in list(self.src_map.items()):
                        if track_id == weaker.id:
                            self.src_map[src_key] = keeper.id
                    if self.verbose:
                        print(f"[MERGE CONFIRMED] t={current_time:.1f} | SILINEN: {weaker.id} ({weaker.hits_count} hit) -> TUTULAN: {keeper.id} ({keeper.hits_count} hit)")
                    to_delete.add(weaker.id)
        for c in confirmed_tracks:
            if c.id in to_delete:
                continue
            for t in tentative_tracks:
                if t.id in to_delete:
                    continue
                if self._tracks_are_duplicate(c, t, chi2_thresh=36.0):
                    c.sources.update(t.sources)
                    c.source_radar_names.update(t.source_radar_names)
                    c.source_measurement_details.update(t.source_measurement_details)
                    for src_key, track_id in list(self.src_map.items()):
                        if track_id == t.id:
                            self.src_map[src_key] = c.id
                        
                    to_delete.add(t.id)
        self.tracks = [gt for gt in self.tracks if gt.id not in to_delete]

    def process_batch(self, t, measurements):
        for gt in self.tracks:
            gt.propagate(t)
            
        # [YENI]: Silinecek olan track'leri son silinenler listesine tası (Hafıza limit: 10 sn)
        deleted_tracks = [gt for gt in self.tracks if gt.status == "DELETED"]
        deleted_ids = {gt.id for gt in deleted_tracks}
        for gt in deleted_tracks:
            self.recently_deleted.append({
                'delete_time': t,
                'track': gt
            })
            
        # 10 saniyeden daha uzun süredir kayıp olanları geri döndürülemez şekilde sil
        self.recently_deleted = [rd for rd in self.recently_deleted if (t - rd['delete_time']) <= 10.0]

        self.tracks = [gt for gt in self.tracks if gt.id not in deleted_ids]
        self.src_map = {k: v for k, v in self.src_map.items() if v not in deleted_ids}

        gt_by_id = {gt.id: i for i, gt in enumerate(self.tracks)}
        matched_gt, matched_m = set(), set()

        # Adım 1: Mevcut aktif trackler ile yeni ölçümleri eşleştir (Gated Matching)
        for mi, m in enumerate(measurements):
            gid = self.src_map.get(m["src"])
            if gid is None or gid not in gt_by_id:
                continue
            gi = gt_by_id[gid]
            if gi in matched_gt:
                continue
            gt = self.tracks[gi]
            S = H_MEAS @ gt.cov @ H_MEAS.T + m["cov"]
            diff = H_MEAS @ gt.state - m["state"]
            try:
                gate = GATE_CHI2_4DOF * (2.0 if gt.dominant_model() == "CA_HIGH" else 1.0)
                if float((diff.T @ np.linalg.inv(S) @ diff).item()) < gate:
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi)
                    matched_m.add(mi)
            except np.linalg.LinAlgError:
                continue

        # Adım 2: Eşleşmeyenleri birbirleriyle küresel olarak eşleştir (Global Assignment)
        r_gt = [i for i in range(len(self.tracks)) if i not in matched_gt]
        r_m = [i for i in range(len(measurements)) if i not in matched_m]
        if r_gt and r_m:
            cost = np.full((len(r_gt), len(r_m)), 1e9)
            for ri, gi in enumerate(r_gt):
                gt = self.tracks[gi]
                for ci, mi in enumerate(r_m):
                    m = measurements[mi]
                    S = H_MEAS @ gt.cov @ H_MEAS.T + m["cov"]
                    diff = H_MEAS @ gt.state - m["state"]
                    try:
                        Si = np.linalg.inv(S)
                        _, ld = np.linalg.slogdet(S)
                        d2 = float((diff.T @ Si @ diff).item())
                        gate = GATE_CHI2_4DOF * (2.0 if gt.dominant_model() == "CA_HIGH" else 1.0)
                        if d2 < gate:
                            cost[ri, ci] = d2 + max(0, ld)
                    except np.linalg.LinAlgError:
                        pass
            if not np.all(cost == 1e9):
                for ri, ci in zip(*linear_sum_assignment(cost)):
                    if cost[ri, ci] >= 1e9:
                        continue
                    gi, mi = r_gt[ri], r_m[ci]
                    gt = self.tracks[gi]
                    m = measurements[mi]
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi)
                    matched_m.add(mi)

        # Adım 3: Hiçbir aktif track'e atanamayan ölçümler için (Revive veya Create)
        for mi, m in enumerate(measurements):
            if mi in matched_m:
                continue
                
            # [YENI]: Sıfırdan track açmadan önce, kopan/silinen track'lerin kinematik devamı mı diye kontrol et
            best_rd_idx = -1
            best_dist = 1e9
            
            for idx, rd in enumerate(self.recently_deleted):
                old_gt = rd['track']
                dt_gap = t - rd['delete_time']
                
                # Sabit hız varsayımıyla hedefin mevcut boşlukta ne kadar ilerlediğini tahmin et
                pred_x = float(old_gt.state[0, 0]) + float(old_gt.state[1, 0]) * dt_gap
                pred_y = float(old_gt.state[3, 0]) + float(old_gt.state[4, 0]) * dt_gap
                
                meas_x = float(m["state"][0, 0])
                meas_y = float(m["state"][2, 0])
                dist = math.hypot(pred_x - meas_x, pred_y - meas_y)
                
                dvx = float(old_gt.state[1, 0]) - float(m["state"][1, 0])
                dvy = float(old_gt.state[4, 0]) - float(m["state"][3, 0])
                vel_diff = math.hypot(dvx, dvy)
                
                # Biraz esnek tolerans (Hedef kopuk sürede manevra yapmış olabilir: ~300m, 45m/s tolerans)
                if dist < (DUPLICATE_DIST_M * 2.0) and vel_diff <= (DUPLICATE_VEL_MPS * 1.5):
                    if dist < best_dist:
                        best_dist = dist
                        best_rd_idx = idx

            if best_rd_idx != -1:
                # Eşleşti! Silinen track'i geri yükle
                rd_entry = self.recently_deleted.pop(best_rd_idx)
                revived_gt = rd_entry['track']
                
                # Önemli: Filtreyi koptuğu sürede körleme olarak matematiksel ilerlet ki kovaryansı doğru büyüsün
                revived_gt.propagate(t)
                
                # Canlanma sonrası status & güven güncellemesi
                revived_gt.status = "TENTATIVE" 
                revived_gt.existence_prob = 0.5 + 0.45 * ((m["tq"] - TQ_MIN) / (TQ_MAX - TQ_MIN))
                revived_gt.update(m["state"], m["cov"], m["tq"], m["src"])
                
                self.tracks.append(revived_gt)
                self.src_map[m["src"]] = revived_gt.id
                
                if self.verbose:
                    print(f"[REVIVE] t={t:.1f} | Koptuğu sanılan hedef yakalandı: {revived_gt.id} (Korunan Hit: {revived_gt.hits_count})")
            else:
                # Son silinenlerle de eşleşmedi, yepyeni bir track açılıyor
                ng = GlobalTrackIMM3(t, m["state"], m["cov"], m["tq"], m["src"], use_ci=self.use_ci)
                self.tracks.append(ng)
                self.src_map[m["src"]] = ng.id

        self._merge_duplicates(t)


# ===========================================================================
# ANA YURUTME
# ===========================================================================
def run_imm3_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
    include_source_measurement_details=False,
) -> pd.DataFrame:
    if verbose:
        mode_str = "IMM (CV+CA_LOW+CA_HIGH) + Covariance Intersection" if use_ci else "IMM (CV+CA_LOW+CA_HIGH) + Standard LMMSE"
        print(f"{mode_str} fuzyon calistiriliyor: {sensor_csv}")

    sensor_df = pd.read_csv(sensor_csv)
    if "is_clutter" in sensor_df.columns:
        sensor_df = sensor_df[sensor_df["is_clutter"] != True]

    GlobalTrackIMM3._cnt = 0
    fc = FusionCenterIMM3(use_ci=use_ci, verbose=verbose)
    output_records = []

    for t_val, group in sensor_df.groupby("time", sort=True):
        measurements = []
        for _, row in group.iterrows():
            state = np.array([row["x"], row["vx"], row["y"], row["vy"]]).reshape(4, 1)
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
            if gt.status != "CONFIRMED":
                continue

            sigma_x = math.sqrt(max(float(gt.cov[0, 0]), 1e-6))
            sigma_vx = math.sqrt(max(float(gt.cov[1, 1]), 1e-6))
            sigma_y = math.sqrt(max(float(gt.cov[3, 3]), 1e-6))
            sigma_vy = math.sqrt(max(float(gt.cov[4, 4]), 1e-6))
            source_names = ", ".join(sorted(gt.source_radar_names))

            rec = {
                "time": t_val,
                "global_track_id": gt.id,
                "x": float(gt.state[0, 0]),
                "y": float(gt.state[3, 0]),
                "vx": float(gt.state[1, 0]),
                "vy": float(gt.state[4, 0]),
                "ax": float(gt.state[2, 0]),
                "ay": float(gt.state[5, 0]),
                "pos_sigma_m": sigma_x,
                "vel_sigma_mps": sigma_vx,
                "sigma_x_m": sigma_x,
                "sigma_vx_mps": sigma_vx,
                "sigma_y_m": sigma_y,
                "sigma_vy_mps": sigma_vy,
                "fused_tq": sigma_pos_to_tq(sigma_x),
                "prob": round(gt.existence_prob, 3),
                "mode_prob_cv": round(gt.mode_prob["CV"], 3),
                "mode_prob_ca_low": round(gt.mode_prob["CA_LOW"], 3),
                "mode_prob_ca_high": round(gt.mode_prob["CA_HIGH"], 3),
                "dominant_model": gt.dominant_model(),
                "n_sources": len(gt.source_radar_names),
                "source_radars": source_names,
            }
            if include_source_measurement_details:
                rec["source_measurement_details"] = "; ".join(sorted(gt.source_measurement_details))
            output_records.append(rec)

    fused_df = pd.DataFrame(output_records)
    fused_df.to_csv(output_csv, index=False)

    if verbose:
        if not fused_df.empty:
            print(f"IMM3 fuzyonu tamamlandi. {len(fused_df)} CONFIRMED kayit bulundu.")
            print("Ornek Cikti (mod olasiliklari dahil):")
            cols = ["time", "global_track_id", "x", "y",
                    "mode_prob_cv", "mode_prob_ca_low", "mode_prob_ca_high", "dominant_model"]
            print(fused_df[cols].tail(10).to_string(index=False))
        else:
            print(f"[!] Hic CONFIRMED track olusamadi. Bos CSV olusturuldu: {output_csv}")
        print(f"Cikti dosyasi: {output_csv}")

    return fused_df


def run_imm_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
    include_source_measurement_details=False,
) -> pd.DataFrame:
    return run_imm3_fusion(
        sensor_csv=sensor_csv,
        output_csv=output_csv,
        verbose=verbose,
        use_ci=use_ci,
        include_source_measurement_details=include_source_measurement_details,
    )

if __name__ == "__main__":
    run_imm3_fusion()