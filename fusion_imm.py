"""
fusion_imm.py
=============
Gorevi: radar_sensor_tracks.csv uzerinden IMM (Interacting Multiple Model)
tabanli track-to-track fuzyon calistirip fused_tracks_imm.csv cikisi olusturmaktir.

Mimari, fusion_basic.py / fusion_advanced.py (fuzyon2.py) ile ayni iskeleti
kullanir (association: src_map + Mahalanobis gate + Hungarian, track lifecycle:
TENTATIVE/CONFIRMED/DELETED). Farki, tek bir hareket modeli (CV ya da CA)
yerine IKI modelin (CV ve CA) IMM ile olasiliksal olarak karistirilmasidir.

Neden IMM?
----------
fusion_advanced.py'deki CA modeli notunda soyle deniyordu:
    "Ani, keskin manevralarda CA modeli de yetersiz kalabilir."
IMM tam olarak bu bosluğu kapatir: hedef duz gidiyorsa CV modeli agirlik
kazanir (dusuk gurultu, dusuk lag); hedef manevra yapiyorsa CA modeli
agirlik kazanir (ivmeyi durumun bir parcasi olarak takip eder). Sistem
bu iki model arasinda anlik olcum uyumuna (likelihood) gore gecis yapar.

Matematiksel temel (Bar-Shalom, Li, Kirubarajan - "Estimation with
Applications to Tracking and Navigation", Bolum 11):
  1) Mixing (Karistirma): Bir onceki adimin model olasiliklari ve model
     gecis matrisi (Pi) kullanilarak, her model icin "karisik baslangic
     kosulu" (mixed initial condition) hesaplanir.
  2) Model-matched filtering: Her model kendi F, Q matrisiyle bagimsiz
     olarak ileri sarilir (propagate) ve gelen olcumle guncellenir (update).
  3) Model probability update: Her modelin olcum ile ne kadar uyumlu
     oldugu (likelihood) hesaplanir, model olasiliklari Bayes kurallariyla
     guncellenir.
  4) Combination: Cikis icin (raporlama, association gate'i) modeller,
     olasiliklariyla agirlikli ortalanarak tek bir durum/kovaryansa
     birlestirilir. Ancak modellerin kendi ic durumlari ayri ayri
     saklanmaya devam eder (bu, klasik "tek model secimi" yaklasimindan
     temel farktir).

CV ve CA'nin ortak 6 boyutlu uzayda temsili:
---------------------------------------------
IMM'in mixing adimi, tum modellerin AYNI durum vektoru boyutunda olmasini
gerektirir. fusion_advanced.py'de CA modeli zaten 6 boyutlu:
    x = [x, vx, ax, y, vy, ay]^T
CV modelini de bu 6 boyutlu uzayda temsil ediyoruz, ancak:
  - Durum gecis matrisinde (F_cv) ivme, pozisyon/hiza dt^2/2 ve dt ile
    KATKI YAPMAZ (o terimler sifirlanir) -> pozisyon sadece hizla,
    hiz sabit kalarak ilerler (klasik CV davranisi).
  - Ivme bilesenleri F_cv'de F=1 ile "donuk" tasinir ve surec gurultusu
    (Q_cv) ivme icin kasitli olarak COK KUCUK tutulur -> ivme tahmini
    pratikte ~0 civarinda kalir ve CV modelini bozmaz.
Bu, literatürde standart bir tekniktir (Bar-Shalom Bolum 11.6): boylece
CV ve CA ayni 6D uzayda "yasar" ve mixing/combination matematiksel
olarak tutarli sekilde yapilabilir.
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
# CA modelinin surec gurultusu (fusion_advanced.py ile ayni - "jerk" siddeti)
CA_PROCESS_NOISE_INTENSITY = 1.5
# CV modelinin ivme bilesenleri icin kasitli olarak kucuk surec gurultusu
# (ivmeyi ~0'da tutmak icin). Pozisyon/hiz icin CV'nin kendi "white noise
# acceleration" siddeti.
CV_PROCESS_NOISE_INTENSITY = 0.4
CV_ACCEL_LEAK_Q = 1e-4          # ivme bileseninin sifirdan sapmasina izin verilen minik varyans

GATE_CHI2_4DOF = 9.0
COAST_TIME_LIMIT = 30.0
CONFIRM_HITS = 3
DUPLICATE_DIST_M = 50.0
DUPLICATE_TIME_S = 5.0

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5

# IMM Model Gecis Matrisi (Pi): satir=onceki model, sutun=yeni model
# [CV, CA] sirasiyla. Diyagonal agirlikli -> modeller "yapiskan" (ani
# sicramalar yerine kademeli gecis), ama manevraya makul hizda tepki verir.
MODEL_NAMES = ["CV", "CA"]
TRANS_PROB = np.array([
    [0.85, 0.15],   # CV -> CV, CV -> CA
    [0.25, 0.75],   # CA -> CV, CA -> CA
])
INIT_MODE_PROB = np.array([0.5, 0.5])  # baslangicta iki modele esit guven

SENSOR_CSV = "radar_sensor_tracks.csv"
OUTPUT_FUSED_CSV = "fused_tracks_imm.csv"

# ===========================================================================
# YARDIMCI FONKSIYONLAR (fusion_advanced.py ile ayni mantik)
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
    """4D olcumu 6D (ortak IMM) durumuna genisletir. Ivme varyansi devasa birakilir
    (ölçüm ivmeyi dogrudan bilmiyor, ivme bilesenleri filtre tarafindan ogrenilir)."""
    state_6d = np.zeros((6, 1))
    state_6d[0, 0] = state_4d[0, 0]  # x
    state_6d[1, 0] = state_4d[1, 0]  # vx
    state_6d[3, 0] = state_4d[2, 0]  # y
    state_6d[4, 0] = state_4d[3, 0]  # vy

    cov_6d = np.eye(6) * 1e5
    cov_6d[0:2, 0:2] = cov_4d[0:2, 0:2]
    cov_6d[3:5, 3:5] = cov_4d[2:4, 2:4]
    return state_6d, cov_6d


# Olcum uzayina izdusum matrisi (6D durum -> 4D [x, vx, y, vy] olcum)
H_MEAS = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
])


# ===========================================================================
# MODEL DINAMIKLERI: F ve Q URETICILERI
# ===========================================================================
def _F_Q_ca(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sabit Ivme (CA) modeli - fusion_advanced.py ile birebir ayni."""
    F = np.array([
        [1, dt, 0.5 * dt**2, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5 * dt**2],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1],
    ])
    q = CA_PROCESS_NOISE_INTENSITY ** 2
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


def _F_Q_cv(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sabit Hiz (CV) modeli, CA ile AYNI 6D uzayda temsil edilir.
    Ivme, pozisyon/hiza katkida bulunmaz (o terimler sifir); ivme
    bileseni kendi icinde 'donuk' tasinir ve minik surec gurultusuyle
    ~0 civarinda tutulur."""
    F = np.array([
        [1, dt, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],   # ivme F=1 ama pozisyon/hiza akmiyor
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


MODEL_FQ = {"CV": _F_Q_cv, "CA": _F_Q_ca}


# ===========================================================================
# FUZYON (OLCUM BIRLESTIRME) FONKSIYONLARI - fusion_advanced.py ile ayni
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
    """Olcum-tahmin farkinin (innovation) Gauss olabilirligi. Model
    olasiliklarinin guncellenmesinde kullanilir (Adim 3: model probability
    update)."""
    try:
        S_inv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e-12
        d2 = float((diff.T @ S_inv @ diff).item())
        k = diff.shape[0]
        log_lik = -0.5 * (d2 + logdet + k * math.log(2 * math.pi))
        return math.exp(max(log_lik, -700.0))  # alt tasma koruma
    except np.linalg.LinAlgError:
        return 1e-12


# ===========================================================================
# IMM GLOBAL TRACK
# ===========================================================================
class GlobalTrackIMM:
    _cnt = 0

    def __init__(self, t, state_4d, cov_4d, tq, src, use_ci=True):
        GlobalTrackIMM._cnt += 1
        self.id = f"GT-{GlobalTrackIMM._cnt:04d}"
        self.time = t
        self.use_ci = use_ci

        state_6d, cov_6d = _pad_4d_to_6d(state_4d, cov_4d)

        # Her model ayni baslangic durumuyla acilir; ayrim zamanla,
        # gelen olcumlerin her modeli farkli derecede "tatmin etmesiyle" olusur.
        self.model_state: Dict[str, np.ndarray] = {m: state_6d.copy() for m in MODEL_NAMES}
        self.model_cov: Dict[str, np.ndarray] = {m: cov_6d.copy() for m in MODEL_NAMES}
        self.mode_prob: Dict[str, float] = dict(zip(MODEL_NAMES, INIT_MODE_PROB))

        # Association / raporlama icin kullanilan, mod-agirlikli birlesik durum
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

    # ---------------------------------------------------------------
    def _combine(self):
        """Adim 4 (Combination): modelleri mod olasiliklarina gore
        agirlikli ortalayarak tek bir cikis durumu/kovaryansi uretir."""
        x_comb = np.zeros((6, 1))
        for m in MODEL_NAMES:
            x_comb += self.mode_prob[m] * self.model_state[m]

        P_comb = np.zeros((6, 6))
        for m in MODEL_NAMES:
            dx = self.model_state[m] - x_comb
            P_comb += self.mode_prob[m] * (self.model_cov[m] + dx @ dx.T)
        return x_comb, 0.5 * (P_comb + P_comb.T)

    # ---------------------------------------------------------------
    def propagate(self, t):
        """Adim 1 (Mixing) + Adim 2 (Model-matched predict)."""
        dt = t - self.time
        if dt <= 0:
            return

        mu = np.array([self.mode_prob[m] for m in MODEL_NAMES])
        # c_bar_j = sum_i Pi[i,j] * mu_i  (her hedef model icin normalize sabiti)
        c_bar = TRANS_PROB.T @ mu
        c_bar = np.clip(c_bar, 1e-12, None)

        # Karisim agirliklari mu_ij = Pi[i,j]*mu_i / c_bar_j
        mix_w = (TRANS_PROB * mu[:, None]) / c_bar[None, :]  # shape (i, j)

        mixed_state = {}
        mixed_cov = {}
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

        # Model-matched predict: her model kendi F,Q'suyla ileri sarilir
        for m in MODEL_NAMES:
            F, Q = MODEL_FQ[m](dt)
            self.model_state[m] = F @ mixed_state[m]
            self.model_cov[m] = 0.5 * ((F @ mixed_cov[m] @ F.T + Q) + (F @ mixed_cov[m] @ F.T + Q).T)

        self.mode_prob = dict(zip(MODEL_NAMES, c_bar))  # normalize edilmis onceki-olasilik tahmini
        self.state, self.cov = self._combine()
        self.time = t
        self.existence_prob *= 0.97
        self._update_status()

    # ---------------------------------------------------------------
    def update(self, meas_state_4d, meas_cov_4d, tq, src):
        """Adim 2 devami (her model kendi tahminini olcumle gunceller) +
        Adim 3 (model olasiliklari, olcum likelihood'una gore guncellenir)."""
        m_state_6d, m_cov_6d = _pad_4d_to_6d(meas_state_4d, meas_cov_4d)

        likelihoods = {}
        for m in MODEL_NAMES:
            x_pred, P_pred = self.model_state[m], self.model_cov[m]

            # Likelihood, olcum uzayindaki innovation uzerinden hesaplanir
            S = H_MEAS @ P_pred @ H_MEAS.T + meas_cov_4d
            diff = meas_state_4d - (H_MEAS @ x_pred)
            likelihoods[m] = _gaussian_likelihood(diff, S)

            # Durum guncellemesi CI ya da standart fuzyon ile (6D uzayda)
            if self.use_ci:
                xf, Pf = _ci_fuse(x_pred, P_pred, m_state_6d, m_cov_6d)
            else:
                xf, Pf = _standard_fuse(x_pred, P_pred, m_state_6d, m_cov_6d)
            self.model_state[m], self.model_cov[m] = xf, Pf

        # Bayes guncellemesi: mu_m ~ c_bar_m * likelihood_m
        c_bar = np.array([self.mode_prob[m] for m in MODEL_NAMES])
        raw = c_bar * np.array([likelihoods[m] for m in MODEL_NAMES])
        total = raw.sum()
        if total <= 0 or not np.isfinite(total):
            new_mu = c_bar  # olabilirlikler dejenere olursa onceki agirliklari koru
        else:
            new_mu = raw / total
        # Modellerden birinin tamamen "olmesini" engellemek icin taban degeri
        new_mu = np.clip(new_mu, 0.05, 0.95)
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

    # ---------------------------------------------------------------
    def _update_status(self):
        if (self.time - self.last_update) > COAST_TIME_LIMIT or self.existence_prob < 0.2:
            self.status = "DELETED"
        elif self.existence_prob > 0.85 and self.hits_count >= CONFIRM_HITS:
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"


# ===========================================================================
# FUZYON MERKEZI - association mantigi fusion_advanced.py ile ayni
# ===========================================================================
class FusionCenterIMM:
    def __init__(self, use_ci=True):
        self.tracks: List[GlobalTrackIMM] = []
        self.src_map: Dict[Tuple, str] = {}
        self.use_ci = use_ci

    def process_batch(self, t, measurements):
        for gt in self.tracks:
            gt.propagate(t)
        deleted = {gt.id for gt in self.tracks if gt.status == "DELETED"}
        self.tracks = [gt for gt in self.tracks if gt.id not in deleted]
        self.src_map = {k: v for k, v in self.src_map.items() if v not in deleted}

        gt_by_id = {gt.id: i for i, gt in enumerate(self.tracks)}
        matched_gt, matched_m = set(), set()

        # --- Adim: Hafizali hizli eslestirme (src_map + gate) ---
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
                if float((diff.T @ np.linalg.inv(S) @ diff).item()) < GATE_CHI2_4DOF:
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi)
                    matched_m.add(mi)
            except np.linalg.LinAlgError:
                continue

        # --- Adim: Global optimum eslestirme (Hungarian) ---
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
                        if d2 < GATE_CHI2_4DOF:
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

        # --- Adim: Yeni track acilisi ---
        for mi, m in enumerate(measurements):
            if mi in matched_m:
                continue
            ng = GlobalTrackIMM(t, m["state"], m["cov"], m["tq"], m["src"], use_ci=self.use_ci)
            self.tracks.append(ng)
            self.src_map[m["src"]] = ng.id


# ===========================================================================
# ANA YURUTME
# ===========================================================================
def run_imm_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
) -> pd.DataFrame:
    if verbose:
        mode_str = "IMM (CV+CA) + Covariance Intersection" if use_ci else "IMM (CV+CA) + Standard LMMSE"
        print(f"{mode_str} füzyon çalıştırılıyor: {sensor_csv}")

  # fusion_imm.py, satır 502-505 yerine:
    sensor_df = pd.read_csv(sensor_csv)
    # Clutter'ları doğrudan eleyelim
    if "is_clutter" in sensor_df.columns:
        sensor_df = sensor_df[sensor_df["is_clutter"] != True]
    GlobalTrackIMM._cnt = 0
    fc = FusionCenterIMM(use_ci=use_ci)
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

            # process_batch sonrası, CONFIRMED track listesi üzerinde çapraz kontrol
            confirmed = [gt for gt in fc.tracks if gt.status == "CONFIRMED"]
            to_delete = set()
            for i in range(len(confirmed)):
                for j in range(i + 1, len(confirmed)):
                    a, b = confirmed[i], confirmed[j]
                    dist = math.hypot(float(a.state[0,0]) - float(b.state[0,0]),
                                    float(a.state[3,0]) - float(b.state[3,0]))
                    if dist <= DUPLICATE_DIST_M:
                        # daha az hit'e / daha düşük existence_prob'a sahip olanı sil
                        weaker = a if a.hits_count < b.hits_count else b
                        to_delete.add(weaker.id)
            fc.tracks = [gt for gt in fc.tracks if gt.id not in to_delete]

            sigma_x = math.sqrt(max(float(gt.cov[0, 0]), 1e-6))
            sigma_vx = math.sqrt(max(float(gt.cov[1, 1]), 1e-6))
            sigma_y = math.sqrt(max(float(gt.cov[3, 3]), 1e-6))
            sigma_vy = math.sqrt(max(float(gt.cov[4, 4]), 1e-6))
            source_names = ", ".join(sorted(gt.source_radar_names))

            output_records.append({
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
                # IMM'e ozgu ek kolonlar: hangi modelin ne kadar agirlikta oldugu
                "mode_prob_cv": round(gt.mode_prob["CV"], 3),
                "mode_prob_ca": round(gt.mode_prob["CA"], 3),
                "dominant_model": "CA" if gt.mode_prob["CA"] > gt.mode_prob["CV"] else "CV",
                "n_sources": len(gt.source_radar_names),
                "source_radars": source_names,
                "source_measurement_details": "; ".join(sorted(gt.source_measurement_details)),
            })

    fused_df = pd.DataFrame(output_records)
    fused_df.to_csv(output_csv, index=False)

    if verbose:
        if not fused_df.empty:
            print(f"IMM füzyonu tamamlandı. {len(fused_df)} CONFIRMED kayıt bulundu.")
            print("Örnek Çıktı (mod olasılıkları dahil):")
            cols = ["time", "global_track_id", "x", "y", "mode_prob_cv", "mode_prob_ca", "dominant_model"]
            print(fused_df[cols].tail(10).to_string(index=False))
        else:
            print(f"[!] Hiç CONFIRMED track oluşamadı. Boş CSV oluşturuldu: {output_csv}")
        print(f"Çıktı dosyası: {output_csv}")

    return fused_df


if __name__ == "__main__":
    run_imm_fusion()