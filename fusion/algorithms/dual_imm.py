"""
fusion_imm3.py
==============
Gorevi: radar_sensor_tracks.csv uzerinden DEKUPLE DUAL-IMM tabanli
track-to-track fuzyon calistirip fused_tracks_imm3.csv cikisi olusturmaktir.

Mimari:
 - Yatay Eksen (X, Y): 4 Modlu IMM [CV, CA, CT_LEFT, CT_RIGHT]
 - Dikey Eksen (Z)   : 2 Modlu IMM [CV, SINGER]
Olcum ve Surec gürültüleri eksenlerin dogasina (manevra vs. tırmanma) ozel ayrıştırılmıştır.
"""

import math
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# ===========================================================================
# KONFIGURASYON & SABITLER
# ===========================================================================

# --- DIKEY (Z) IMM SABITLERI ---
CV_Z_PROCESS_NOISE_INTENSITY = 0.2
SINGER_TAU_Z = 10.0      # İvme korelasyon süresi (Pilotun tırmanmayı sürdürme süresi ~10 sn)
SINGER_SIGMA_Z = 2.0     # Dikey manevra standart sapması (m/s^2)

# --- AYRIK GATING (MAHALANOBIS) ESIKLERI ---
GATE_CONFIDENCE = 0.997
GATE_CHI2_4DOF = float(chi2.ppf(GATE_CONFIDENCE, df=4))
GATE_CHI2_2DOF = float(chi2.ppf(GATE_CONFIDENCE, df=2))
MANEUVER_GATE_MULTIPLIER = 1.4

COAST_TIME_LIMIT = 30.0
CONFIRM_HITS = 4
CONFIRM_EXISTENCE_THRESHOLD = 0.85
MIN_CONFIRM_RADARS = 2
SINGLE_RADAR_CONFIRM_HITS = 8
SINGLE_RADAR_CONFIRM_EXISTENCE = 0.95
EXISTENCE_DECAY_TAU_S = 30.0
MISS_DECAY_START_S = 5.0
MISS_DECAY_TAU_S = 15.0
MEASUREMENT_EXISTENCE_GAIN_MIN = 0.50
MEASUREMENT_EXISTENCE_GAIN_MAX = 0.95
SOURCE_MAP_TIMEOUT_S = 20.0
REVIVE_WINDOW_S = 10.0
REVIVE_DIST_XY_M = 400.0
REVIVE_DIST_Z_M = 1000.0
REVIVE_VEL_MPS = 80.0

# --- CIFT KAYIT (DUPLICATE) BIRLESTIRME ---
DUPLICATE_DIST_XY_M = 400.0
DUPLICATE_DIST_Z_M = 1000.0
DUPLICATE_VEL_MPS = 80.0
DUPLICATE_STRONG_MAHALANOBIS = 9.0

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5

ELEVATION_ERROR_RAD = 0.005       # Radarin irtifa acisi hatasi (orn: ~0.3 derece)
MISSING_Z_COVAR_PENALTY = 1e6     # Z verisi yoksa filtreyi korlemek icin devasa kovaryans
COV_MIN_EIG = 1e-6
XY_VARIANCE_FLOOR = np.array([25.0, 0.25, 0.01, 25.0, 0.25, 0.01])
Z_VARIANCE_FLOOR = np.array([25.0, 0.25, 0.01])
STATE9_VARIANCE_FLOOR = np.array([25.0, 0.25, 0.01, 25.0, 0.25, 0.01, 25.0, 0.25, 0.01])
INITIAL_ACCEL_VARIANCE = 100.0

# --- AYRIK IMM GECIS OLASILIKLARI VE MODLAR ---
# --- YATAY (XY) IMM SABITLERI ---
CV_XY_PROCESS_NOISE_INTENSITY = 0.4
CA_XY_PROCESS_NOISE_INTENSITY = 8.0      # Genel keskin manevralar için tek CA
CT_XY_PROCESS_NOISE_INTENSITY = 2.0      # Dönüş sırasındaki ufak sapmalar
CV_ACCEL_LEAK_Q = 1e-3

TURN_RATE_DEG_PER_SEC = 1.5             # Standart dönüş hızı (saniyede 3 derece)
OMEGA_LEFT = math.radians(TURN_RATE_DEG_PER_SEC)
OMEGA_RIGHT = math.radians(-TURN_RATE_DEG_PER_SEC)

# --- AYRIK IMM GECIS OLASILIKLARI VE MODLAR ---
MODELS_XY = ["CV", "CA", "CT_LEFT", "CT_RIGHT"]
TRANS_PROB_XY = np.array([
    # ->CV    ->CA     ->CT_L   ->CT_R
    [0.94, 0.03, 0.015, 0.015],
    [0.10, 0.80, 0.05,  0.05 ],
    [0.05, 0.03, 0.90,  0.02 ],
    [0.05, 0.03, 0.02,  0.90 ],
])
INIT_MODE_PROB_XY = np.array([0.70, 0.10, 0.10, 0.10]) 

MODELS_Z = ["CV", "SINGER"]
TRANS_PROB_Z = np.array([
    # ->CV    ->SINGER
    [0.90,    0.10],   # CV
    [0.10,    0.90],   # SINGER
])
INIT_MODE_PROB_Z = np.array([0.90, 0.10])

assert np.allclose(TRANS_PROB_XY.sum(axis=1), 1.0)
assert np.allclose(TRANS_PROB_Z.sum(axis=1), 1.0)

SENSOR_CSV = "radar_sensor_tracks.csv"
OUTPUT_FUSED_CSV = "fused_tracks_imm3.csv"
OUTPUT_COLUMNS = [
    "time", "global_track_id", "x", "y", "z", "vx", "vy", "vz",
    "ax", "ay", "az", "pos_sigma_m", "vel_sigma_mps", "sigma_x_m",
    "sigma_vx_mps", "sigma_y_m", "sigma_vy_mps", "sigma_z_m",
    "sigma_vz_mps", "fused_tq", "prob", "mode_prob_xy_cv",
    "mode_prob_xy_ca", "mode_prob_xy_ct_left", "mode_prob_xy_ct_right",
    "mode_prob_z_cv", "mode_prob_z_singer", "dominant_model", "n_sources",
    "source_radars", "update_used", "is_prediction_only", "time_since_update_s",
]
DEBUG_OUTPUT_COLUMNS = [
    "min_cov_eig", "max_cov_eig", "cov_condition_number",
    "xy_mode_entropy", "z_mode_entropy",
]

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

def sigma_pos_to_tq(sigma_pos: float) -> int:
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))


def _normalize_probabilities(p, floor=1e-12):
    p = np.asarray(p, dtype=float).reshape(-1)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    p = np.maximum(p, floor)
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full_like(p, 1.0 / len(p))
    return p / total


def regularize_covariance(P, min_eig=COV_MIN_EIG, variance_floor=None):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Covariance kare bir matris olmalı")
    P = np.nan_to_num(0.5 * (P + P.T), nan=0.0, posinf=1e12, neginf=-1e12)
    if variance_floor is not None:
        floor = np.asarray(variance_floor, dtype=float)
        if floor.shape != (P.shape[0],):
            raise ValueError("Variance floor covariance boyutuyla uyuşmuyor")
        diagonal = np.maximum(np.diag(P), floor)
        P[np.diag_indices_from(P)] = diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (P + P.T))
    eigenvalues = np.maximum(eigenvalues, min_eig)
    P = (eigenvectors * eigenvalues) @ eigenvectors.T
    if variance_floor is not None:
        P[np.diag_indices_from(P)] = np.maximum(np.diag(P), floor)
    return 0.5 * (P + P.T)


def _state_variance_floor(dimension):
    if dimension == 3:
        return Z_VARIANCE_FLOOR
    if dimension == 6:
        return XY_VARIANCE_FLOOR
    if dimension == 9:
        return STATE9_VARIANCE_FLOOR
    return None


def _solve_or_pinv(A, B):
    A = regularize_covariance(A)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


def _mahalanobis_squared(diff, covariance):
    solution = _solve_or_pinv(covariance, diff)
    value = float((diff.T @ solution).item())
    return max(value, 0.0) if math.isfinite(value) else math.inf


def _validate_measurement(state, covariance):
    state = np.asarray(state, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    return (
        np.isfinite(state).all()
        and np.isfinite(covariance).all()
        and covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
        and np.all(np.diag(covariance) > 0.0)
    )

def measurement_cov_from_row(row) -> np.ndarray:
    sp = float(row.get("sigma_pos_m", tq_to_sigma_pos(row.get("track_quality", TQ_MAX))))
    sv = float(row.get("sigma_vel_mps", tq_to_sigma_vel(row.get("track_quality", TQ_MAX))))
    if not math.isfinite(sp) or sp <= 0:
        sp = tq_to_sigma_pos(row.get("track_quality", TQ_MAX))
    if not math.isfinite(sv) or sv <= 0:
        sv = tq_to_sigma_vel(row.get("track_quality", TQ_MAX))
    
    if all(k in row for k in ["range", "azimuth", "elevation", "sigma_range", "sigma_az", "sigma_el"]):
        r, az, el = float(row["range"]), float(row["azimuth"]), float(row["elevation"])
        J = np.array([
            [math.cos(el)*math.cos(az), -r*math.cos(el)*math.sin(az), -r*math.sin(el)*math.cos(az)],
            [math.cos(el)*math.sin(az),  r*math.cos(el)*math.cos(az), -r*math.sin(el)*math.sin(az)],
            [math.sin(el),               0.0,                          r*math.cos(el)],
        ])
        R_polar = np.diag([float(row["sigma_range"])**2, float(row["sigma_az"])**2, float(row["sigma_el"])**2])
        R_cart = J @ R_polar @ J.T
        
        R = np.zeros((6, 6))
        position_indices = [0, 2, 4]
        R[np.ix_(position_indices, position_indices)] = R_cart
        R[1, 1] = R[3, 3] = R[5, 5] = sv**2
        return regularize_covariance(R)
        
    if pd.isna(row.get("z")):
        return regularize_covariance(np.diag([sp**2, sv**2, sp**2, sv**2, MISSING_Z_COVAR_PENALTY, MISSING_Z_COVAR_PENALTY]))

    x, y = float(row.get("x", 0.0)), float(row.get("y", 0.0))
    rng = math.sqrt(x**2 + y**2)
    sz = math.sqrt(sp**2 + (rng * ELEVATION_ERROR_RAD)**2)
    svz = sv * 1.5 
    
    return regularize_covariance(np.diag([sp**2, sv**2, sp**2, sv**2, sz**2, svz**2]))

def _pad_xy_meas(meas_4d, cov_4d):
    s = np.zeros((6, 1))
    s[0,0], s[1,0], s[3,0], s[4,0] = meas_4d[0,0], meas_4d[1,0], meas_4d[2,0], meas_4d[3,0]
    c = np.eye(6) * INITIAL_ACCEL_VARIANCE
    mapped = [0, 1, 3, 4]
    c[np.ix_(mapped, mapped)] = cov_4d
    return s, c

def _pad_z_meas(meas_2d, cov_2d):
    s = np.zeros((3, 1))
    s[0,0], s[1,0] = meas_2d[0,0], meas_2d[1,0]
    c = np.eye(3) * INITIAL_ACCEL_VARIANCE
    c[0:2, 0:2] = cov_2d[0:2, 0:2]
    return s, c

H_MEAS_XY = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
])

H_MEAS_Z = np.array([
    [1, 0, 0],
    [0, 1, 0],
])

H_MEAS_9D = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# ===========================================================================
# MODEL DINAMIKLERI: F ve Q URETICILERI
# ===========================================================================

# --- YATAY (XY) MODELLERI ---
def _F_Q_xy_ca(dt: float, q_intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, dt, 0.5 * dt**2, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5 * dt**2],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1],
    ])
    q = q_intensity ** 2
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    qb = q * np.array([
        [dt5 / 20, dt4 / 8, dt3 / 6],
        [dt4 / 8, dt3 / 3, dt2 / 2],
        [dt3 / 6, dt2 / 2, dt],
    ])
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = qb
    Q[3:6, 3:6] = qb
    return F, Q

def _F_Q_xy_cv(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, dt, 0, 0, 0, 0],
        [0, 1,  0, 0, 0, 0],
        [0, 0,  0, 0, 0, 0],
        [0, 0,  0, 1, dt, 0],
        [0, 0,  0, 0, 1,  0],
        [0, 0,  0, 0, 0,  0],
    ], dtype=float)
    q = CV_XY_PROCESS_NOISE_INTENSITY ** 2
    qb = q * np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ])
    Q = np.zeros((6, 6))
    Q[np.ix_([0, 1], [0, 1])] = qb
    Q[np.ix_([3, 4], [3, 4])] = qb
    Q[2, 2] = CV_ACCEL_LEAK_Q
    Q[5, 5] = CV_ACCEL_LEAK_Q
    return F, Q

def _F_Q_xy_ct(dt: float, omega: float, q_intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinated Turn (Koordineli Dönüş) Modeli. X ve Y hızlarını birbirine bağlar."""
    F = np.zeros((6, 6))
    if abs(omega) < 1e-6:
        sw, cw = dt, 0.0
    else:
        sw = math.sin(omega * dt) / omega
        cw = (1 - math.cos(omega * dt)) / omega
        
    # X ve Y eksenlerinin Coordinated Turn fiziksel baglantisi
    F[0, 0] = 1.0; F[0, 1] = sw;               F[0, 4] = -cw
    F[1, 1] = math.cos(omega * dt);            F[1, 4] = -math.sin(omega * dt)
    F[3, 3] = 1.0; F[3, 1] = cw;               F[3, 4] = sw
    F[4, 1] = math.sin(omega * dt);            F[4, 4] = math.cos(omega * dt)
    
    # İvmeler (index 2 ve 5) bu modelde kullanilmaz, sönümlenir
    F[2, 2] = 0.0
    F[5, 5] = 0.0
    
    # Süreç Gürültüsü (Dönüş sırasındaki rüzgar vb. sapmalar için)
    q = q_intensity ** 2
    dt2, dt3 = dt**2, dt**3
    qb_posvel = q * np.array([
        [dt3 / 3, dt2 / 2],
        [dt2 / 2, dt],
    ])
    Q = np.zeros((6, 6))
    Q[np.ix_([0, 1], [0, 1])] = qb_posvel
    Q[np.ix_([3, 4], [3, 4])] = qb_posvel
    
    # YENİ EKLENEN KISIM: Tekillik (Singular Matrix) hatasını önlemek için sızıntı
    Q[2, 2] = CV_ACCEL_LEAK_Q
    Q[5, 5] = CV_ACCEL_LEAK_Q
    
    return F, Q

# MODEL SÖZLÜĞÜNÜ GÜNCELLE
MODEL_FQ_XY = {
    "CV": _F_Q_xy_cv,
    "CA": lambda dt: _F_Q_xy_ca(dt, CA_XY_PROCESS_NOISE_INTENSITY),
    "CT_LEFT": lambda dt: _F_Q_xy_ct(dt, OMEGA_LEFT, CT_XY_PROCESS_NOISE_INTENSITY),
    "CT_RIGHT": lambda dt: _F_Q_xy_ct(dt, OMEGA_RIGHT, CT_XY_PROCESS_NOISE_INTENSITY)
}

# --- DIKEY (Z) MODELLERI ---
def _F_Q_z_cv(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, dt, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    q = CV_Z_PROCESS_NOISE_INTENSITY ** 2
    dt2, dt3 = dt**2, dt**3
    Q = np.zeros((3, 3))
    Q[0:2, 0:2] = q * np.array([[dt3/3, dt2/2], [dt2/2, dt]])
    Q[2, 2] = CV_ACCEL_LEAK_Q
    return F, Q

def _F_Q_z_singer(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 / SINGER_TAU_Z
    ad = alpha * dt
    emad = math.exp(-ad)
    em2ad = math.exp(-2*ad)
    
    F = np.array([
        [1, dt, (emad + ad - 1) / (alpha**2)],
        [0, 1, (1 - emad) / alpha],
        [0, 0, emad]
    ])
    
    q_var = 2 * alpha * (SINGER_SIGMA_Z**2)
    Q = np.zeros((3, 3))
    Q[0,0] = q_var * (1 - em2ad + 2*ad + 2*ad**3/3 - 2*ad**2 - 4*ad*emad) / (2 * alpha**5)
    Q[0,1] = q_var * (em2ad + 1 - 2*emad + 2*ad*emad - 2*ad + ad**2) / (2 * alpha**4)
    Q[1,0] = Q[0,1]
    Q[0,2] = q_var * (1 - em2ad - 2*ad*emad) / (2 * alpha**3)
    Q[2,0] = Q[0,2]
    Q[1,1] = q_var * (4*emad - 3 - em2ad + 2*ad) / (2 * alpha**3)
    Q[1,2] = q_var * (em2ad + 1 - 2*emad) / (2 * alpha**2)
    Q[2,1] = Q[1,2]
    Q[2,2] = q_var * (1 - em2ad) / (2 * alpha)
    
    return F, Q

MODEL_FQ_Z = {
    "CV": _F_Q_z_cv,
    "SINGER": _F_Q_z_singer
}

# ===========================================================================
# FUZYON (OLCUM BIRLESTIRME) FONKSIYONLARI
# ===========================================================================
def _ci_fuse(x1, P1, x2, P2):
    floor = _state_variance_floor(P1.shape[0])
    P1 = regularize_covariance(P1, variance_floor=floor)
    P2 = regularize_covariance(P2, variance_floor=floor)
    identity = np.eye(P1.shape[0])
    P1i = _solve_or_pinv(P1, identity)
    P2i = _solve_or_pinv(P2, identity)
    def trace_f(omega):
        information = omega * P1i + (1 - omega) * P2i
        return float(np.trace(_solve_or_pinv(information, identity)))

    res = minimize_scalar(trace_f, bounds=(1e-3, 1 - 1e-3), method="bounded")
    omega = res.x
    Pf = _solve_or_pinv(omega * P1i + (1 - omega) * P2i, identity)
    Pf = regularize_covariance(Pf, variance_floor=floor)
    xf = Pf @ (omega * P1i @ x1 + (1 - omega) * P2i @ x2)
    return xf, Pf

def _standard_fuse(x1, P1, x2, P2):
    identity = np.eye(P1.shape[0])
    return _kalman_measurement_update(x1, P1, x2, P2, identity)


def _kalman_measurement_update(x, P, z, R, H):
    """Numerically stable direct-measurement update using Joseph covariance form."""
    floor = _state_variance_floor(P.shape[0])
    P = regularize_covariance(P, variance_floor=floor)
    R = regularize_covariance(R)
    innovation = z - H @ x
    S = regularize_covariance(H @ P @ H.T + R)
    K = _solve_or_pinv(S, H @ P).T
    x_new = x + K @ innovation
    identity = np.eye(P.shape[0])
    residual_map = identity - K @ H
    P_new = residual_map @ P @ residual_map.T + K @ R @ K.T
    return x_new, regularize_covariance(P_new, variance_floor=floor)


def _gaussian_log_likelihood(diff: np.ndarray, S: np.ndarray) -> float:
    S = regularize_covariance(S)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not math.isfinite(logdet):
        return -math.inf
    d2 = _mahalanobis_squared(diff, S)
    if not math.isfinite(d2):
        return -math.inf
    k = diff.shape[0]
    return -0.5 * (d2 + logdet + k * math.log(2 * math.pi))

# ===========================================================================
# DEKUPLE DUAL-IMM GLOBAL TRACK 
# ===========================================================================
class GlobalTrackIMM3:
    _cnt = 0
    def __init__(self, t, state_6d, cov_6d, tq, src, use_ci=True, debug=False):
        if not _validate_measurement(state_6d, cov_6d):
            raise ValueError("Başlangıç state/covariance değerleri geçersiz")
        GlobalTrackIMM3._cnt += 1
        self.id = f"GT-{GlobalTrackIMM3._cnt:04d}"
        self.time = t
        self.use_ci = use_ci
        self.debug = debug
        self.mode_change_log = []
        
        # 6D ölçümü, XY (4D ölçüm -> 6D state) ve Z (2D ölçüm -> 3D state) olarak ayır
        xy_state = np.array([[state_6d[0,0]], [state_6d[1,0]], [0.0], [state_6d[2,0]], [state_6d[3,0]], [0.0]])
        z_state = np.array([[state_6d[4,0]], [state_6d[5,0]], [0.0]])
        
        _, xy_cov = _pad_xy_meas(
            state_6d[0:4], cov_6d[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
        )
        _, z_cov = _pad_z_meas(
            state_6d[4:6], cov_6d[np.ix_([4, 5], [4, 5])]
        )
        xy_cov = regularize_covariance(xy_cov, variance_floor=XY_VARIANCE_FLOOR)
        z_cov = regularize_covariance(z_cov, variance_floor=Z_VARIANCE_FLOOR)

        # --- YATAY IMM ---
        self.xy_models = MODELS_XY
        self.xy_model_state = {m: xy_state.copy() for m in self.xy_models}
        self.xy_model_cov = {m: xy_cov.copy() for m in self.xy_models}
        self.xy_mode_prob = dict(zip(self.xy_models, INIT_MODE_PROB_XY))

        # --- DIKEY IMM ---
        self.z_models = MODELS_Z
        self.z_model_state = {m: z_state.copy() for m in self.z_models}
        self.z_model_cov = {m: z_cov.copy() for m in self.z_models}
        self.z_mode_prob = dict(zip(self.z_models, INIT_MODE_PROB_Z))

        self.state, self.cov = self._combine_to_9d()
        
        self.last_update = t
        self.existence_prob = 0.1 + 0.7 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.status = "TENTATIVE"
        self.sources: set = {src}
        self.source_radar_names: set = {src[0] if isinstance(src, tuple) else src}
        self.source_measurement_details: set = {f"{src[0] if isinstance(src, tuple) else src}@{t:.2f}"}
        self.hits_count = 1
        self.creation_time = t
        self.position_history = [(float(self.state[0, 0]), float(self.state[3, 0]), float(self.state[6, 0]), float(t))]
        self._last_dominant_model = self.dominant_model()

    def _combine_to_9d(self):
        # XY Birlestirme
        x_xy = sum(self.xy_mode_prob[m] * self.xy_model_state[m] for m in self.xy_models)
        P_xy = sum(self.xy_mode_prob[m] * (self.xy_model_cov[m] + (self.xy_model_state[m] - x_xy) @ (self.xy_model_state[m] - x_xy).T) for m in self.xy_models)
        
        # Z Birlestirme
        x_z = sum(self.z_mode_prob[m] * self.z_model_state[m] for m in self.z_models)
        P_z = sum(self.z_mode_prob[m] * (self.z_model_cov[m] + (self.z_model_state[m] - x_z) @ (self.z_model_state[m] - x_z).T) for m in self.z_models)
        
        # 9D Tekil Forma Dönüştürme: [x, vx, ax, y, vy, ay, z, vz, az]^T
        state_9d = np.zeros((9, 1))
        state_9d[0:6, 0] = x_xy[:, 0]
        state_9d[6:9, 0] = x_z[:, 0]
        
        cov_9d = np.zeros((9, 9))
        cov_9d[0:6, 0:6] = P_xy
        cov_9d[6:9, 6:9] = P_z
        if not np.isfinite(state_9d).all():
            raise FloatingPointError("IMM birleşik state içinde NaN/inf oluştu")
        return state_9d, regularize_covariance(
            cov_9d, variance_floor=STATE9_VARIANCE_FLOOR
        )

    def _validate_mode_probabilities(self, fallback_xy=None, fallback_z=None):
        xy = np.array([self.xy_mode_prob[m] for m in self.xy_models], dtype=float)
        z = np.array([self.z_mode_prob[m] for m in self.z_models], dtype=float)
        if not np.isfinite(xy).all():
            warnings.warn(f"{self.id}: geçersiz XY mode probability; önceki değer korundu")
            xy = INIT_MODE_PROB_XY if fallback_xy is None else fallback_xy
        if not np.isfinite(z).all():
            warnings.warn(f"{self.id}: geçersiz Z mode probability; önceki değer korundu")
            z = INIT_MODE_PROB_Z if fallback_z is None else fallback_z
        xy = _normalize_probabilities(xy)
        z = _normalize_probabilities(z)
        self.xy_mode_prob = dict(zip(self.xy_models, xy))
        self.z_mode_prob = dict(zip(self.z_models, z))
        current = self.dominant_model()
        if self.debug and current != self._last_dominant_model:
            self.mode_change_log.append((float(self.time), self._last_dominant_model, current))
        self._last_dominant_model = current

    def covariance_debug_fields(self):
        eigenvalues = np.linalg.eigvalsh(self.cov)
        positive = np.maximum(eigenvalues, COV_MIN_EIG)
        xy = _normalize_probabilities(list(self.xy_mode_prob.values()))
        z = _normalize_probabilities(list(self.z_mode_prob.values()))
        return {
            "min_cov_eig": float(eigenvalues.min()),
            "max_cov_eig": float(eigenvalues.max()),
            "cov_condition_number": float(positive.max() / positive.min()),
            "xy_mode_entropy": float(-np.sum(xy * np.log(xy))),
            "z_mode_entropy": float(-np.sum(z * np.log(z))),
        }

    def propagate(self, t):
        dt = t - self.time
        if dt < 0:
            raise ValueError("Track zamanı geriye doğru ilerleyemez")
        if dt == 0:
            return

        # --- YATAY IMM TAHMINI ---
        mu_xy = _normalize_probabilities([self.xy_mode_prob[m] for m in self.xy_models])
        c_bar_xy = _normalize_probabilities(TRANS_PROB_XY.T @ mu_xy)
        mix_w_xy = (TRANS_PROB_XY * mu_xy[:, None]) / c_bar_xy[None, :]
        mix_w_xy = np.column_stack([
            _normalize_probabilities(mix_w_xy[:, j])
            for j in range(mix_w_xy.shape[1])
        ])
        
        for j, mj in enumerate(self.xy_models):
            x0j = sum(mix_w_xy[i, j] * self.xy_model_state[mi] for i, mi in enumerate(self.xy_models))
            P0j = sum(mix_w_xy[i, j] * (self.xy_model_cov[mi] + (self.xy_model_state[mi] - x0j) @ (self.xy_model_state[mi] - x0j).T) for i, mi in enumerate(self.xy_models))
            P0j = regularize_covariance(P0j, variance_floor=XY_VARIANCE_FLOOR)
            
            F, Q = MODEL_FQ_XY[mj](dt)
            self.xy_model_state[mj] = F @ x0j
            self.xy_model_cov[mj] = regularize_covariance(
                F @ P0j @ F.T + Q, variance_floor=XY_VARIANCE_FLOOR
            )
            
        self.xy_mode_prob = dict(zip(self.xy_models, c_bar_xy))

        # --- DIKEY IMM TAHMINI ---
        mu_z = _normalize_probabilities([self.z_mode_prob[m] for m in self.z_models])
        c_bar_z = _normalize_probabilities(TRANS_PROB_Z.T @ mu_z)
        mix_w_z = (TRANS_PROB_Z * mu_z[:, None]) / c_bar_z[None, :]
        mix_w_z = np.column_stack([
            _normalize_probabilities(mix_w_z[:, j])
            for j in range(mix_w_z.shape[1])
        ])
        
        for j, mj in enumerate(self.z_models):
            x0j = sum(mix_w_z[i, j] * self.z_model_state[mi] for i, mi in enumerate(self.z_models))
            P0j = sum(mix_w_z[i, j] * (self.z_model_cov[mi] + (self.z_model_state[mi] - x0j) @ (self.z_model_state[mi] - x0j).T) for i, mi in enumerate(self.z_models))
            P0j = regularize_covariance(P0j, variance_floor=Z_VARIANCE_FLOOR)
            
            F, Q = MODEL_FQ_Z[mj](dt)
            self.z_model_state[mj] = F @ x0j
            self.z_model_cov[mj] = regularize_covariance(
                F @ P0j @ F.T + Q, variance_floor=Z_VARIANCE_FLOOR
            )
            
        self.z_mode_prob = dict(zip(self.z_models, c_bar_z))

        self.time = t
        self._validate_mode_probabilities(mu_xy, mu_z)
        self.state, self.cov = self._combine_to_9d()
        self.existence_prob *= math.exp(-dt / EXISTENCE_DECAY_TAU_S)
        time_since_update = self.time - self.last_update
        if time_since_update > MISS_DECAY_START_S:
            extra_dt = min(dt, time_since_update - MISS_DECAY_START_S)
            self.existence_prob *= math.exp(-extra_dt / MISS_DECAY_TAU_S)
        self.existence_prob = float(np.clip(self.existence_prob, 0.0, 1.0))
        self._update_status()

    def update(self, meas_state_6d, meas_cov_6d, tq, src):
        if not _validate_measurement(meas_state_6d, meas_cov_6d):
            warnings.warn(f"{self.id}: NaN/inf veya geçersiz covariance içeren ölçüm atlandı")
            return False
        meas_xy = meas_state_6d[0:4] # x, vx, y, vy
        cov_xy = meas_cov_6d[np.ix_([0,1,2,3], [0,1,2,3])]
        m_s_xy, m_c_xy = _pad_xy_meas(meas_xy, cov_xy)

        meas_z = meas_state_6d[4:6] # z, vz
        cov_z = meas_cov_6d[np.ix_([4,5], [4,5])]
        m_s_z, m_c_z = _pad_z_meas(meas_z, cov_z)

        # --- YATAY IMM GUNCELLEMESI ---
        previous_mu_xy = np.array([self.xy_mode_prob[m] for m in self.xy_models])
        log_likelihoods_xy = []
        for m in self.xy_models:
            x_pred, P_pred = self.xy_model_state[m], self.xy_model_cov[m]
            S = H_MEAS_XY @ P_pred @ H_MEAS_XY.T + cov_xy
            diff = meas_xy - (H_MEAS_XY @ x_pred)
            log_likelihoods_xy.append(_gaussian_log_likelihood(diff, S))

            # Direct radar measurements use a Kalman update. ``use_ci`` is kept
            # for API compatibility and future correlated track-to-track input.
            xf, Pf = _kalman_measurement_update(
                x_pred, P_pred, meas_xy, cov_xy, H_MEAS_XY
            )
            self.xy_model_state[m] = xf
            self.xy_model_cov[m] = regularize_covariance(
                Pf, variance_floor=XY_VARIANCE_FLOOR
            )

        prior_xy = _normalize_probabilities(previous_mu_xy)
        log_raw_xy = np.log(prior_xy) + np.asarray(log_likelihoods_xy)
        if np.isfinite(log_raw_xy).any():
            new_mu_xy = np.exp(log_raw_xy - logsumexp(log_raw_xy))
        else:
            warnings.warn("XY IMM olasılık güncellemesi geçersiz; önceki dağılım korundu")
            new_mu_xy = prior_xy
        new_mu_xy = _normalize_probabilities(new_mu_xy)
        self.xy_mode_prob = dict(zip(self.xy_models, new_mu_xy))

        # --- DIKEY IMM GUNCELLEMESI ---
        previous_mu_z = np.array([self.z_mode_prob[m] for m in self.z_models])
        log_likelihoods_z = []
        for m in self.z_models:
            x_pred, P_pred = self.z_model_state[m], self.z_model_cov[m]
            S = H_MEAS_Z @ P_pred @ H_MEAS_Z.T + cov_z
            diff = meas_z - (H_MEAS_Z @ x_pred)
            log_likelihoods_z.append(_gaussian_log_likelihood(diff, S))

            xf, Pf = _kalman_measurement_update(
                x_pred, P_pred, meas_z, cov_z, H_MEAS_Z
            )
            self.z_model_state[m] = xf
            self.z_model_cov[m] = regularize_covariance(
                Pf, variance_floor=Z_VARIANCE_FLOOR
            )

        prior_z = _normalize_probabilities(previous_mu_z)
        log_raw_z = np.log(prior_z) + np.asarray(log_likelihoods_z)
        if np.isfinite(log_raw_z).any():
            new_mu_z = np.exp(log_raw_z - logsumexp(log_raw_z))
        else:
            warnings.warn("Z IMM olasılık güncellemesi geçersiz; önceki dağılım korundu")
            new_mu_z = prior_z
        new_mu_z = _normalize_probabilities(new_mu_z)
        self.z_mode_prob = dict(zip(self.z_models, new_mu_z))
        self._validate_mode_probabilities(previous_mu_xy, previous_mu_z)

        self.last_update = self.time
        mp = MEASUREMENT_EXISTENCE_GAIN_MIN + (
            MEASUREMENT_EXISTENCE_GAIN_MAX - MEASUREMENT_EXISTENCE_GAIN_MIN
        ) * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.existence_prob = self.existence_prob + (1 - self.existence_prob) * mp
        self.sources.add(src)
        self.source_radar_names.add(src[0] if isinstance(src, tuple) else src)
        self.source_measurement_details.add(f"{src[0] if isinstance(src, tuple) else src}@{self.time:.2f}")
        self.hits_count += 1
        
        self.state, self.cov = self._combine_to_9d()
        self.position_history.append((float(self.state[0, 0]), float(self.state[3, 0]), float(self.state[6, 0]), float(self.time)))
        self._update_status()
        return True

    def _update_status(self):
        if (self.time - self.last_update) > COAST_TIME_LIMIT or self.existence_prob < 0.2:
            self.status = "DELETED"
        elif (
            self.existence_prob > CONFIRM_EXISTENCE_THRESHOLD
            and self.hits_count >= CONFIRM_HITS
            and len(self.source_radar_names) >= MIN_CONFIRM_RADARS
        ) or (
            self.existence_prob > SINGLE_RADAR_CONFIRM_EXISTENCE
            and self.hits_count >= SINGLE_RADAR_CONFIRM_HITS
        ):
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"

    def dominant_model(self) -> str:
        # Hata ayıklama ve kayıt için her iki eksenin modunu tek bir string yapar
        xy_dom = max(self.xy_models, key=lambda m: self.xy_mode_prob[m])
        z_dom = max(self.z_models, key=lambda m: self.z_mode_prob[m])
        return f"XY:{xy_dom}|Z:{z_dom}"


# ===========================================================================
# FUZYON MERKEZI 
# ===========================================================================
class FusionCenterIMM3:
    def __init__(self, use_ci=True, verbose=True, debug=False, collect_diagnostics=False):
        self.tracks: List[GlobalTrackIMM3] = []
        self.src_map: Dict[Tuple, dict] = {}
        self.use_ci = use_ci
        self.verbose = verbose
        self.debug = debug
        self.collect_diagnostics = bool(collect_diagnostics)
        self.recently_deleted: List[dict] = [] 
        # Read-only association audit trail.  It does not participate in any
        # filtering or association decision.
        self.diagnostic_events: List[dict] = []

    def _source_track_id(self, src, current_time):
        entry = self.src_map.get(src)
        if entry is None:
            return None
        if current_time - entry["last_seen_time"] > SOURCE_MAP_TIMEOUT_S:
            return None
        return entry["global_track_id"]

    def _remember_source(self, src, track_id, current_time):
        self.src_map[src] = {
            "global_track_id": track_id,
            "last_seen_time": float(current_time),
        }

    def _replace_source_track(self, old_id, new_id):
        for entry in self.src_map.values():
            if entry["global_track_id"] == old_id:
                entry["global_track_id"] = new_id

    def _tracks_are_duplicate(self, t1, t2, chi2_thresh=36.0):
        # 1. AYRIK MESAFE KONTROLU (Yatay ve Dikey toleranslar farkli)
        dx = float(t1.state[0, 0]) - float(t2.state[0, 0])
        dy = float(t1.state[3, 0]) - float(t2.state[3, 0])
        dz = float(t1.state[6, 0]) - float(t2.state[6, 0])
        
        dist_xy = math.sqrt(dx * dx + dy * dy)
        dist_z = abs(dz)
        
        # 2. HIZ FARKI
        dvx = float(t1.state[1, 0]) - float(t2.state[1, 0])
        dvy = float(t1.state[4, 0]) - float(t2.state[4, 0])
        dvz = float(t1.state[7, 0]) - float(t2.state[7, 0])
        vel_diff = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

        diff = np.array([[dx], [dy], [dz]])
        P_sum = t1.cov[np.ix_([0, 3, 6], [0, 3, 6])] + t2.cov[np.ix_([0, 3, 6], [0, 3, 6])]
        d2 = _mahalanobis_squared(diff, P_sum)
        position_close = dist_xy < DUPLICATE_DIST_XY_M and dist_z < DUPLICATE_DIST_Z_M
        velocity_close = vel_diff <= DUPLICATE_VEL_MPS
        shares_radar = bool(t1.source_radar_names & t2.source_radar_names)
        statistically_very_close = d2 < DUPLICATE_STRONG_MAHALANOBIS
        return position_close and velocity_close and (shares_radar or statistically_very_close)

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
                    self._replace_source_track(weaker.id, keeper.id)
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
                    self._replace_source_track(t.id, c.id)
                    to_delete.add(t.id)
        self.tracks = [gt for gt in self.tracks if gt.id not in to_delete]

    def process_batch(self, t, measurements):
        for gt in self.tracks:
            gt.propagate(t)

        batch_diag = [{
            "event_type": "measurement",
            "time": float(t),
            "label": str(m.get("diagnostic_label", "")),
            "sensor": str(m["src"][0]),
            "local_track_id": str(m["src"][1]),
            "measurement_id": m.get("measurement_id"),
            "sequence_number": m.get("sequence_number"),
            "measurement_timestamp": m.get("measurement_timestamp", float(t)),
            "source_map_track_id": None,
            "source_d2_xy": np.nan,
            "source_gate_xy": np.nan,
            "source_d2_z": np.nan,
            "source_gate_z": np.nan,
            "source_xy_rejected": False,
            "source_z_rejected": False,
            "accepted_stage": "",
            "assigned_global_track_id": None,
            "created": False,
            "revived": False,
        } for m in measurements]
            
        deleted_tracks = [gt for gt in self.tracks if gt.status == "DELETED"]
        deleted_ids = {gt.id for gt in deleted_tracks}
        for gt in deleted_tracks:
            if self.collect_diagnostics:
                for label in sorted(getattr(gt, "_diagnostic_labels", set())):
                    self.diagnostic_events.append({
                        "event_type": "track_deleted",
                        "time": float(t),
                        "label": str(label),
                        "assigned_global_track_id": str(gt.id),
                    })
            self.recently_deleted.append({
                'delete_time': t,
                'track': gt
            })
            
        self.recently_deleted = [
            rd for rd in self.recently_deleted
            if (t - rd['delete_time']) <= REVIVE_WINDOW_S
        ]

        self.tracks = [gt for gt in self.tracks if gt.id not in deleted_ids]
        self.src_map = {
            k: v for k, v in self.src_map.items()
            if v["global_track_id"] not in deleted_ids
            and (t - v["last_seen_time"]) <= SOURCE_MAP_TIMEOUT_S
        }

        gt_by_id = {gt.id: i for i, gt in enumerate(self.tracks)}
        matched_gt, matched_m = set(), set()

        idx_xy = [0, 1, 2, 3] # x, vx, y, vy
        idx_z = [4, 5]        # z, vz

        # Adım 1: Mevcut aktif trackler ile yeni ölçümleri eşleştir (Ayrık Mahalanobis)
        for mi, m in enumerate(measurements):
            gid = self._source_track_id(m["src"], t)
            if gid is None or gid not in gt_by_id:
                continue
            gi = gt_by_id[gid]
            if gi in matched_gt:
                continue
            gt = self.tracks[gi]
            batch_diag[mi]["source_map_track_id"] = str(gt.id)
            S = H_MEAS_9D @ gt.cov @ H_MEAS_9D.T + m["cov"]
            diff = H_MEAS_9D @ gt.state - m["state"]
            try:
                dom = gt.dominant_model()
                mult = MANEUVER_GATE_MULTIPLIER if any(k in dom for k in ["CA", "CT_LEFT", "CT_RIGHT"]) else 1.0
                gate_xy = GATE_CHI2_4DOF * mult
                gate_z = GATE_CHI2_2DOF * mult

                S_xy = S[np.ix_(idx_xy, idx_xy)]
                diff_xy = diff[idx_xy]
                d2_xy = _mahalanobis_squared(diff_xy, S_xy)

                S_z = S[np.ix_(idx_z, idx_z)]
                diff_z = diff[idx_z]
                d2_z = _mahalanobis_squared(diff_z, S_z)

                batch_diag[mi].update({
                    "source_d2_xy": float(d2_xy),
                    "source_gate_xy": float(gate_xy),
                    "source_d2_z": float(d2_z),
                    "source_gate_z": float(gate_z),
                    "source_xy_rejected": bool(d2_xy >= gate_xy),
                    "source_z_rejected": bool(d2_z >= gate_z),
                })

                if d2_xy < gate_xy and d2_z < gate_z:
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    gt._diagnostic_labels = getattr(gt, "_diagnostic_labels", set())
                    gt._diagnostic_labels.add(str(m.get("diagnostic_label", "")))
                    self._remember_source(m["src"], gt.id, t)
                    matched_gt.add(gi)
                    matched_m.add(mi)
                    batch_diag[mi]["accepted_stage"] = "source_map"
                    batch_diag[mi]["assigned_global_track_id"] = str(gt.id)
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
                    S = H_MEAS_9D @ gt.cov @ H_MEAS_9D.T + m["cov"]
                    diff = H_MEAS_9D @ gt.state - m["state"]
                    try:
                        S_xy = S[np.ix_(idx_xy, idx_xy)]
                        diff_xy = diff[idx_xy]
                        d2_xy = _mahalanobis_squared(diff_xy, S_xy)

                        S_z = S[np.ix_(idx_z, idx_z)]
                        diff_z = diff[idx_z]
                        d2_z = _mahalanobis_squared(diff_z, S_z)

                        dom = gt.dominant_model()
                        mult = MANEUVER_GATE_MULTIPLIER if any(k in dom for k in ["CA", "CT_LEFT", "CT_RIGHT"]) else 1.0
                        gate_xy = GATE_CHI2_4DOF * mult
                        gate_z = GATE_CHI2_2DOF * mult

                        if d2_xy < gate_xy and d2_z < gate_z:
                            cost[ri, ci] = d2_xy + d2_z
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
                    gt._diagnostic_labels = getattr(gt, "_diagnostic_labels", set())
                    gt._diagnostic_labels.add(str(m.get("diagnostic_label", "")))
                    self._remember_source(m["src"], gt.id, t)
                    matched_gt.add(gi)
                    matched_m.add(mi)
                    batch_diag[mi]["accepted_stage"] = "hungarian"
                    batch_diag[mi]["assigned_global_track_id"] = str(gt.id)

        # Adım 3: Hiçbir aktif track'e atanamayan ölçümler için (Revive veya Create)
        for mi, m in enumerate(measurements):
            if mi in matched_m:
                continue
                
            best_rd_idx = -1
            best_dist_xy = 1e9
            
            for idx, rd in enumerate(self.recently_deleted):
                old_gt = rd['track']
                dt_gap = t - rd['delete_time']
                
                pred_x = float(old_gt.state[0, 0]) + float(old_gt.state[1, 0]) * dt_gap
                pred_y = float(old_gt.state[3, 0]) + float(old_gt.state[4, 0]) * dt_gap
                pred_z = float(old_gt.state[6, 0]) + float(old_gt.state[7, 0]) * dt_gap
                
                meas_x = float(m["state"][0, 0])
                meas_y = float(m["state"][2, 0])
                meas_z = float(m["state"][4, 0])
                
                ddx = pred_x - meas_x
                ddy = pred_y - meas_y
                ddz = pred_z - meas_z
                
                dist_xy = math.sqrt(ddx * ddx + ddy * ddy)
                dist_z = abs(ddz)
                
                dvx = float(old_gt.state[1, 0]) - float(m["state"][1, 0])
                dvy = float(old_gt.state[4, 0]) - float(m["state"][3, 0])
                dvz = float(old_gt.state[7, 0]) - float(m["state"][5, 0])
                vel_diff = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

                position_diff = np.array([[ddx], [ddy], [ddz]])
                position_cov = (
                    old_gt.cov[np.ix_([0, 3, 6], [0, 3, 6])]
                    + m["cov"][np.ix_([0, 2, 4], [0, 2, 4])]
                )
                revive_d2 = _mahalanobis_squared(position_diff, position_cov)

                if (
                    dt_gap <= REVIVE_WINDOW_S
                    and dist_xy < REVIVE_DIST_XY_M
                    and dist_z < REVIVE_DIST_Z_M
                    and vel_diff <= REVIVE_VEL_MPS
                    and revive_d2 < float(chi2.ppf(GATE_CONFIDENCE, df=3))
                ):
                    if dist_xy < best_dist_xy:
                        best_dist_xy = dist_xy
                        best_rd_idx = idx

            if best_rd_idx != -1:
                rd_entry = self.recently_deleted.pop(best_rd_idx)
                revived_gt = rd_entry['track']
                revived_gt.propagate(t)
                
                revived_gt.status = "TENTATIVE" 
                revived_gt.existence_prob = max(0.4, revived_gt.existence_prob * 0.8)
                revived_gt.update(m["state"], m["cov"], m["tq"], m["src"])
                revived_gt._diagnostic_labels = getattr(revived_gt, "_diagnostic_labels", set())
                revived_gt._diagnostic_labels.add(str(m.get("diagnostic_label", "")))
                
                self.tracks.append(revived_gt)
                self._remember_source(m["src"], revived_gt.id, t)
                batch_diag[mi]["accepted_stage"] = "revive"
                batch_diag[mi]["assigned_global_track_id"] = str(revived_gt.id)
                batch_diag[mi]["revived"] = True
            else:
                ng = GlobalTrackIMM3(
                    t, m["state"], m["cov"], m["tq"], m["src"],
                    use_ci=self.use_ci, debug=self.debug,
                )
                self.tracks.append(ng)
                ng._diagnostic_labels = {str(m.get("diagnostic_label", ""))}
                self._remember_source(m["src"], ng.id, t)
                batch_diag[mi]["accepted_stage"] = "create"
                batch_diag[mi]["assigned_global_track_id"] = str(ng.id)
                batch_diag[mi]["created"] = True

        self._merge_duplicates(t)
        if self.collect_diagnostics:
            self.diagnostic_events.extend(batch_diag)

# ===========================================================================
# ANA YURUTME
# ===========================================================================
def run_imm3_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
    include_source_measurement_details=False,
    *,
    include_debug_fields=False,
    diagnostics_csv=None,
) -> pd.DataFrame:
    if verbose:
        mode_str = (
            "DECOUPLED DUAL-IMM + Kalman measurement update "
            "(CI correlated track fusion için hazır)"
        )
        print(f"{mode_str} fuzyon calistiriliyor: {sensor_csv}")

    sensor_df = pd.read_csv(sensor_csv)
    # ``is_clutter`` is simulation ground-truth metadata, not an observable
    # radar feature.  Using it here would give IMM an unfair oracle advantage.

    GlobalTrackIMM3._cnt = 0
    fc = FusionCenterIMM3(
        use_ci=use_ci,
        verbose=verbose,
        debug=include_debug_fields,
        collect_diagnostics=diagnostics_csv is not None,
    )
    output_records = []

    for t_val, group in sensor_df.groupby("time", sort=True):
        if not math.isfinite(float(t_val)):
            warnings.warn("NaN/inf zaman damgalı ölçüm grubu atlandı")
            continue
        measurements = []
        for _, row in group.iterrows():
            z_val = row.get("z", np.nan)
            vz_val = row.get("vz", np.nan)
            
            state = np.array([
                row["x"],
                row["vx"],
                row["y"],
                row["vy"],
                0.0 if pd.isna(z_val) else float(z_val),
                0.0 if pd.isna(vz_val) else float(vz_val),
            ]).reshape(6, 1)
            
            cov = measurement_cov_from_row(row)
            tq = float(row.get("track_quality", TQ_MAX))
            if not math.isfinite(tq):
                tq = TQ_MAX
            tq = float(np.clip(tq, TQ_MIN, TQ_MAX))
            if not _validate_measurement(state, cov):
                warnings.warn("NaN/inf state veya geçersiz covariance içeren ölçüm atlandı")
                continue
            
            measurements.append({
                "state": state,
                "cov": cov,
                "tq": tq,
                "src": (row["sensor"], row["local_track_id"]),
                "diagnostic_label": row.get("callsign_true", ""),
            })
        fc.process_batch(t_val, measurements)

        for gt in fc.tracks:
            if gt.status != "CONFIRMED":
                continue

            sigma_x = math.sqrt(max(float(gt.cov[0, 0]), 1e-6))
            sigma_vx = math.sqrt(max(float(gt.cov[1, 1]), 1e-6))
            sigma_y = math.sqrt(max(float(gt.cov[3, 3]), 1e-6))
            sigma_vy = math.sqrt(max(float(gt.cov[4, 4]), 1e-6))
            sigma_z = math.sqrt(max(float(gt.cov[6, 6]), 1e-6))
            sigma_vz = math.sqrt(max(float(gt.cov[7, 7]), 1e-6))
            source_names = ", ".join(sorted(gt.source_radar_names))

            rec = {
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
                "mode_prob_xy_cv": round(gt.xy_mode_prob["CV"], 3),
                "mode_prob_xy_ca": round(gt.xy_mode_prob["CA"], 3),
                "mode_prob_xy_ct_left": round(gt.xy_mode_prob["CT_LEFT"], 3),
                "mode_prob_xy_ct_right": round(gt.xy_mode_prob["CT_RIGHT"], 3),
                "mode_prob_z_cv": round(gt.z_mode_prob["CV"], 3),
                "mode_prob_z_singer": round(gt.z_mode_prob["SINGER"], 3),
                "dominant_model": gt.dominant_model(),
                "n_sources": len(gt.source_radar_names),
                "source_radars": source_names,
                "update_used": bool(math.isclose(float(gt.last_update), float(t_val), abs_tol=1e-9)),
                "is_prediction_only": not bool(math.isclose(float(gt.last_update), float(t_val), abs_tol=1e-9)),
                "time_since_update_s": max(0.0, float(t_val) - float(gt.last_update)),
            }
            if include_source_measurement_details:
                rec["source_measurement_details"] = "; ".join(sorted(gt.source_measurement_details))
            if include_debug_fields:
                rec.update(gt.covariance_debug_fields())
            output_records.append(rec)

    output_columns = OUTPUT_COLUMNS.copy()
    if include_source_measurement_details:
        output_columns.append("source_measurement_details")
    if include_debug_fields:
        output_columns.extend(DEBUG_OUTPUT_COLUMNS)
    fused_df = pd.DataFrame(output_records, columns=output_columns)
    fused_df.to_csv(output_csv, index=False)

    if verbose:
        if not fused_df.empty:
            print(f"IMM3 fuzyonu tamamlandi. {len(fused_df)} CONFIRMED kayit bulundu.")
            cols = ["time", "global_track_id", "x", "y", "z", "mode_prob_xy_ca", "dominant_model"]
            print(fused_df[cols].tail(10).to_string(index=False))
        else:
            print(f"[!] Hic CONFIRMED track olusamadi. Bos CSV olusturuldu: {output_csv}")
        print(f"Cikti dosyasi: {output_csv}")

    if diagnostics_csv is not None:
        pd.DataFrame(fc.diagnostic_events).to_csv(diagnostics_csv, index=False)

    return fused_df

def run_imm_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
    include_source_measurement_details=False,
    *,
    include_debug_fields=False,
    diagnostics_csv=None,
) -> pd.DataFrame:
    return run_imm3_fusion(
        sensor_csv=sensor_csv,
        output_csv=output_csv,
        verbose=verbose,
        use_ci=use_ci,
        include_source_measurement_details=include_source_measurement_details,
        include_debug_fields=include_debug_fields,
        diagnostics_csv=diagnostics_csv,
    )

if __name__ == "__main__":
    run_imm3_fusion()
