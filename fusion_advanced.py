"""
füzyon2.py
==========
Gorevi: radar_sensor_tracks.csv uzerinden Covariance Intersection (CI)
fuzyon algoritmasini calistirip fused_tracks.csv cikisi olusturmaktir.
"""

import math
import os
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar

warnings.filterwarnings("ignore")

# ===========================================================================
# KONFIGURASYON & SABITLER
# ===========================================================================
PROCESS_NOISE_INTENSITY = 1.5
GATE_CHI2_4DOF = 18.47   
COAST_TIME_LIMIT = 30.0

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
    def __init__(self, t, state, cov, tq, src, use_ci=True):
        GlobalTrack._cnt += 1
        self.id = f"GT-{GlobalTrack._cnt:04d}"
        self.time = t
        self.state = state.reshape(4, 1)
        self.cov = cov
        self.last_update = t
        self.existence_prob = 0.1 + 0.7 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.status = "TENTATIVE"
        self.sources: set = {src}
        self.use_ci = use_ci # Parametreyi kaydet

    def propagate(self, t):
        dt = t - self.time
        if dt <= 0: return
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        q = PROCESS_NOISE_INTENSITY ** 2
        qb = q * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0,1],[0,1])] = qb
        Q[np.ix_([2,3],[2,3])] = qb
        self.state = F @ self.state
        self.cov = 0.5 * ((F @ self.cov @ F.T + Q) + (F @ self.cov @ F.T + Q).T)
        self.time = t
        self.existence_prob *= 0.97
        self._update_status()

    def update(self, meas_state, meas_cov, tq, src):
        if self.use_ci:
            xf, Pf = _ci_fuse(self.state, self.cov, meas_state, meas_cov)
        else:
            xf, Pf = _standard_fuse(self.state, self.cov, meas_state, meas_cov)
        self.state, self.cov = xf, Pf
        self.last_update = self.time
        mp = 0.5 + 0.45 * ((tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.existence_prob = self.existence_prob + (1 - self.existence_prob) * mp
        self.sources.add(src)
        self._update_status()

    def _update_status(self):
        if (self.time - self.last_update) > COAST_TIME_LIMIT or self.existence_prob < 0.2:
            self.status = "DELETED"
        elif self.existence_prob > 0.85:
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"

class FusionCenter:
    def __init__(self, use_ci=True):
        self.tracks: List[GlobalTrack] = []
        self.src_map: Dict[Tuple, str] = {}  
        self.use_ci = use_ci

    def process_batch(self, t, measurements):
        for gt in self.tracks: gt.propagate(t)
        deleted = {gt.id for gt in self.tracks if gt.status == "DELETED"}
        self.tracks = [gt for gt in self.tracks if gt.id not in deleted]
        self.src_map = {k: v for k, v in self.src_map.items() if v not in deleted}

        gt_by_id = {gt.id: i for i, gt in enumerate(self.tracks)}
        matched_gt, matched_m = set(), set()

        for mi, m in enumerate(measurements):
            gid = self.src_map.get(m["src"])
            if gid is None or gid not in gt_by_id: continue
            gi = gt_by_id[gid]
            if gi in matched_gt: continue
            gt = self.tracks[gi]
            S = gt.cov + m["cov"]
            diff = gt.state - m["state"]
            try:
                if float((diff.T @ np.linalg.inv(S) @ diff).item()) < GATE_CHI2_4DOF:
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi); matched_m.add(mi)
            except np.linalg.LinAlgError:
                continue

        r_gt = [i for i in range(len(self.tracks)) if i not in matched_gt]
        r_m  = [i for i in range(len(measurements)) if i not in matched_m]
        if r_gt and r_m:
            # DÜZELTME 1: np.inf yerine çok büyük bir rakam (1e9) veriyoruz
            cost = np.full((len(r_gt), len(r_m)), 1e9) 
            for ri, gi in enumerate(r_gt):
                gt = self.tracks[gi]
                for ci, mi in enumerate(r_m):
                    m = measurements[mi]
                    S = gt.cov + m["cov"]
                    diff = gt.state - m["state"]
                    try:
                        Si = np.linalg.inv(S)
                        _, ld = np.linalg.slogdet(S)
                        d2 = float((diff.T @ Si @ diff).item())
                        if d2 < GATE_CHI2_4DOF:
                            cost[ri, ci] = d2 + max(0, ld)
                    except np.linalg.LinAlgError:
                        pass

            # DÜZELTME 2: np.isinf yerine 1e9 kontrolü yapıyoruz
            if not np.all(cost == 1e9):
                for ri, ci in zip(*linear_sum_assignment(cost)):
                    # DÜZELTME 3: np.inf kontrolü yerine ceza limiti kontrolü yapıyoruz
                    if cost[ri, ci] >= 1e9: continue 
                    
                    gi, mi = r_gt[ri], r_m[ci]
                    gt = self.tracks[gi]
                    m = measurements[mi]
                    gt.update(m["state"], m["cov"], m["tq"], m["src"])
                    self.src_map[m["src"]] = gt.id
                    matched_gt.add(gi); matched_m.add(mi)
        for mi, m in enumerate(measurements):
            if mi in matched_m: continue
            ng = GlobalTrack(t, m["state"], m["cov"], m["tq"], m["src"], use_ci=self.use_ci)
            self.tracks.append(ng)
            self.src_map[m["src"]] = ng.id
# ===========================================================================
# ANA YURUTME
# ===========================================================================
def run_advanced_fusion(
    sensor_csv=SENSOR_CSV,
    output_csv=OUTPUT_FUSED_CSV,
    verbose=True,
    use_ci=True,
) -> pd.DataFrame:
    if verbose:
        mode_str = "Covariance Intersection (CI)" if use_ci else "Standard LMMSE (No-CI)"
        print(f"{mode_str} füzyon çalıştırılıyor: {sensor_csv}")

    sensor_df = pd.read_csv(sensor_csv)
    fusion_input = sensor_df[sensor_df["is_clutter"] != True].copy() if "is_clutter" in sensor_df.columns else sensor_df.copy()

    GlobalTrack._cnt = 0
    fc = FusionCenter(use_ci=use_ci)
    output_records = []

    for t_val, group in fusion_input.groupby("time", sort=True):
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
            if gt.status == "CONFIRMED":
                sigma_x = math.sqrt(max(float(gt.cov[0, 0]), 1e-6))
                sigma_vx = math.sqrt(max(float(gt.cov[1, 1]), 1e-6))
                sigma_y = math.sqrt(max(float(gt.cov[2, 2]), 1e-6))
                sigma_vy = math.sqrt(max(float(gt.cov[3, 3]), 1e-6))
                output_records.append({
                    "time": t_val,
                    "global_track_id": gt.id,
                    "x": float(gt.state[0, 0]),
                    "y": float(gt.state[2, 0]),
                    "vx": float(gt.state[1, 0]),
                    "vy": float(gt.state[3, 0]),
                    "pos_sigma_m": sigma_x,
                    "vel_sigma_mps": sigma_vx,
                    "sigma_x_m": sigma_x,
                    "sigma_vx_mps": sigma_vx,
                    "sigma_y_m": sigma_y,
                    "sigma_vy_mps": sigma_vy,
                    "fused_tq": sigma_pos_to_tq(sigma_x),
                    "prob": round(gt.existence_prob, 3),
                    "n_sources": len(gt.sources),
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
