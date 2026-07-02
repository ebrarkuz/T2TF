"""
02_radar_simulatoru.py
======================
Gorevi: Hazir olan Ground Truth CSV dosyasini okumak ve 3 farkli 
sanal radarin (A, B, C) olcum karakteristiklerini taklit ederek 
asenkron, gurultulu ve clutter (sahte iz) iceren sensor verilerini uretmek.

CIKTI: radar_sensor_tracks.csv
"""

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===========================================================================
# KONFIGURASYON & PARAMETRELER
# ===========================================================================
INPUT_CSV = "ground_truth_adsb4.csv"
OUTPUT_CSV = "radar_sensor_tracks.csv"

# Track Quality (TQ) ve Standart Sapma (Kovaryans) Sinirlari
TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5

# Sanal Radar Karakteristikleri
RADARS = [
    {
        "name":            "RADAR_A",
        "revisit_mean_s":  2.5,        # 2.5 saniyede bir tarar
        "revisit_jitter":  0.4,
        "delay_s":         0.3,
        "base_tq":         13,
        "tq_jitter":       1,
        "maneuver_tq_pen": 3,
        "pd":              0.95,       # %95 tespit olasiligi
        "bias_x":          15.0,
        "bias_y":          -10.0,
        "clutter_rate":    1.5,
        "pos_x":           0.0,
        "pos_y":           0.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,       # 360 derece
        "fov_half_deg":    180.0,
        "clutter_box":     (-100000, 100000, -100000, 100000),
    },
    {
        "name":            "RADAR_B",
        "revisit_mean_s":  5.0,
        "revisit_jitter":  1.0,
        "delay_s":         0.8,
        "base_tq":         10,
        "tq_jitter":       2,
        "maneuver_tq_pen": 3,
        "pd":              0.88,
        "bias_x":          -25.0,
        "bias_y":          20.0,
        "clutter_rate":    2.5,
        "pos_x":           50000.0,
        "pos_y":           -5000.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,
        "fov_half_deg":    180.0,
        "clutter_box":     (-100000, 100000, -100000, 100000),
    },
    {
        "name":            "RADAR_C",
        "revisit_mean_s":  10.0,
        "revisit_jitter":  2.5,
        "delay_s":         1.5,
        "base_tq":         7,
        "tq_jitter":       2,
        "maneuver_tq_pen": 2,
        "pd":              0.80,
        "bias_x":          40.0,
        "bias_y":          35.0,
        "clutter_rate":    4.0,
        "pos_x":           15000.0,
        "pos_y":           60000.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,
        "fov_half_deg":    180.0,
        "clutter_box":     (-100000, 100000, -100000, 100000),
    },
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

def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0

def in_fov(radar, x: float, y: float) -> bool:
    dx, dy = x - radar["pos_x"], y - radar["pos_y"]
    rng = math.hypot(dx, dy)
    if radar["max_range_m"] is not None and rng > radar["max_range_m"]:
        return False
    if radar["fov_center_deg"] is not None:
        bearing = math.degrees(math.atan2(dy, dx))
        if abs(_angle_diff_deg(bearing, radar["fov_center_deg"])) > radar["fov_half_deg"]:
            return False
    return True

def is_maneuvering(callsign: str, t: float, gt_df: pd.DataFrame, accel_threshold: float = 5.0) -> bool:
    sub = gt_df[gt_df["callsign"] == callsign].sort_values("time")
    if len(sub) < 3: return False
    idx = np.searchsorted(sub["time"].values, t)
    if idx == 0 or idx >= len(sub): return False
    dvx = np.interp(t, sub["time"], sub["vx"].diff().fillna(0))
    dvy = np.interp(t, sub["time"], sub["vy"].diff().fillna(0))
    return math.hypot(dvx, dvy) > accel_threshold

# ===========================================================================
# RADAR SIMULASYONU ISLEYISI
# ===========================================================================
def simulate_radar(radar: dict, gt_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    callsigns = gt_df["callsign"].unique()
    t_max = gt_df["time"].max()

    # Gercek hedefler icin olcum uretimi
    for cs in callsigns:
        sub = gt_df[gt_df["callsign"] == cs].sort_values("time")
        t_start = float(sub["time"].iloc[0])
        t_end = float(sub["time"].iloc[-1])

        local_id = f"{radar['name']}-T{np.random.randint(1000,9999)}"
        t = t_start + np.random.uniform(0, radar["revisit_mean_s"])
        dropped = False

        while t <= t_end:
            t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
            if t > t_end: break

            # Zemin verisini interpole et (saniyedeki gercek konumu bul)
            x_true = float(np.interp(t, sub["time"], sub["x"]))
            y_true = float(np.interp(t, sub["time"], sub["y"]))
            vx_true = float(np.interp(t, sub["time"], sub["vx"]))
            vy_true = float(np.interp(t, sub["time"], sub["vy"]))

            # Gorus alani veya Tespit Olasiligi (Pd) disindaysa pas gec
            if not in_fov(radar, x_true, y_true) or np.random.rand() > radar["pd"]:
                dropped = True
                continue

            # Hedef tekrar bulunursa ID degisir (Gercek radar davranisi)
            if dropped:
                local_id = f"{radar['name']}-T{np.random.randint(1000,9999)}"
                dropped = False

            # TQ (Track Quality) Belirleme
            tq = radar["base_tq"]
            if is_maneuvering(cs, t, gt_df):
                tq -= radar["maneuver_tq_pen"]
            tq = int(np.clip(tq + np.random.randint(-radar["tq_jitter"], radar["tq_jitter"] + 1), TQ_MIN, TQ_MAX))

            # Hatayi hesapla ve gürültü ekle
            sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
            x_m = x_true + radar["bias_x"] + np.random.normal(0, sp)
            y_m = y_true + radar["bias_y"] + np.random.normal(0, sp)
            vx_m = vx_true + np.random.normal(0, sv)
            vy_m = vy_true + np.random.normal(0, sv)
            t_received = round(t + radar["delay_s"], 2)

            records.append({
                "time": t_received, "sensor": radar["name"], "local_track_id": local_id,
                "callsign_true": cs, "x": x_m, "y": y_m, "vx": vx_m, "vy": vy_m,
                "track_quality": tq, "sigma_pos_m": sp, "sigma_vel_mps": sv, "is_clutter": False,
            })

    # Ortama Clutter (Sahte İz) ekleme
    xmin, xmax, ymin, ymax = radar["clutter_box"]
    t = np.random.uniform(0, radar["revisit_mean_s"])
    while t <= t_max:
        t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
        if t > t_max: break
        
        for _ in range(np.random.poisson(radar["clutter_rate"])):
            cx, cy = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
            if not in_fov(radar, cx, cy): continue
            
            tq = int(np.clip(np.random.randint(1, max(2, radar["base_tq"] - 3)), TQ_MIN, TQ_MAX))
            sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
            records.append({
                "time": round(t + radar["delay_s"], 2), "sensor": radar["name"],
                "local_track_id": f"{radar['name']}-CL{np.random.randint(10000,99999)}",
                "callsign_true": "CLUTTER", "x": cx, "y": cy,
                "vx": np.random.normal(0, sv), "vy": np.random.normal(0, sv),
                "track_quality": tq, "sigma_pos_m": sp, "sigma_vel_mps": sv, "is_clutter": True,
            })
            
    return pd.DataFrame(records)

# ===========================================================================
# ANA YURUTME
# ===========================================================================
if __name__ == "__main__":
    print(f"1. Ground Truth dosyasi okunuyor: {INPUT_CSV}")
    try:
        gt_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(f"HATA: '{INPUT_CSV}' dosyasi bulunamadi. Ayni dizinde olduguna emin olun.")

    print(f"   -> Toplam {len(gt_df)} satir veri yuklendi.")
    print("\n2. Radar Simulasyonu Calistiriliyor...")
    
    all_tracks = []
    for radar in RADARS:
        df_r = simulate_radar(radar, gt_df)
        all_tracks.append(df_r)
        
        real_count = len(df_r[df_r["is_clutter"] == False])
        clut_count = len(df_r[df_r["is_clutter"] == True])
        print(f"   [OK] {radar['name']}: {real_count} gercek olcum, {clut_count} sahte iz (clutter) uretildi.")

    sensor_df = pd.concat(all_tracks, ignore_index=True).sort_values("time").reset_index(drop=True)
    sensor_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n3. Islem Tamamlandi! Toplam {len(sensor_df)} lokal track olusturuldu.")
    print(f"   Veriler '{OUTPUT_CSV}' dosyasina kaydedildi.")