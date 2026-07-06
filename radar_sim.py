"""
02_radar_simulatoru.py
======================
Gorevi: Hazir olan Ground Truth CSV dosyasini okumak ve 3 farkli
sanal radarin (A, B, C) olcum karakteristiklerini taklit ederek
asenkron, gurultulu ve clutter (sahte iz) iceren sensor verilerini uretmek.

Bu script iki farkli CSV dosyasi uretir:
  1) radar_sensor_tracks_idealize.csv  — orijinal, saf Gaussian gurultulu, basit clutter
  2) radar_sensor_tracks_gercekci.csv  — asagidaki iyilestirmeler eklenmis:
       [YEN1] Pd mesafeye bagli: uzak hedefte tespit olasiligi dusuyor
       [YEN2/5] Polar Gurultu (Fiziksel Model): Gercek radar polar (R, theta) olcum yapar.
                Gurultu polar koordinatlarda eklenir, kartezyene cevrildiginde hedefe
                yakin konumda dairesel, uzakta uzayan eliptik bir hata dagilimi olusur.
       [YEN3] Statik clutter: bazi sahte izler her taramada ayni yerde tekrar ediyor
       [YEN4] Bias drift: sistematik hata zamanla rastgele yuruyusuyle kayiyor
       [YEN6] Doppler Hiz Modeli: Hiz olcumu dogrudan kartezyen degil, radyal (Doppler)
                ve capraz-radyal (turetilen) olarak modellenmistir.

AKADEMIK NOTLAR:
* Clutter Modeli: Bu calismada "simplified uniform clutter model" (basitlestirilmis
  tekduze clutter modeli) kullanilmistir. Gercek radar sistemlerinde clutter cografi
  olarak kumeli olup cogunlukla Weibull dagilimi gosterir. Bu kargasiklik mevcut 
  calismanin kapsami disinda birakilmistir.
* Idealize modelde uygulanan "kartezyen yaklasim", dogrudan kartezyen koordinatlarda
  izotropik Gaussian gürültü varsaymaktadır ve gercek fizikten sapmalar icerir. 
  Fiziksel gerceklige yakinlik icin "gercekci" simulasyon ciktilari referans alinmalidir.
"""

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5
GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"
SENSOR_TRACKS_IDEALIZE_CSV = "radar_sensor_tracks_idealize.csv"
SENSOR_TRACKS_GERCEKCI_CSV = "radar_sensor_tracks_gercekci.csv"


def tq_to_sigma_pos(tq):
    tq = float(np.clip(tq, TQ_MIN, TQ_MAX))
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac


def tq_to_sigma_vel(tq):
    tq = float(np.clip(tq, TQ_MIN, TQ_MAX))
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac


# ===========================================================================
# RADAR TANIMLARI
# ===========================================================================

RADARS = [
    {
        "name":            "RADAR_A",
        "revisit_mean_s":  2.5,
        "revisit_jitter":  0.4,
        "delay_s":         0.3,
        "base_tq":         13,
        "tq_jitter":       1,
        "maneuver_tq_pen": 3,
        "pd":              0.95,
        "pd_ref_range_m":  150000.0,
        "pd_min":          0.50,
        "bias_x":          15.0,
        "bias_y":         -10.0,
        "bias_drift_std":  0.05,
        "clutter_rate":    0.0020,
        "n_static_clutter":2, 
        "static_visible_prob": 0.05,
        "pos_x":           0.0,
        "pos_y":           0.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,
        "fov_half_deg":    180.0,
        "clutter_box":    (-400000, 400000, -400000, 400000),
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
        "pd_ref_range_m":  120000.0,
        "pd_min":          0.40,
        "bias_x":         -25.0,
        "bias_y":          20.0,
        "bias_drift_std":  0.08,
        "clutter_rate":    0.0012,
        "n_static_clutter":3,
        "static_visible_prob": 0.10,
        "pos_x":           50000.0,
        "pos_y":          -5000.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,
        "fov_half_deg":    180.0,
        "clutter_box":    (-400000, 400000, -400000, 400000),
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
        "pd_ref_range_m":  80000.0,
        "pd_min":          0.30,
        "bias_x":          40.0,
        "bias_y":          35.0,
        "bias_drift_std":  0.12,
        "clutter_rate":    0.0008,
        "n_static_clutter":1,  
        "static_visible_prob": 0.15,
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

def _angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def in_fov(radar, x, y):
    dx, dy = x - radar["pos_x"], y - radar["pos_y"]
    rng = math.hypot(dx, dy)
    if radar["max_range_m"] is not None and rng > radar["max_range_m"]:
        return False
    if radar["fov_center_deg"] is not None:
        bearing = math.degrees(math.atan2(dy, dx))
        if abs(_angle_diff_deg(bearing, radar["fov_center_deg"])) > radar["fov_half_deg"]:
            return False
    return True


def is_maneuvering(callsign, t, gt_df, accel_threshold=5.0):
    sub = gt_df[gt_df["callsign"] == callsign].sort_values("time")
    if len(sub) < 3:
        return False
    dvx = np.interp(t, sub["time"], sub["vx"].diff().fillna(0))
    dvy = np.interp(t, sub["time"], sub["vy"].diff().fillna(0))
    return math.hypot(dvx, dvy) > accel_threshold


def range_dependent_pd(radar, x, y):
    dx = x - radar["pos_x"]
    dy = y - radar["pos_y"]
    rng = math.hypot(dx, dy)
    ref_r = radar["pd_ref_range_m"]
    max_r = radar["max_range_m"]
    if rng <= ref_r:
        return radar["pd"]
    if rng >= max_r:
        return radar["pd_min"]
    frac = (rng - ref_r) / (max_r - ref_r)
    return radar["pd"] * (1 - frac) + radar["pd_min"] * frac


SIGMA_AZ_RAD = math.radians(0.05)   # tipik azimut hatasi


# ===========================================================================
# IDEALIZE SIMULASYON (orijinal kod, degistirilmedi)
# ===========================================================================

def simulate_radar_idealize(radar, gt_df):
    """
    Orijinal simulasyon: Sabit Pd, kartezyen yaklasim ile izotropik Gaussian gurultu,
    rastgele clutter (simplified uniform clutter model), sabit bias.
    NOT: Bu metot sadece referans kiyaslamasi icindir, gercek radar fiziginden sapar.
    """
    records = []
    callsigns = gt_df["callsign"].unique()
    t_max = gt_df["time"].max()

    for cs in callsigns:
        sub = gt_df[gt_df["callsign"] == cs].sort_values("time")
        t_end = float(sub["time"].iloc[-1])

        local_id = f"{radar['name']}-T{np.random.randint(1000, 9999)}"
        t = float(sub["time"].iloc[0]) + np.random.uniform(0, radar["revisit_mean_s"])
        dropped = False

        while t <= t_end:
            t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
            if t > t_end:
                break

            x_true = float(np.interp(t, sub["time"], sub["x"]))
            y_true = float(np.interp(t, sub["time"], sub["y"]))
            vx_true = float(np.interp(t, sub["time"], sub["vx"]))
            vy_true = float(np.interp(t, sub["time"], sub["vy"]))

            if not in_fov(radar, x_true, y_true) or np.random.rand() > radar["pd"]:
                dropped = True
                continue

            if dropped:
                local_id = f"{radar['name']}-T{np.random.randint(1000, 9999)}"
                dropped = False

            tq = int(np.clip(
                radar["base_tq"]
                - (radar["maneuver_tq_pen"] if is_maneuvering(cs, t, gt_df) else 0)
                + np.random.randint(-radar["tq_jitter"], radar["tq_jitter"] + 1),
                TQ_MIN, TQ_MAX
            ))

            sp = tq_to_sigma_pos(tq)   
            sv = tq_to_sigma_vel(tq)

            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": local_id,
                "callsign_true":  cs,
                "x":  x_true + radar["bias_x"] + np.random.normal(0, sp),
                "y":  y_true + radar["bias_y"] + np.random.normal(0, sp),
                "vx": vx_true + np.random.normal(0, sv),
                "vy": vy_true + np.random.normal(0, sv),
                "track_quality": tq,
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    False,
            })

    xmin, xmax, ymin, ymax = radar["clutter_box"]
    t = np.random.uniform(0, radar["revisit_mean_s"])
    while t <= t_max:
        t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
        if t > t_max:
            break
        for _ in range(np.random.poisson(radar["clutter_rate"])):
            cx, cy = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
            if not in_fov(radar, cx, cy):
                continue
            tq = int(np.clip(np.random.randint(1, max(2, radar["base_tq"] - 3)), TQ_MIN, TQ_MAX))
            sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": f"{radar['name']}-CL{np.random.randint(10000, 99999)}",
                "callsign_true":  "CLUTTER",
                "x": cx, "y": cy,
                "vx": np.random.normal(0, sv),
                "vy": np.random.normal(0, sv),
                "track_quality": tq,
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    True,
            })

    return pd.DataFrame(records)


# ===========================================================================
# GERCEKCI SIMULASYON (Fiziksel iyilestirmeler eklendi)
# ===========================================================================

def simulate_radar_gercekci(radar, gt_df):
    records = []
    callsigns = gt_df["callsign"].unique()
    t_max = gt_df["time"].max()

    current_bias_x = float(radar["bias_x"])
    current_bias_y = float(radar["bias_y"])

    for cs in callsigns:
        sub = gt_df[gt_df["callsign"] == cs].sort_values("time")
        t_end = float(sub["time"].iloc[-1])

        local_id = f"{radar['name']}-T{np.random.randint(1000, 9999)}"
        t = float(sub["time"].iloc[0]) + np.random.uniform(0, radar["revisit_mean_s"])
        dropped = False

        while t <= t_end:
            t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
            if t > t_end:
                break

            x_true = float(np.interp(t, sub["time"], sub["x"]))
            y_true = float(np.interp(t, sub["time"], sub["y"]))
            vx_true = float(np.interp(t, sub["time"], sub["vx"]))
            vy_true = float(np.interp(t, sub["time"], sub["vy"]))

            current_bias_x += np.random.normal(0, radar["bias_drift_std"])
            current_bias_y += np.random.normal(0, radar["bias_drift_std"])

            effective_pd = range_dependent_pd(radar, x_true, y_true)
            if not in_fov(radar, x_true, y_true) or np.random.rand() > effective_pd:
                dropped = True
                continue

            if dropped:
                local_id = f"{radar['name']}-T{np.random.randint(1000, 9999)}"
                dropped = False

            tq = int(np.clip(
                radar["base_tq"]
                - (radar["maneuver_tq_pen"] if is_maneuvering(cs, t, gt_df) else 0)
                + np.random.randint(-radar["tq_jitter"], radar["tq_jitter"] + 1),
                TQ_MIN, TQ_MAX
            ))

            sigma_r = tq_to_sigma_pos(tq)
            sv = tq_to_sigma_vel(tq)

            # [YEN5] POLAR GURULTU MODELI (Gercek radar olcumu)
            dx_true = x_true - radar["pos_x"]
            dy_true = y_true - radar["pos_y"]
            r_true = math.hypot(dx_true, dy_true)
            theta_true = math.atan2(dy_true, dx_true)

            r_meas = r_true + np.random.normal(0, sigma_r)
            theta_meas = theta_true + np.random.normal(0, SIGMA_AZ_RAD)

            x_meas = radar["pos_x"] + current_bias_x + r_meas * math.cos(theta_meas)
            y_meas = radar["pos_y"] + current_bias_y + r_meas * math.sin(theta_meas)

            # [YEN6] DOPPLER HIZ MODELI
            # Radyal (Doppler) dogrudan olculur, capraz-radyal konumlardan turetildigi icin daha gurultuludur.
            v_rad_true = vx_true * math.cos(theta_true) + vy_true * math.sin(theta_true)
            v_cross_true = -vx_true * math.sin(theta_true) + vy_true * math.cos(theta_true)
            
            v_rad_meas = v_rad_true + np.random.normal(0, sv)
            v_cross_meas = v_cross_true + np.random.normal(0, sv * 4.0) # Turetilen hizda artirilmis hata

            vx_meas = v_rad_meas * math.cos(theta_meas) - v_cross_meas * math.sin(theta_meas)
            vy_meas = v_rad_meas * math.sin(theta_meas) + v_cross_meas * math.cos(theta_meas)

            # Cikti icin efektif kartezyen esdeger hata hesaplamasi
            effective_sigma_pos = math.sqrt(sigma_r**2 + (r_true * SIGMA_AZ_RAD)**2)

            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": local_id,
                "callsign_true":  cs,
                "x":  x_meas,
                "y":  y_meas,
                "vx": vx_meas,
                "vy": vy_meas,
                "track_quality": tq,
                "sigma_pos_m":   effective_sigma_pos,
                "sigma_vel_mps": sv,
                "is_clutter":    False,
            })

    # CLUTTER: Dinamik (rastgele) + Statik (sabit konumlu)
    xmin, xmax, ymin, ymax = radar["clutter_box"]
    n_static = radar.get("n_static_clutter", 0)
    static_positions = [
        (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        for _ in range(n_static)
    ]
    static_positions = [(cx, cy) for cx, cy in static_positions if in_fov(radar, cx, cy)]
    static_vis_prob = radar.get("static_visible_prob", 1.0)

    t = np.random.uniform(0, radar["revisit_mean_s"])
    while t <= t_max:
        t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
        if t > t_max:
            break

        for _ in range(np.random.poisson(radar["clutter_rate"])):
            cx, cy = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
            if not in_fov(radar, cx, cy):
                continue
            tq = int(np.clip(np.random.randint(1, max(2, radar["base_tq"] - 3)), TQ_MIN, TQ_MAX))
            sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": f"{radar['name']}-CL{np.random.randint(10000, 99999)}",
                "callsign_true":  "CLUTTER",
                "x": cx, "y": cy,
                "vx": np.random.normal(0, sv * 0.1),
                "vy": np.random.normal(0, sv * 0.1),
                "track_quality": tq,
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    True,
            })

        for cx, cy in static_positions:
            if np.random.rand() > static_vis_prob:
                continue
            tq = int(np.clip(np.random.randint(1, 5), TQ_MIN, TQ_MAX))
            sp, sv = tq_to_sigma_pos(tq), tq_to_sigma_vel(tq)
            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": f"{radar['name']}-SC{abs(hash((cx,cy))) % 90000:05d}",
                "callsign_true":  "STATIC_CLUTTER",
                "x": cx + np.random.normal(0, 20.0),
                "y": cy + np.random.normal(0, 20.0),
                "vx": np.random.normal(0, 0.5),
                "vy": np.random.normal(0, 0.5),
                "track_quality": tq,
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    True,
            })

    return pd.DataFrame(records)


# ===========================================================================
# ANA YURUTME
# ===========================================================================

def generate_sensor_csvs(
    gt_csv: str = GROUND_TRUTH_CSV,
    idealize_csv: str = SENSOR_TRACKS_IDEALIZE_CSV,
    gercekci_csv: str = SENSOR_TRACKS_GERCEKCI_CSV,
):
    print(f"Ground Truth dosyasi okunuyor: {gt_csv}")
    gt_df = pd.read_csv(gt_csv)
    print(f"   -> {len(gt_df)} satir yuklendi.\n")

    print("=" * 55)
    print("IDEALIZE SIMULASYON (orijinal, degistirilmemis mantik)")
    print("=" * 55)
    all_idealize = []
    for radar in RADARS:
        df_r = simulate_radar_idealize(radar, gt_df)
        all_idealize.append(df_r)
        real_c = (df_r["is_clutter"] == False).sum()
        clut_c = (df_r["is_clutter"] == True).sum()
        print(f"  {radar['name']}: {real_c} gercek olcum, {clut_c} clutter")

    sensor_idealize = (pd.concat(all_idealize, ignore_index=True)
                         .sort_values("time").reset_index(drop=True))
    sensor_idealize.to_csv(idealize_csv, index=False)
    print(f"\n  -> '{idealize_csv}' kaydedildi "
          f"({len(sensor_idealize)} toplam kayit)\n")

    print("=" * 55)
    print("GERCEKCI SIMULASYON (YEN1-YEN6 iyilestirmeleri)")
    print("=" * 55)
    all_gercekci = []
    for radar in RADARS:
        df_r = simulate_radar_gercekci(radar, gt_df)
        all_gercekci.append(df_r)
        real_c = (df_r["is_clutter"] == False).sum()
        clut_c = (df_r["is_clutter"] == True).sum()
        print(f"  {radar['name']}: {real_c} gercek olcum, {clut_c} clutter "
              f"(dinamik + statik)")

    sensor_gercekci = (pd.concat(all_gercekci, ignore_index=True)
                         .sort_values("time").reset_index(drop=True))
    sensor_gercekci.to_csv(gercekci_csv, index=False)
    print(f"\n  -> '{gercekci_csv}' kaydedildi "
          f"({len(sensor_gercekci)} toplam kayit)\n")

    return idealize_csv, gercekci_csv


if __name__ == "__main__":
    generate_sensor_csvs()