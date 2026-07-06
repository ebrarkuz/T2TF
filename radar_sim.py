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
       [YEN2] Sigma mesafeye bagli: uzaklik arttikca metre hatasi buyuyor
       [YEN3] Statik clutter: bazi sahte izler her taramada ayni yerde tekrar ediyor
       [YEN4] Bias drift: sistematik hata zamanla rastgele yuruyusuyle kayiyor
"""

import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# config.py yerine direkt tanimlar (proje yapiniza gore degistirin)
TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5
GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"  # YENİ ÇOKLU HEDEF DOSYASI BURAYA
# Cikti dosyasi adlari (degistirildi — ikisi ayri ayri uretiliyor)
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
        "pd":              0.95,        # idealize: sabit Pd
        "pd_ref_range_m":  150000.0,    # [YEN1] bu mesafede pd gecerli; otesinde dusuyor
        "pd_min":          0.50,        # [YEN1] maksimum menzilde alt sinir Pd
        "bias_x":          15.0,
        "bias_y":         -10.0,
        "bias_drift_std":  0.05,        # [YEN4] her taramada bias'a eklenen std (m)
        "clutter_rate":    0.0020,
        "n_static_clutter":2, 
        "static_visible_prob": 0.05,          # [YEN3] sahnede kac adet statik clutter noktasi
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
        "pd_ref_range_m":  120000.0,    # [YEN1]
        "pd_min":          0.40,        # [YEN1]
        "bias_x":         -25.0,
        "bias_y":          20.0,
        "bias_drift_std":  0.08,        # [YEN4]
        "clutter_rate":    0.0012,
        "n_static_clutter":3,          # [YEN3]
        "static_visible_prob": 0.10,          # [YEN3]
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
        "pd_ref_range_m":  80000.0,     # [YEN1]
        "pd_min":          0.30,        # [YEN1]
        "bias_x":          40.0,
        "bias_y":          35.0,
        "bias_drift_std":  0.12,        # [YEN4]
        "clutter_rate":    0.0008,
        "n_static_clutter":1,  
        "static_visible_prob": 0.05,                  # [YEN3] azaltildi — çok fazla statik clutteri engellemek icin
        "pos_x":           15000.0,
        "pos_y":           60000.0,
        "max_range_m":     500000.0,
        "fov_center_deg":  None,
        "fov_half_deg":    180.0,
        "clutter_box":     (-100000, 100000, -100000, 100000),
        # Statik clutter her taramada mutlaka gorunmesin — olasilik ile belirsizlik ver
        "static_visible_prob": 0.15,
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


# [YEN1] Mesafeye bagli Pd hesaplama
# Orijinal kodda Pd sabit (pd=0.95 her zaman).
# Gercekte radar denklemi: SNR ~ 1/R^4, Pd ise SNR'ye bagli olarak dusuyor.
# Burada basit bir lineer interpolasyon kullaniyoruz:
#   - R <= pd_ref_range_m  : nominal Pd (tam verimli calisma)
#   - R = max_range_m      : pd_min (en dusuk Pd)
# Aradaki degerler lineer interpolasyonla bulunur.
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


# [YEN2] Mesafeye bagli sigma (pozisyon hatasi buyume)
# Orijinal kodda sigma yalnizca TQ'ya bagli, mesafeden bagimsiz.
# Gercekte azimut hatasi acisaldir (sigma_az sabit derece),
# bu da R arttikca metre cinsinden buyur: sigma_pos_extra ~ R * sigma_az_rad
# sigma_az_rad olarak 0.05 derece (tipik S-band radar) kullaniyoruz.
SIGMA_AZ_RAD = math.radians(0.05)   # tipik azimut hatasi

def range_dependent_sigma(radar, x, y, base_sigma_pos):
    dx = x - radar["pos_x"]
    dy = y - radar["pos_y"]
    rng = math.hypot(dx, dy)
    extra = rng * SIGMA_AZ_RAD      # uzakliktan gelen ek pozisyon hatasi (m)
    return math.sqrt(base_sigma_pos**2 + extra**2)


# ===========================================================================
# IDEALIZE SIMULASYON (orijinal kod, degistirilmedi)
# ===========================================================================

def simulate_radar_idealize(radar, gt_df):
    """
    Orijinal simulasyon:
    - Sabit Pd
    - TQ'ya bagli ama mesafeden bagimsiz sigma
    - Tamamen rastgele clutter (her taramada farkli konumlar)
    - Sabit bias (drift yok)
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

            sp = tq_to_sigma_pos(tq)   # mesafeden bagimsiz sigma (idealize)
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

    # Rastgele clutter (her taramada tamamen farkli konumlar — idealize)
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
# GERCEKCI SIMULASYON (iyilestirmeler eklendi)
# ===========================================================================

def simulate_radar_gercekci(radar, gt_df):
    """
    Gercekci simulasyon — 4 ek katman:

    [YEN1] Pd mesafeye bagli:
        Uzak hedefte SNR duser, dolayisiyla tespit olasiligi de dusuyor.
        pd_ref_range_m'de nominal Pd, max_range_m'de pd_min degeri kullanilir.

    [YEN2] Sigma mesafeye bagli:
        Azimut hatasi acisaldir; mesafe arttikca metre cinsinden pozisyon
        hatasi buyur: sigma_toplam = sqrt(sigma_TQ^2 + (R * sigma_az)^2).

    [YEN3] Statik clutter:
        Gercek clutter'in bir kismi zemin/bina gibi sabit nesnelerden gelir
        ve her taramada ayni koordinatlarda tekrar eder. Bu sahte izler
        'bellek' olusturarak association algoritmalarini daha gercekci
        sekilde zorlar (sadece dinamik/rastgele clutter bunu yapmaz).

    [YEN4] Bias drift:
        Gercek radar kalibrasyonu muksemmel degildir; sistematik konum
        hatasi (bias) sicaklik, mekanik titresim gibi etkenlerle zamanla
        yavasca kayar. Bunu random walk ile modelledik:
        bias(t+1) = bias(t) + N(0, bias_drift_std).
    """
    records = []
    callsigns = gt_df["callsign"].unique()
    t_max = gt_df["time"].max()

    # [YEN4] Baslangic bias degerleri (sabit bias'in uzerine drift eklenecek)
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

            # [YEN4] Her taramada bias yavasce kayiyor (random walk)
            current_bias_x += np.random.normal(0, radar["bias_drift_std"])
            current_bias_y += np.random.normal(0, radar["bias_drift_std"])

            # [YEN1] Mesafeye bagli Pd kullan (orijinalde: radar["pd"] sabit)
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

            # [YEN2] Mesafeye bagli sigma (orijinalde: yalnizca TQ'ya bagli)
            base_sp = tq_to_sigma_pos(tq)
            sp = range_dependent_sigma(radar, x_true, y_true, base_sp)
            sv = tq_to_sigma_vel(tq)

            records.append({
                "time":           round(t + radar["delay_s"], 2),
                "sensor":         radar["name"],
                "local_track_id": local_id,
                "callsign_true":  cs,
                # [YEN4] Sabit bias yerine zamanla kayan bias kullaniliyor
                "x":  x_true + current_bias_x + np.random.normal(0, sp),
                "y":  y_true + current_bias_y + np.random.normal(0, sp),
                "vx": vx_true + np.random.normal(0, sv),
                "vy": vy_true + np.random.normal(0, sv),
                "track_quality": tq,
                # sigma_pos_m artik mesafeyi de iceriyor (YEN2)
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    False,
            })

    # -----------------------------------------------------------------------
    # CLUTTER: Dinamik (rastgele) + [YEN3] Statik (sabit konumlu)
    # -----------------------------------------------------------------------
    xmin, xmax, ymin, ymax = radar["clutter_box"]

    # [YEN3] Statik clutter: her radar icin sahnede sabit konumlu noktalar belirlenir.
    # Bunlar her taramada AYNI koordinatlarda tekrar eder (zemin/bina yansimasi).
    n_static = radar.get("n_static_clutter", 0)
    static_positions = [
        (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        for _ in range(n_static)
    ]
    # FOV disindakileri ele
    static_positions = [(cx, cy) for cx, cy in static_positions
                        if in_fov(radar, cx, cy)]
    # Statik clutter her taramada mutlaka kaydedilirse sayi oldukca buyur.
    # Burada her statik noktanin gorunme olasiligini ekleyerek toplam yükü azaltıyoruz.
    static_vis_prob = radar.get("static_visible_prob", 1.0)

    t = np.random.uniform(0, radar["revisit_mean_s"])
    while t <= t_max:
        t += max(0.1, np.random.normal(radar["revisit_mean_s"], radar["revisit_jitter"]))
        if t > t_max:
            break

        # Dinamik clutter (orijinal ile ayni — tamamen rastgele)
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
                "vx": np.random.normal(0, sv * 0.1),  # clutter hizi sifira yakin
                "vy": np.random.normal(0, sv * 0.1),
                "track_quality": tq,
                "sigma_pos_m":   sp,
                "sigma_vel_mps": sv,
                "is_clutter":    True,
            })

        # [YEN3] Statik clutter: ayni konumlar her taramada kucuk titresimle tekrar
        # Ancak burada her pozisyonun gorunme olasiligini kontrol ederek toplam
        # sayiyi dusuruyoruz (ortalama olarak static_vis_prob ile carpiliyor).
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
                # Kucuk titresim: gercekte de statik clutter biraz oynar
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
    print("GERCEKCI SIMULASYON (YEN1-YEN4 iyilestirmeleri)")
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