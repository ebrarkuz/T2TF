"""
Link 16 Tarzı Track Quality (TQ) ile Asenkron Çoklu-Sensör Track Simülatörü
V2: ZORLU SENARYO (Clutter, Crossing Targets, Range Limits, Sensor Bias)
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. ZAMAN VE GROUND TRUTH (GERÇEK HEDEF HAREKETİ) ÜRETİMİ
# ---------------------------------------------------------------------------

DT = 1.0          # ground truth zaman adımı (saniye)
T_TOTAL = 300.0   # toplam simülasyon süresi (saniye)
timestamps = np.arange(0, T_TOTAL, DT)


def generate_linear_target(ts, x0=0.0, y0=10000.0, vx=100.0, vy=0.0, noise_std=0.4):
    """Doğrusal hedef. Manevra yapan hedef ile kesişmesi için ayarlandı."""
    n = len(ts)
    states = np.zeros((n, 4))  # x, vx, y, vy
    x, y, vxc, vyc = x0, y0, vx, vy
    for i in range(n):
        vxc += np.random.normal(0, noise_std)
        vyc += np.random.normal(0, noise_std)
        x += vxc * DT
        y += vyc * DT
        states[i] = [x, vxc, y, vyc]
    return states


def generate_maneuvering_target(ts, cx=10000.0, cy=10000.0, speed=100.0,
                                radius=5000.0, noise_std=1.5, turn_period=60.0):
    """Dairesel/manevralı hedef. Merkez ve yarıçap, doğrusal hedefle
    çarpışma/yakın geçiş (crossing) yaratacak şekilde ayarlandı."""
    n = len(ts)
    states = np.zeros((n, 4))
    omega = speed / radius
    theta = math.pi / 2  # Başlangıç açısı
    for i, t in enumerate(ts):
        if i > 0 and i % int(turn_period / DT) == 0:
            omega *= -1.0
        theta += omega * DT
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        vx = -radius * omega * math.sin(theta) + np.random.normal(0, noise_std)
        vy = radius * omega * math.cos(theta) + np.random.normal(0, noise_std)
        states[i] = [x, vx, y, vy]
    return states


gt_target1 = generate_linear_target(timestamps)
gt_target2 = generate_maneuvering_target(timestamps)

ground_truth = {
    "TARGET_1_LINEAR": pd.DataFrame(gt_target1, columns=["x", "vx", "y", "vy"]).assign(time=timestamps),
    "TARGET_2_MANEUVER": pd.DataFrame(gt_target2, columns=["x", "vx", "y", "vy"]).assign(time=timestamps),
}


def is_maneuvering(target_name: str, t: float, turn_period: float = 60.0) -> bool:
    if target_name != "TARGET_2_MANEUVER":
        return False
    cycle_pos = t % turn_period
    return cycle_pos < 6.0 

# ---------------------------------------------------------------------------
# 2. LINK 16 TRACK QUALITY (TQ) <-> KOVARYANS DÖNÜŞÜMÜ
# ---------------------------------------------------------------------------
TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX = 1500.0
SIGMA_POS_MIN = 30.0
SIGMA_VEL_MAX = 25.0
SIGMA_VEL_MIN = 0.5

def tq_to_sigma_pos(tq) -> float:
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac

def tq_to_sigma_vel(tq) -> float:
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac

def sigma_pos_to_tq(sigma_pos: float) -> int:
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))

def tq_to_R(tq) -> np.ndarray:
    sp = tq_to_sigma_pos(tq)
    sv = tq_to_sigma_vel(tq)
    return np.diag([sp ** 2, sv ** 2, sp ** 2, sv ** 2])

# ---------------------------------------------------------------------------
# 3. SENSOR TANIMLARI (FOV, BIAS VE CLUTTER EKLENDİ)
# ---------------------------------------------------------------------------

@dataclass
class SensorProfile:
    name: str
    x: float                       # Sensörün X konumu
    y: float                       # Sensörün Y konumu
    max_range: float               # Maksimum görüş menzili
    bias_x: float                  # Sistematik X kayması (m)
    bias_y: float                  # Sistematik Y kayması (m)
    clutter_rate: float            # Her taramada ortalama sahte alarm sayısı
    revisit_interval: float
    jitter: float
    base_tq: int
    tq_jitter: int
    pd: float
    maneuver_tq_penalty: int = 0


SENSORS: List[SensorProfile] = [
    SensorProfile(name="RADAR_A", x=5000.0, y=5000.0, max_range=15000.0, 
                  bias_x=150.0, bias_y=-80.0, clutter_rate=2.0,
                  revisit_interval=2.0, jitter=0.3, base_tq=13, tq_jitter=1, pd=0.95, maneuver_tq_penalty=3),
    
    SensorProfile(name="RADAR_B", x=20000.0, y=10000.0, max_range=18000.0, 
                  bias_x=-200.0, bias_y=250.0, clutter_rate=4.0,
                  revisit_interval=5.0, jitter=0.8, base_tq=10, tq_jitter=2, pd=0.90, maneuver_tq_penalty=3),
    
    SensorProfile(name="RADAR_C", x=0.0, y=0.0, max_range=40000.0, 
                  bias_x=400.0, bias_y=400.0, clutter_rate=8.0,
                  revisit_interval=10.0, jitter=2.0, base_tq=7, tq_jitter=2, pd=0.80, maneuver_tq_penalty=2),
]


def generate_sensor_tracks(sensor: SensorProfile, target_name: str, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Gerçek hedefler için asenkron ölçümler üretir. Menzil kısıtı ve Bias içerir."""
    records = []
    local_track_id = f"{sensor.name}-T{np.random.randint(1000, 9999)}"
    t = float(gt_df["time"].iloc[0]) + np.random.uniform(0, sensor.revisit_interval)
    t_max = float(gt_df["time"].iloc[-1])
    track_dropped = False

    while t <= t_max:
        t += max(0.1, np.random.normal(sensor.revisit_interval, sensor.jitter))
        if t > t_max:
            break

        x_true = np.interp(t, gt_df["time"], gt_df["x"])
        y_true = np.interp(t, gt_df["time"], gt_df["y"])
        
        # 1. MENZİL KONTROLÜ (FOV)
        distance_to_sensor = math.hypot(x_true - sensor.x, y_true - sensor.y)
        if distance_to_sensor > sensor.max_range:
            track_dropped = True
            continue # Hedef menzil dışında, ölçüm yok

        vx_true = np.interp(t, gt_df["time"], gt_df["vx"])
        vy_true = np.interp(t, gt_df["time"], gt_df["vy"])

        # 2. TESPİT OLASILIĞI (Pd)
        if np.random.rand() > sensor.pd:
            track_dropped = True
            continue

        if track_dropped:
            local_track_id = f"{sensor.name}-T{np.random.randint(1000, 9999)}"
            track_dropped = False

        tq = sensor.base_tq
        if is_maneuvering(target_name, t):
            tq -= sensor.maneuver_tq_penalty
        tq += np.random.randint(-sensor.tq_jitter, sensor.tq_jitter + 1)
        tq = int(np.clip(tq, TQ_MIN, TQ_MAX))

        sigma_pos = tq_to_sigma_pos(tq)
        sigma_vel = tq_to_sigma_vel(tq)

        # 3. SİSTEMATİK HATA (BIAS) VE RASTGELE GÜRÜLTÜ EKLENMESİ
        x_meas = x_true + np.random.normal(0, sigma_pos) + sensor.bias_x
        y_meas = y_true + np.random.normal(0, sigma_pos) + sensor.bias_y
        vx_meas = vx_true + np.random.normal(0, sigma_vel)
        vy_meas = vy_true + np.random.normal(0, sigma_vel)

        records.append({
            "time": round(t, 2),
            "sensor": sensor.name,
            "local_track_id": local_track_id,
            "true_target": target_name,
            "x": x_meas,
            "y": y_meas,
            "vx": vx_meas,
            "vy": vy_meas,
            "track_quality": tq,
            "sigma_pos_m": sigma_pos,
            "sigma_vel_mps": sigma_vel,
        })

    return pd.DataFrame(records)


def generate_sensor_clutter(sensor: SensorProfile, t_max: float) -> pd.DataFrame:
    """Sensörün menzili içinde rastgele sahte ölçümler (clutter) üretir."""
    records = []
    t = np.random.uniform(0, sensor.revisit_interval)
    
    while t <= t_max:
        t += max(0.1, np.random.normal(sensor.revisit_interval, sensor.jitter))
        if t > t_max:
            break
            
        # Poisson dağılımı ile bu taramadaki sahte alarm sayısını belirle
        num_clutter = np.random.poisson(sensor.clutter_rate)
        
        for _ in range(num_clutter):
            # Menzil içinde rastgele bir konum
            angle = np.random.uniform(0, 2 * math.pi)
            r = np.random.uniform(0, sensor.max_range)
            cx = sensor.x + r * math.cos(angle)
            cy = sensor.y + r * math.sin(angle)
            
            # Rastgele, genellikle mantıksız hızlar
            cvx = np.random.normal(0, 80)
            cvy = np.random.normal(0, 80)
            
            # Clutter genellikle düşük TQ'ya sahiptir, ama bazen yanıltıcı olabilir
            tq = np.random.randint(TQ_MIN, TQ_MIN + 6)
            sigma_pos = tq_to_sigma_pos(tq)
            sigma_vel = tq_to_sigma_vel(tq)
            
            records.append({
                "time": round(t, 2),
                "sensor": sensor.name,
                "local_track_id": f"{sensor.name}-CLUT-{np.random.randint(1000, 9999)}",
                "true_target": "CLUTTER", # Hata analizi yaparken filtreleyebilmen için
                "x": cx,
                "y": cy,
                "vx": cvx,
                "vy": cvy,
                "track_quality": tq,
                "sigma_pos_m": sigma_pos,
                "sigma_vel_mps": sigma_vel,
            })
            
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. VERİ ÜRETİMİ VE BİRLEŞTİRİLMESİ
# ---------------------------------------------------------------------------

all_tracks = []

# Gerçek hedefleri üret
for target_name, gt_df in ground_truth.items():
    for sensor in SENSORS:
        df = generate_sensor_tracks(sensor, target_name, gt_df)
        if not df.empty:
            all_tracks.append(df)

# Clutter (Sahte alarmları) üret
for sensor in SENSORS:
    df_clutter = generate_sensor_clutter(sensor, T_TOTAL)
    if not df_clutter.empty:
        all_tracks.append(df_clutter)

sensor_tracks_df = pd.concat(all_tracks, ignore_index=True)
sensor_tracks_df = sensor_tracks_df.sort_values("time").reset_index(drop=True)

ground_truth_df = pd.concat(
    [df.assign(target=name) for name, df in ground_truth.items()], ignore_index=True
)

sensor_tracks_df.to_csv("link16_sensor_tracks_hard.csv", index=False)
ground_truth_df.to_csv("ground_truth_hard.csv", index=False)


if __name__ == "__main__":
    print(sensor_tracks_df.head(15).to_string(index=False))
    print(f"\nToplam sensor track kaydi: {len(sensor_tracks_df)}")
    
    print("\nGerçek vs Clutter Dağılımı:")
    print(sensor_tracks_df['true_target'].value_counts())
    
    print("\nSensör Başina Kayit Sayisi:")
    print(sensor_tracks_df['sensor'].value_counts())