"""
Link 16 Tarzı Track Quality (TQ) ile Asenkron Çoklu-Sensör Track Simülatörü
============================================================================

Senaryo:
- 2 hedef: biri doğrusal/az manevralı (TARGET_1_LINEAR),
           diğeri dairesel/manevralı (TARGET_2_MANEUVER, periyodik dönüşlerle).
- Birden fazla ASENKRON sensör (her birinin kendi revisit periyodu ve jitter'ı var).
- NO-FEEDBACK mimari: her sensör kendi lokal track ID'sini bağımsız üretir,
  merkezi bir füzyon/düzeltme döngüsünden besleme almaz.
- Her sensör çıktısı: zaman, konum (x,y), hız (vx,vy), lokal track ID,
  ve Link 16 benzeri 1-15 arası Track Quality (TQ) değeri içerir.
- TQ <-> ölçüm kovaryansı dönüşüm fonksiyonları, association/fusion
  hesaplamalarında doğrudan kullanılabilecek şekilde ayrıca sağlanmıştır.

Çıktılar:
- ground_truth.csv          : Her iki hedefin gerçek (simüle) trajectory'si
- link16_sensor_tracks.csv  : Tüm sensörlerin ürettiği asenkron track akışı
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


def generate_linear_target(ts, x0=0.0, y0=0.0, vx=120.0, vy=40.0, noise_std=0.4):
    """Doğrusal, az manevralı hedef: sabit hız modeli + küçük process noise."""
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


def generate_maneuvering_target(ts, cx=20000.0, cy=20000.0, speed=150.0,
                                 radius=8000.0, noise_std=1.5, turn_period=60.0):
    """Dairesel/manevralı hedef: koordineli dönüş modeli.
    Belirli aralıklarla (turn_period) dönüş yönü değiştirilerek
    S-manevrası benzeri gerçekçi manevralar üretilir."""
    n = len(ts)
    states = np.zeros((n, 4))
    omega = speed / radius  # açısal hız (rad/s)
    theta = 0.0
    for i, t in enumerate(ts):
        if i > 0 and i % int(turn_period / DT) == 0:
            omega *= -1.0  # dönüş yönü değişimi -> manevra
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
    """Hedefin o anda manevra yapıp yapmadığını belirler (TQ düşüşü için kullanılır).
    Gerçek sistemde ivme/yön değişiminden hesaplanır; burada senaryo bilgisini
    doğrudan kullanıyoruz."""
    if target_name != "TARGET_2_MANEUVER":
        return False
    cycle_pos = t % turn_period
    return cycle_pos < 6.0  # dönüş anının etrafındaki birkaç saniye


# ---------------------------------------------------------------------------
# 2. LINK 16 TRACK QUALITY (TQ) <-> KOVARYANS DÖNÜŞÜMÜ
# ---------------------------------------------------------------------------
# TQ: 1 (en kötü/güvenilmez) ... 15 (en iyi/yüksek güven), Link 16 ölçeğine benzer.
# TQ'dan ölçüm std sapmasına (pozisyon ve hız için ayrı ayrı) üstel bir
# interpolasyon kullanılıyor; istenirse tabloya/başka fonksiyona çevrilebilir.

TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX = 1500.0   # TQ=1 icin pozisyon std sapmasi (m)  -> en kotu
SIGMA_POS_MIN = 30.0     # TQ=15 icin pozisyon std sapmasi (m) -> en iyi
SIGMA_VEL_MAX = 25.0     # TQ=1 icin hiz std sapmasi (m/s)
SIGMA_VEL_MIN = 0.5      # TQ=15 icin hiz std sapmasi (m/s)


def tq_to_sigma_pos(tq) -> float:
    """TQ -> pozisyon olcum std sapmasi (metre)."""
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac


def tq_to_sigma_vel(tq) -> float:
    """TQ -> hiz olcum std sapmasi (m/s)."""
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    return SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac


def sigma_pos_to_tq(sigma_pos: float) -> int:
    """Ters donusum: bir pozisyon std sapmasindan 'efektif TQ' turetmek icin.
    Association/fusion sirasinda guncellenen belirsizlikten TQ raporlamak
    istenirse kullanilabilir."""
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))


def tq_to_R(tq) -> np.ndarray:
    """TQ -> 4x4 olcum kovaryans matrisi (x, vx, y, vy sirasiyla).
    Association (Mahalanobis/likelihood) ve fusion (information-weighted /
    covariance intersection) hesaplamalarinda dogrudan kullanilabilir."""
    sp = tq_to_sigma_pos(tq)
    sv = tq_to_sigma_vel(tq)
    return np.diag([sp ** 2, sv ** 2, sp ** 2, sv ** 2])


def tq_to_weight(tq) -> float:
    """TQ'yu dogrudan agirlik olarak kullanmak isteyenler icin basit
    normalize edilebilir agirlik (gercek Link16 fusion pratiginde
    w_i = TQ_i / sum(TQ_j) seklinde kullanilir)."""
    return float(np.clip(tq, TQ_MIN, TQ_MAX))


# ---------------------------------------------------------------------------
# 3. SENSOR TANIMLARI (ASENKRON, NO-FEEDBACK, LOKAL TRACK ID)
# ---------------------------------------------------------------------------

@dataclass
class SensorProfile:
    name: str
    revisit_interval: float        # ortalama guncelleme periyodu (s)
    jitter: float                  # zamanlama duzensizligi (s, std)
    base_tq: int                   # bu sensorun tipik/baseline TQ degeri
    tq_jitter: int                 # TQ'nun zamanla rastgele oynama miktari (+-)
    pd: float                      # tespit olasiligi (track surekliligini etkiler)
    maneuver_tq_penalty: int = 0   # hedef manevra yaparken TQ dususu


SENSORS: List[SensorProfile] = [
    # Hizli donen / kisa menzilli izleme radari: sik guncelleme, yuksek kalite
    SensorProfile(name="RADAR_A", revisit_interval=2.0, jitter=0.3,
                  base_tq=13, tq_jitter=1, pd=0.95, maneuver_tq_penalty=3),
    # Orta hizli, genel maksatli arama radari
    SensorProfile(name="RADAR_B", revisit_interval=5.0, jitter=0.8,
                  base_tq=10, tq_jitter=2, pd=0.90, maneuver_tq_penalty=3),
    # Yavas donen, uzun menzilli erken ihbar radari: seyrek ama daha gurultulu
    SensorProfile(name="RADAR_C", revisit_interval=10.0, jitter=2.0,
                  base_tq=7, tq_jitter=2, pd=0.80, maneuver_tq_penalty=2),
]


def generate_sensor_tracks(sensor: SensorProfile, target_name: str,
                            gt_df: pd.DataFrame) -> pd.DataFrame:
    """Bir sensorun, verilen ground truth hedefe ait ASENKRON, gurultulu
    track akisini uretir. No-feedback: sensor kendi lokal track ID'sini
    bagimsiz atar; bir tespit kaybi (Pd) sonrasi yeniden tespitte yeni bir
    lokal track ID baslatilir (gercek sensor davranisini taklit eder)."""

    records = []
    local_track_id = f"{sensor.name}-T{np.random.randint(1000, 9999)}"
    t = float(gt_df["time"].iloc[0]) + np.random.uniform(0, sensor.revisit_interval)
    t_max = float(gt_df["time"].iloc[-1])

    track_dropped = False

    while t <= t_max:
        # asenkron ornekleme: ortalama periyot + jitter (Gaussian)
        t += max(0.1, np.random.normal(sensor.revisit_interval, sensor.jitter))
        if t > t_max:
            break

        # ground truth'a zaman ekseninde enterpolasyon
        x_true = np.interp(t, gt_df["time"], gt_df["x"])
        y_true = np.interp(t, gt_df["time"], gt_df["y"])
        vx_true = np.interp(t, gt_df["time"], gt_df["vx"])
        vy_true = np.interp(t, gt_df["time"], gt_df["vy"])

        # Pd: tespit basarisiz olursa track dusurulur (drop)
        if np.random.rand() > sensor.pd:
            track_dropped = True
            continue

        # drop sonrasi yeniden tespit -> yeni lokal track ID (gercekci davranis)
        if track_dropped:
            local_track_id = f"{sensor.name}-T{np.random.randint(1000, 9999)}"
            track_dropped = False

        # TQ hesaplama: hedef manevra yapiyorsa kalite duser
        tq = sensor.base_tq
        if is_maneuvering(target_name, t):
            tq -= sensor.maneuver_tq_penalty
        tq += np.random.randint(-sensor.tq_jitter, sensor.tq_jitter + 1)
        tq = int(np.clip(tq, TQ_MIN, TQ_MAX))

        sigma_pos = tq_to_sigma_pos(tq)
        sigma_vel = tq_to_sigma_vel(tq)

        # TQ'ya bagli olcum gurultusu eklenerek sensor track'i uretilir
        x_meas = x_true + np.random.normal(0, sigma_pos)
        y_meas = y_true + np.random.normal(0, sigma_pos)
        vx_meas = vx_true + np.random.normal(0, sigma_vel)
        vy_meas = vy_true + np.random.normal(0, sigma_vel)

        records.append({
            "time": round(t, 2),
            "sensor": sensor.name,
            "local_track_id": local_track_id,
            "true_target": target_name,   # sadece degerlendirme/dogrulama icin
            "x": x_meas,
            "y": y_meas,
            "vx": vx_meas,
            "vy": vy_meas,
            "track_quality": tq,
            "sigma_pos_m": sigma_pos,
            "sigma_vel_mps": sigma_vel,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. TUM SENSOR-HEDEF KOMBINASYONLARI ICIN VERI URETIMI
# ---------------------------------------------------------------------------

all_tracks = []
for target_name, gt_df in ground_truth.items():
    for sensor in SENSORS:
        df = generate_sensor_tracks(sensor, target_name, gt_df)
        all_tracks.append(df)

sensor_tracks_df = pd.concat(all_tracks, ignore_index=True)
sensor_tracks_df = sensor_tracks_df.sort_values("time").reset_index(drop=True)

ground_truth_df = pd.concat(
    [df.assign(target=name) for name, df in ground_truth.items()], ignore_index=True
)

sensor_tracks_df.to_csv("link16_sensor_tracks.csv", index=False)
ground_truth_df.to_csv("ground_truth.csv", index=False)


# ---------------------------------------------------------------------------
# 5. ASSOCIATION/FUSION'DA TQ KULLANIMINA KUCUK BIR ORNEK
# ---------------------------------------------------------------------------
# Bu kisim, ureilen TQ degerlerinin nasil kovaryansa/agirliga cevrilip
# kullanilabilecegini gostermek icindir; gercek association/fusion
# algoritmaniza entegre edilmek uzere bir baslangic noktasidir.

def example_two_sensor_fusion(meas1: dict, meas2: dict) -> dict:
    """Iki ayri sensorden gelen, ayni hedefe ait oldugu varsayilan iki
    track/measurement'i, TQ'lardan turetilen kovaryanslarla
    information-weighted (kovaryans agirlikli) sekilde birlestirir."""
    state1 = np.array([meas1["x"], meas1["vx"], meas1["y"], meas1["vy"]])
    state2 = np.array([meas2["x"], meas2["vx"], meas2["y"], meas2["vy"]])

    R1 = tq_to_R(meas1["track_quality"])
    R2 = tq_to_R(meas2["track_quality"])

    # information (precision) agirlikli ortalama: optimal lineer fuzyon
    info1 = np.linalg.inv(R1)
    info2 = np.linalg.inv(R2)
    fused_info = info1 + info2
    fused_cov = np.linalg.inv(fused_info)
    fused_state = fused_cov @ (info1 @ state1 + info2 @ state2)

    fused_tq = sigma_pos_to_tq(math.sqrt(fused_cov[0, 0]))

    return {
        "x": fused_state[0], "vx": fused_state[1],
        "y": fused_state[2], "vy": fused_state[3],
        "fused_track_quality": fused_tq,
    }


if __name__ == "__main__":
    print(sensor_tracks_df.head(15).to_string(index=False))
    print(f"\nToplam sensor track kaydi: {len(sensor_tracks_df)}")
    print(f"Sensor basina kayit sayisi:\n{sensor_tracks_df['sensor'].value_counts()}")

    # ornek fuzyon: ayni an civarinda 2 farkli sensorden gelen kayitlar
    sample = sensor_tracks_df[sensor_tracks_df["sensor"].isin(["RADAR_A", "RADAR_C"])].head(2)
    if len(sample) == 2:
        fused = example_two_sensor_fusion(sample.iloc[0].to_dict(), sample.iloc[1].to_dict())
        print("\nOrnek 2-sensor fuzyon sonucu (TQ-tabanli kovaryans agirlikli):")
        print(fused)