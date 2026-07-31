import pandas as pd
import numpy as np


def enu_to_latlon(x, y, ref_lat, ref_lon):
    """ENU koordinatlarından lat/lon'a dönüş"""
    R = 6371000.0
    lat = ref_lat + (y / R) * 180.0 / np.pi
    lon = ref_lon + (x / (R * np.cos(ref_lat * np.pi / 180.0))) * 180.0 / np.pi
    return lat, lon


# 1. Mevcut tek uçaklı veriyi oku
KEEP_COLS = [
    "time", "x", "y", "z",
    "vx", "vy", "vz",
    "callsign",
    "lat", "lon", "alt_m", "time_unix"
]

df1 = pd.read_csv("ground_truth_adsb.csv")[KEEP_COLS].copy()
df1["callsign"] = "HEDEF_1"


# Referans noktalarını al
ref_lat = df1["lat"].median()
ref_lon = df1["lon"].median()

merkez_x = df1["x"].mean()
merkez_y = df1["y"].mean()


# ============================================================
# 2. İkinci uçak
# Aynalanmış lineer rota
# ============================================================

df2 = df1.copy()
df2["callsign"] = "HEDEF_2"

df2["x"] = merkez_x - (df1["x"] - merkez_x)
df2["vx"] = -df1["vx"]

df2[["lat", "lon"]] = df2.apply(
    lambda row: pd.Series(
        enu_to_latlon(
            row["x"],
            row["y"],
            ref_lat,
            ref_lon
        )
    ),
    axis=1
)


# ============================================================
# 3. Üçüncü uçak
# Yumuşak manevralı rota
# ============================================================

df3 = df1.copy()
df3["callsign"] = "HEDEF_3"

t_norm = df3["time"] - df3["time"].min()

# Y'de ters uçsun
df3["y"] = merkez_y - (df1["y"] - merkez_y)
df3["vy"] = -df1["vy"]

# X'te yumuşak sinüs manevrası
T_boleni_soft = 100.0
Genlik_soft = 3000.0

df3["x"] = (
    df1["x"]
    + np.sin(t_norm / T_boleni_soft) * Genlik_soft
)

df3["vx"] = (
    df1["vx"]
    + np.cos(t_norm / T_boleni_soft)
    * (Genlik_soft / T_boleni_soft)
)

df3[["lat", "lon"]] = df3.apply(
    lambda row: pd.Series(
        enu_to_latlon(
            row["x"],
            row["y"],
            ref_lat,
            ref_lon
        )
    ),
    axis=1
)


# ============================================================
# 4. Dördüncü uçak
# Fiziksel olarak daha gerçekçi agresif manevralı rota
# Coordinated Turn benzeri model
# ============================================================

df4 = df1.copy()
df4["callsign"] = "HEDEF_4"

times = df4["time"].to_numpy(dtype=float)

# Başlangıç konumu:
# Hedef 2 gibi X ekseninde aynalanmış başlasın
x0 = float(
    merkez_x
    - (df1["x"].iloc[0] - merkez_x)
)

y0 = float(df1["y"].iloc[0])

# Başlangıç hızını mevcut veriden türet
vx0 = -float(df1["vx"].iloc[0])
vy0 = float(df1["vy"].iloc[0])

initial_speed = np.hypot(vx0, vy0)

# Çok düşük / çok yüksek ise sınırla
initial_speed = np.clip(
    initial_speed,
    180.0,
    300.0
)

initial_heading = np.arctan2(
    vy0,
    vx0
)


# ------------------------------------------------------------
# MANEVRA PARAMETRELERİ
# ------------------------------------------------------------

# Modern yüksek performanslı uçak için agresif fakat makul
MAX_TURN_RATE_DEG_S = 8.0

MAX_TURN_RATE_RAD_S = np.deg2rad(
    MAX_TURN_RATE_DEG_S
)

# Turn-rate'in yumuşak ama belirgin biçimde sağ-sol değişmesi
TURN_PERIOD_S = 100.0

# Hızın küçük miktarda değişmesine izin ver
BASE_SPEED_MPS = float(initial_speed)

SPEED_VARIATION_MPS = 20.0

SPEED_PERIOD_S = 150.0


# ------------------------------------------------------------
# HEDEF 4 TRAJEKTORYASINI ENTEGRE ET
# ------------------------------------------------------------

x_arr = np.zeros(len(df4))
y_arr = np.zeros(len(df4))

vx_arr = np.zeros(len(df4))
vy_arr = np.zeros(len(df4))

heading_arr = np.zeros(len(df4))
speed_arr = np.zeros(len(df4))

x_arr[0] = x0
y_arr[0] = y0

heading_arr[0] = initial_heading

speed_arr[0] = BASE_SPEED_MPS

vx_arr[0] = (
    speed_arr[0]
    * np.cos(heading_arr[0])
)

vy_arr[0] = (
    speed_arr[0]
    * np.sin(heading_arr[0])
)


for i in range(1, len(df4)):

    dt = times[i] - times[i - 1]

    if dt <= 0:
        dt = 1e-3

    t = times[i] - times[0]

    # --------------------------------------------------------
    # Turn rate
    # Sağ-sol dönüşler sinüzoidal değişiyor
    # Fakat fiziksel olarak sınırlı
    # --------------------------------------------------------

    turn_rate = (
        MAX_TURN_RATE_RAD_S
        * np.sin(
            2.0 * np.pi
            * t
            / TURN_PERIOD_S
        )
    )

    # Heading entegrasyonu
    heading_arr[i] = (
        heading_arr[i - 1]
        + turn_rate * dt
    )

    # --------------------------------------------------------
    # Hız büyüklüğünde küçük değişim
    # --------------------------------------------------------

    speed_arr[i] = (
        BASE_SPEED_MPS
        + SPEED_VARIATION_MPS
        * np.sin(
            2.0 * np.pi
            * t
            / SPEED_PERIOD_S
        )
    )

    speed_arr[i] = np.clip(
        speed_arr[i],
        170.0,
        330.0
    )

    # --------------------------------------------------------
    # Heading + speed -> velocity
    # --------------------------------------------------------

    vx_arr[i] = (
        speed_arr[i]
        * np.cos(heading_arr[i])
    )

    vy_arr[i] = (
        speed_arr[i]
        * np.sin(heading_arr[i])
    )

    # --------------------------------------------------------
    # Velocity -> position
    # Trapezoidal integration
    # --------------------------------------------------------

    x_arr[i] = (
        x_arr[i - 1]
        + 0.5
        * (vx_arr[i - 1] + vx_arr[i])
        * dt
    )

    y_arr[i] = (
        y_arr[i - 1]
        + 0.5
        * (vy_arr[i - 1] + vy_arr[i])
        * dt
    )


# Hesaplanan trajectory'i dataframe'e aktar
df4["x"] = x_arr
df4["y"] = y_arr

df4["vx"] = vx_arr
df4["vy"] = vy_arr


# ------------------------------------------------------------
# Dikey hareketi mevcut ADS-B ground truth'tan koruyoruz
# ------------------------------------------------------------

df4["z"] = df1["z"].to_numpy()
df4["vz"] = df1["vz"].to_numpy()

df4["alt_m"] = df1["alt_m"].to_numpy()


# ------------------------------------------------------------
# Yeni ENU koordinatlarından lat/lon oluştur
# ------------------------------------------------------------

df4[["lat", "lon"]] = df4.apply(
    lambda row: pd.Series(
        enu_to_latlon(
            row["x"],
            row["y"],
            ref_lat,
            ref_lon
        )
    ),
    axis=1
)


# ============================================================
# 5. Dört uçağın verilerini birleştir
# ============================================================

df_multi = pd.concat(
    [
        df1,
        df2,
        df3,
        df4
    ]
).sort_values(
    "time"
).reset_index(
    drop=True
)


# ============================================================
# 6. Yeni veri setini kaydet
# ============================================================

df_multi.to_csv(
    "ground_truth_adsb_multi.csv",
    index=False
)

print(
    "Çoklu hedef verisi başarıyla oluşturuldu: "
    "ground_truth_adsb_multi.csv (4 Hedef)"
)

print()
print("HEDEF_4 parametreleri:")
print(
    f"  Maksimum turn rate : "
    f"{MAX_TURN_RATE_DEG_S:.1f} deg/s"
)
print(
    f"  Ortalama hız       : "
    f"{BASE_SPEED_MPS:.1f} m/s"
)
print(
    f"  Hız değişimi       : "
    f"±{SPEED_VARIATION_MPS:.1f} m/s"
)
print(
    f"  Dönüş periyodu     : "
    f"{TURN_PERIOD_S:.1f} s"
)