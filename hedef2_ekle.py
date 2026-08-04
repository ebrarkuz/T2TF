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
# Keskin ve agresif manevralı rota
# ============================================================

df4 = df1.copy()
df4["callsign"] = "HEDEF_4"

# Hedef 4 diğerlerinden ayrılsın diye X'te aynalanmış uçsun,
# Y'de ise çok sert S manevraları çizsin
T_boleni_sharp = 50.0
Genlik_sharp = 15000.0

df4["x"] = merkez_x - (df1["x"] - merkez_x)
df4["vx"] = -df1["vx"]

df4["y"] = (
    df1["y"]
    + np.sin(t_norm / T_boleni_sharp) * Genlik_sharp
)

df4["vy"] = (
    df1["vy"]
    + np.cos(t_norm / T_boleni_sharp)
    * (Genlik_sharp / T_boleni_sharp)
)

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