"""
OpenSky Network - Gercek ADS-B Track Verisi Cekici
====================================================
OpenSky Network'un yeni OAuth2 tabanli ucretsiz API'sini kullanir.

KURULUM:
    pip install requests

HAZIRLIK (bir kez yapilir):
    1) https://opensky-network.org adresine giris yap
    2) Profil -> My OpenSky -> "API Client" bolumune git
    3) "Reset Credential" butonuna tikla -> Client Secret goruntulenir
    4) Client Secret'i asagidaki OPENSKY_CLIENT_SECRET'e yaz
    5) Client ID zaten sayfada yazili: "kullanici_adi-api-client"

HEDEF UCAGI BULMAK:
    FlightRadar24'te uca tikla -> sol panelde "Mode S" veya hex kod gorursun
    Bunu TARGET_ICAO24'e yaz. Ya da callsign ile ara (daha yavas).

CIKTI:
    ground_truth_adsb.csv  (pipeline bu dosyayi okur)
"""

import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List

# ===========================================================================
# YAPILANDIRMA — BURAYA KENDI BILGILERINIZI YAZIN
# ===========================================================================

# OpenSky API Client bilgileri (profil sayfanizdan alin)
OPENSKY_CLIENT_ID     = "ebrar-api-client"   # Client ID (sayfada goruyorsunuz)
OPENSKY_CLIENT_SECRET = "RiOiqaoNORwfLDSwqEvMEPddc3yXFDlJ"                   # Reset Credential ile alinir

# Hangi ucagi takip etmek istiyorsun?
# ICAO24: FlightRadar24'te ucaga tikla -> sol panel -> "Mode S" altindaki hex kod
# Callsign: sefer numarasi (ICAO24 yoksa bununla aranir, daha yavas)
TARGET_ICAO24   = "4BC854"          # ornek: "896190"
TARGET_CALLSIGN = ""   # ornek: "RJA166"

# Kac dakika geri gidecegiz?
# Ucak hala havadaysa ve az once kalktiysa 30-60 dk yeterli.
# Biten bir uçuşun tamamini almak icin daha fazla artir.
LOOKBACK_MINUTES = 90

# Cografi sinir — callsign aramasinda kullanilir
AREA_LAMIN, AREA_LAMAX = -90.0, 90.0
AREA_LOMIN, AREA_LOMAX = -180.0, 180.0

# Cikti dosyasi
OUTPUT_CSV = "ground_truth_adsb4.csv"

# ===========================================================================
# OAUTH2 TOKEN ALMA
# ===========================================================================

BASE_URL   = "https://opensky-network.org/api"
TOKEN_URL  = "https://auth.opensky-network.org/auth/realms/opensky-network/" \
             "protocol/openid-connect/token"

_token_cache = {"token": None, "expires": 0}


def get_token() -> str:
    """Client Credentials akisiyla Bearer token alir (cache'li)."""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires"] - 30:
        return _token_cache["token"]

    if not OPENSKY_CLIENT_SECRET:
        raise RuntimeError(
            "\n[HATA] OPENSKY_CLIENT_SECRET bos!\n"
            "  1) https://opensky-network.org adresine giris yap\n"
            "  2) Profil -> My OpenSky -> API Client bolumu\n"
            "  3) 'Reset Credential' butonuna tikla\n"
            "  4) Gelen Client Secret'i bu dosyaya yaz:\n"
            "     OPENSKY_CLIENT_SECRET = 'buraya_yaz'\n"
        )

    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     OPENSKY_CLIENT_ID,
        "client_secret": OPENSKY_CLIENT_SECRET,
    }, timeout=15)

    if resp.status_code == 401:
        raise RuntimeError(
            "\n[HATA] Kimlik dogrulama basarisiz (401).\n"
            "  -> OPENSKY_CLIENT_ID ve OPENSKY_CLIENT_SECRET'i kontrol edin.\n"
            "  -> Secret yanlis girdiyseniz 'Reset Credential' ile yenileyin.\n"
            f"  -> Sunucu mesaji: {resp.text[:300]}"
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"\n[HATA] Token alinamadi: HTTP {resp.status_code}\n{resp.text[:300]}"
        )

    data = resp.json()
    _token_cache["token"]   = data["access_token"]
    _token_cache["expires"] = now + data.get("expires_in", 300)
    return _token_cache["token"]


def _get(endpoint: str, params: dict) -> dict:
    """Bearer token ile OpenSky REST API'ye GET istegi."""
    headers = {"Authorization": f"Bearer {get_token()}"}
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, params=params, headers=headers, timeout=20)

    if r.status_code == 404:
        return {}
    if r.status_code == 429:
        raise RuntimeError(
            "OpenSky rate limit asildi. Birka dakika bekleyip tekrar deneyin.\n"
            f"Kalan kredi: {r.headers.get('X-Rate-Limit-Remaining', '?')}"
        )
    if r.status_code == 403:
        raise RuntimeError(
            f"Erisim reddedildi (403): {r.text[:200]}\n"
            "Ucretsiz hesapta bazi tarihsel endpoint'ler kisitli olabilir.\n"
            "Su anda aktif ucen bir ucak secin (LOOKBACK_MINUTES kucultun)."
        )
    if r.status_code != 200:
        raise RuntimeError(f"OpenSky API hatasi: HTTP {r.status_code} — {r.text[:300]}")

    return r.json() if r.text else {}


# ===========================================================================
# TRACK CEKME FONKSIYONLARI
# ===========================================================================

def fetch_track(icao24: str, begin_unix: int) -> Optional[list]:
    """
    /tracks/all endpoint'i: tek istekte tum trajectory'yi dondurur.
    path = [[time, lat, lon, baro_alt, true_track, on_ground], ...]
    """
    print(f"  /tracks/all sorgulanıyor: icao24={icao24}, begin={begin_unix}...")
    data = _get("/tracks/all", {"icao24": icao24.lower(), "time": begin_unix})
    path = data.get("path")
    if path:
        callsign = data.get("callsign", icao24).strip() or icao24
        print(f"  {len(path)} nokta alindi. Callsign: {callsign}")
        return path, callsign
    return None, icao24


def find_icao24_by_callsign(callsign: str) -> Optional[str]:
    """
    Canlı durumu (states/all) tarayarak callsign'a gore ICAO24 bulur.
    Ucak o an bu cografi kutunun icinde olmalidir.
    """
    print(f"  Callsign '{callsign}' icin ICAO24 aranıyor (canlı trama)...")
    params = {
        "lamin": AREA_LAMIN, "lamax": AREA_LAMAX,
        "lomin": AREA_LOMIN, "lomax": AREA_LOMAX,
    }
    data = _get("/states/all", params)
    states = data.get("states") or []
    for s in states:
        cs = (s[1] or "").strip()
        if callsign.upper() in cs.upper():
            print(f"  Bulundu: callsign='{cs}' -> ICAO24='{s[0]}'")
            return s[0].lower()
    print(f"  '{callsign}' bu anda bolge icinde bulunamadi.")
    print("  Ipucu: FlightRadar24'te ucaga tikla -> sol panel -> Mode S kodu")
    return None


def states_fallback(icao24: str, begin_unix: int, end_unix: int) -> list:
    """
    /tracks/all bos donerse 5'er dakika araliklarla /states/all sorgular.
    Her sorgu 1 kredi harcar. 60 dk = 12 sorgu.
    """
    print(f"  States fallback: {icao24} icin {LOOKBACK_MINUTES} dk taraniyor...")
    path = []
    step = 300  # 5 dakika
    t = begin_unix
    while t <= end_unix:
        params = {"time": t, "icao24": icao24}
        try:
            data = _get("/states/all", params)
            for s in (data.get("states") or []):
                if s[0].lower() == icao24.lower():
                    if s[6] is not None and s[5] is not None:  # lat, lon
                        path.append([s[3] or t, s[6], s[5], s[7] or 0])
                    break
        except Exception as e:
            print(f"    t={t}: {e}")
        t += step
        time.sleep(0.3)
    return path


# ===========================================================================
# KOORDINAT DONUSUMU & DATAFRAME HAZIRLAMA
# ===========================================================================

def latlon_to_enu(lat, lon, alt_m, ref_lat, ref_lon, ref_alt=0.0):
    R = 6371000.0
    x = R * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = R * math.radians(lat - ref_lat)
    z = float(alt_m or 0) - ref_alt
    return x, y, z


def build_dataframe(path: list, callsign: str) -> pd.DataFrame:
    rows = [{"time_unix": float(p[0]),
             "lat": float(p[1]), "lon": float(p[2]),
             "alt_m": float(p[3]) if p[3] is not None else 0.0}
            for p in path if p[1] is not None and p[2] is not None]

    if len(rows) < 5:
        raise RuntimeError(f"Yeterli gecerli nokta yok: {len(rows)} nokta.")

    df = (pd.DataFrame(rows)
            .sort_values("time_unix")
            .drop_duplicates("time_unix")
            .reset_index(drop=True))

    df["time"] = df["time_unix"] - df["time_unix"].iloc[0]

    ref_lat = df["lat"].median()
    ref_lon = df["lon"].median()
    ref_alt = df["alt_m"].median()

    enu = df.apply(
        lambda r: latlon_to_enu(r["lat"], r["lon"], r["alt_m"], ref_lat, ref_lon, ref_alt),
        axis=1, result_type="expand"
    )
    df["x"], df["y"], df["z"] = enu[0], enu[1], enu[2]

    dt = df["time"].diff().fillna(1.0).clip(lower=0.1)
    df["vx"] = df["x"].diff().fillna(0) / dt
    df["vy"] = df["y"].diff().fillna(0) / dt
    df["vz"] = df["z"].diff().fillna(0) / dt

    # Spike temizleme
    MAX_SPD = 350.0
    for c in ["vx", "vy", "vz"]:
        df[c] = df[c].clip(-MAX_SPD, MAX_SPD).rolling(3, center=True, min_periods=1).mean()

    df["callsign"] = callsign

    print(f"  Sure: {df['time'].max():.0f}s | Nokta: {len(df)} | "
          f"x: {df['x'].min():.0f}..{df['x'].max():.0f} m | "
          f"y: {df['y'].min():.0f}..{df['y'].max():.0f} m")

    return df[["time", "x", "y", "z", "vx", "vy", "vz",
               "callsign", "lat", "lon", "alt_m", "time_unix"]].copy()


# ===========================================================================
# ANA FONKSIYON
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OpenSky Network - Ground Truth Veri Cekici")
    print("=" * 60)
    print(f"  Client ID   : {OPENSKY_CLIENT_ID}")
    print(f"  ICAO24      : {TARGET_ICAO24 or '(callsign ile aranacak)'}")
    print(f"  Callsign    : {TARGET_CALLSIGN or '(ICAO24 kullanilacak)'}")
    print(f"  Lookback    : {LOOKBACK_MINUTES} dk")
    print(f"  Cikti       : {OUTPUT_CSV}\n")

    now_unix   = int(datetime.now(timezone.utc).timestamp())
    begin_unix = now_unix - LOOKBACK_MINUTES * 60

    # --- Token al ---
    print("Token aliniyor...")
    token = get_token()
    print("  [OK] Kimlik dogrulandi.\n")

    # --- ICAO24 belirle ---
    icao24 = TARGET_ICAO24.strip().lower() if TARGET_ICAO24.strip() else None
    callsign = TARGET_CALLSIGN.strip() or "AIRCRAFT"

    if not icao24:
        icao24 = find_icao24_by_callsign(callsign)
        if not icao24:
            print("\n[!] Callsign ile ICAO24 bulunamadi.")
            print("    Cozum: FlightRadar24'te ucaga tikla -> 'Mode S' kutusundaki")
            print("    hex kodu (ornek: 896190) TARGET_ICAO24'e yaz ve tekrar calistir.")
            exit(1)

    # --- Track cek ---
    path, callsign_found = fetch_track(icao24, begin_unix)

    if not path or len(path) < 5:
        print("  /tracks/all yeterli veri vermedi, states fallback deneniyor...")
        path = states_fallback(icao24, begin_unix, now_unix)
        callsign_found = callsign

    if not path or len(path) < 5:
        print(f"\n[HATA] {icao24} icin yeterli veri alinamadi ({len(path) if path else 0} nokta).")
        print("  Oneriler:")
        print("  1) LOOKBACK_MINUTES degerini artirin (ornek: 180)")
        print("  2) Ucak hala havadaysa ICAO24'u FlightRadar24'ten dogrudan alin")
        print("  3) Farkli bir sefer deneyin")
        exit(1)

    # --- DataFrame & kaydet ---
    df = build_dataframe(path, callsign_found)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[OK] {len(df)} nokta '{OUTPUT_CSV}' dosyasina kaydedildi.")
    print("     Simdi su komutu calistirabilirsiniz:")
    print("     python3 adsb_radar_fusion_pipeline.py")