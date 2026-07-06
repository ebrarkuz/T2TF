import os
import pandas as pd

from fusion_advanced import run_advanced_fusion
from fusion_basic import run_basic_fusion
from fusion_evaluation import compute_tracking_metrics
from radar_sim import generate_sensor_csvs

# ==========================================
# KONFİGÜRASYON
# =========================================
GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"
IDEALIZE_SENSOR_CSV = "radar_sensor_tracks_idealize.csv"
GERCEKCI_SENSOR_CSV = "radar_sensor_tracks_gercekci.csv"

# Çıktı Dosyaları
RES_IDEAL_BASIC = "res_ideal_basic.csv"
RES_IDEAL_ADV = "res_ideal_adv.csv"
RES_REAL_BASIC = "res_real_basic.csv"
RES_REAL_ADV = "res_real_adv.csv"
RES_REAL_ADV_NOCI = "res_real_adv_noci.csv"


def step1_generate_data():
    """Ground truth verisinden radar simülasyon verileri üretir."""
    print("\n[AŞAMA 1] Simülasyon verileri üretiliyor...")
    generate_sensor_csvs(
        gt_csv=GROUND_TRUTH_CSV,
        idealize_csv=IDEALIZE_SENSOR_CSV,
        gercekci_csv=GERCEKCI_SENSOR_CSV,
    )


def step2_run_scenarios():
    """Temel ve gelişmiş füzyon senaryolarını çalıştırır."""
    print("\n[AŞAMA 2] Füzyon senaryoları çalıştırılıyor...")

    run_basic_fusion(sensor_csv=IDEALIZE_SENSOR_CSV, output_csv=RES_IDEAL_BASIC, verbose=False)
    run_advanced_fusion(sensor_csv=IDEALIZE_SENSOR_CSV, output_csv=RES_IDEAL_ADV, use_ci=True, verbose=False)

    run_basic_fusion(sensor_csv=GERCEKCI_SENSOR_CSV, output_csv=RES_REAL_BASIC, verbose=False)
    run_advanced_fusion(sensor_csv=GERCEKCI_SENSOR_CSV, output_csv=RES_REAL_ADV, use_ci=True, verbose=False)
    run_advanced_fusion(sensor_csv=GERCEKCI_SENSOR_CSV, output_csv=RES_REAL_ADV_NOCI, use_ci=False, verbose=False)


def _safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def step3_compute_metrics():
    """Metrikleri hesaplar ve tablo halinde ekrana basar."""
    print("\n[AŞAMA 3] Metrikler hesaplanıyor...")

    if not os.path.exists(GROUND_TRUTH_CSV):
        raise FileNotFoundError(f"Ground truth dosyası bulunamadı: {GROUND_TRUTH_CSV}")

    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    dfs = {
        "İdealize + Temel": _safe_read_csv(RES_IDEAL_BASIC),
        "İdealize + Gelişmiş (CI)": _safe_read_csv(RES_IDEAL_ADV),
        "Gerçekçi + Temel": _safe_read_csv(RES_REAL_BASIC),
        "Gerçekçi + Gelişmiş (CI)": _safe_read_csv(RES_REAL_ADV),
        "Gerçekçi + Gelişmiş (No-CI)": _safe_read_csv(RES_REAL_ADV_NOCI),
    }

    metrics = {name: compute_tracking_metrics(gt_df, df) for name, df in dfs.items()}
    comp_df = pd.DataFrame(metrics).T
    comp_df = comp_df[["precision", "recall", "f1_score", "mota", "id_switches", "rmse_pos_m", "rmse_vel_mps", "nees"]]
    comp_df.columns = ["Precision", "Recall", "F1 Score", "MOTA", "ID Switch", "RMSE Pos", "RMSE Vel", "NEES"]

    print("\n" + "=" * 90)
    print("Füzyon kombinasyonları metrik tablosu".center(90))
    print("=" * 90)
    print(comp_df.round(3).to_string())
    print("=" * 90)

    return comp_df


def main():
    print("=" * 70)
    print("RADAR FÜZYON - METRİK TABLOSU".center(70))
    print("=" * 70)

    step1_generate_data()
    step2_run_scenarios()
    step3_compute_metrics()


if __name__ == '__main__':
    main()          