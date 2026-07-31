import argparse
import os

import pandas as pd

from fusion_advanced import run_advanced_fusion
from fusion_basic import run_basic_fusion
from fusion_evaluation import compute_target_specific_metrics, compute_tracking_metrics
from fusion_imm import run_imm_fusion
from radar_sim import generate_sensor_csvs


# Legacy ADS-B simulation inputs
GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"
IDEALIZE_SENSOR_CSV = "radar_sensor_tracks_idealize.csv"
GERCEKCI_SENSOR_CSV = "radar_sensor_tracks_gercekci.csv"

RES_IDEAL_BASIC = "res_ideal_basic.csv"
RES_IDEAL_ADV = "res_ideal_adv.csv"
RES_REAL_BASIC = "res_real_basic.csv"
RES_REAL_ADV = "res_real_adv.csv"
RES_REAL_ADV_NOCI = "res_real_adv_noci.csv"
RES_REAL_IMM = "res_real_imm.csv"

# Stone Soup inputs and outputs
# Both simulators are scored against the exact same four-target truth.
STONESOUP_GT_CSV = GROUND_TRUTH_CSV
STONESOUP_RADAR_CSV = "radar_measurements_stonesoup.csv"
RES_STONESOUP_BASIC = "res_stonesoup_basic.csv"
RES_STONESOUP_ADV = "res_stonesoup_advanced.csv"
RES_STONESOUP_IMM = "res_stonesoup_imm.csv"
STONESOUP_METRICS_CSV = "stonesoup_metrics.csv"
STONESOUP_TARGET_METRICS_CSV = "stonesoup_target_metrics.csv"
ADSB_METRICS_CSV = "adsb_metrics.csv"
ADSB_TARGET_METRICS_CSV = "adsb_target_metrics.csv"
ALL_DATASETS_METRICS_CSV = "all_datasets_metrics.csv"
ALL_DATASETS_TARGET_METRICS_CSV = "all_datasets_target_metrics.csv"

GT_REQUIRED_COLUMNS = {"time", "x", "vx", "y", "vy"}
RADAR_REQUIRED_COLUMNS = {
    "time", "sensor", "local_track_id", "x", "vx", "y", "vy"
}

METRIC_COLUMNS = [
    "precision", "id_precision", "recall", "f1_score", "id_f1", "mota",
    "id_switches", "rmse_pos_m", "rmse_x_m", "rmse_y_m", "rmse_z_m",
    "rmse_vel_mps", "rmse_vx_mps", "rmse_vy_mps", "rmse_vz_mps", "nees",
]


def _safe_read_csv(path):
    try:
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_input(df, required_columns, path, kind):
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{kind} dosyasında eksik sütunlar var ({path}): {missing}")
    if df.empty:
        raise ValueError(f"{kind} dosyası boş: {path}")

    numeric = [c for c in ("time", "x", "vx", "y", "vy", "z", "vz") if c in df]
    if df[numeric].isna().any().any():
        bad = df[numeric].columns[df[numeric].isna().any()].tolist()
        raise ValueError(f"{kind} dosyasında boş sayısal değerler var: {bad}")


def load_stonesoup_data(gt_csv=STONESOUP_GT_CSV, radar_csv=STONESOUP_RADAR_CSV):
    """Load and validate the CSV contract shared by all three algorithms."""
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(
            f"Ortak dört hedefli ground truth bulunamadı: {gt_csv}."
        )
    if not os.path.exists(radar_csv):
        raise FileNotFoundError(
            f"Stone Soup radar verisi bulunamadı: {radar_csv}. Önce `python stone.py` çalıştırın."
        )

    gt_df = pd.read_csv(gt_csv)
    radar_df = pd.read_csv(radar_csv)
    _validate_input(gt_df, GT_REQUIRED_COLUMNS, gt_csv, "Ground truth")
    _validate_input(radar_df, RADAR_REQUIRED_COLUMNS, radar_csv, "Radar")

    if "callsign" not in gt_df and "target" not in gt_df:
        gt_df = gt_df.copy()
        gt_df["callsign"] = "TARGET_001"

    gt_key = "callsign" if "callsign" in gt_df else "target"
    if "callsign_true" in radar_df:
        gt_ids = set(gt_df[gt_key].dropna().astype(str))
        measured_ids = set(
            radar_df.loc[
                radar_df["callsign_true"].astype(str) != "CLUTTER", "callsign_true"
            ].dropna().astype(str)
        )
        unknown_ids = sorted(measured_ids - gt_ids)
        if unknown_ids:
            raise ValueError(
                "Radar ve ground-truth aynı senaryoya ait değil. "
                f"Ground-truth içinde bulunmayan radar hedefleri: {unknown_ids}"
            )

    return gt_df.sort_values("time"), radar_df.sort_values("time")


def _metrics_table(gt_df, radar_df, scenarios, max_match_distance=500.0):
    # All algorithms are scored at every sensor scan, including scans where an
    # algorithm emitted no confirmed track.
    evaluation_times = radar_df["time"].dropna().unique().tolist()
    metrics = {
        name: compute_tracking_metrics(
            gt_df,
            fused_df,
            max_match_distance=max_match_distance,
            evaluation_times=evaluation_times,
        )
        for name, fused_df in scenarios.items()
    }
    return pd.DataFrame(metrics).T.reindex(columns=METRIC_COLUMNS)


def _target_metrics_table(gt_df, radar_df, scenarios, max_match_distance=500.0):
    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    evaluation_times = radar_df["time"].dropna().unique().tolist()
    rows = []
    targets = sorted(gt_df[gt_key].dropna().unique())
    for algorithm, fused_df in scenarios.items():
        prepared_frames = {
            t: frame.reset_index(drop=True)
            for t, frame in fused_df.groupby("time", sort=False)
        }
        for target in targets:
            values = compute_target_specific_metrics(
                gt_df,
                fused_df,
                target,
                max_match_distance=max_match_distance,
                evaluation_times=evaluation_times,
                prepared_fused_frames=prepared_frames,
            )
            rows.append({"target": target, "algorithm": algorithm, **values})
    return pd.DataFrame(rows)


def run_stonesoup_benchmark(
    gt_csv=STONESOUP_GT_CSV,
    radar_csv=STONESOUP_RADAR_CSV,
    max_match_distance=500.0,
    print_report=True,
):
    """Run Basic, Advanced and IMM fairly on the same Stone Soup data."""
    gt_df, radar_df = load_stonesoup_data(gt_csv, radar_csv)
    print("\n[1/3] Stone Soup verileri doğrulandı")
    print(f"  Ground truth: {len(gt_df)} satır | Radar: {len(radar_df)} ölçüm")

    print("\n[2/3] Üç füzyon algoritması çalıştırılıyor...")
    scenarios = {
        "Basic (CV)": run_basic_fusion(radar_csv, RES_STONESOUP_BASIC, verbose=False),
        "Advanced (CA+CI)": run_advanced_fusion(
            radar_csv, RES_STONESOUP_ADV, verbose=False, use_ci=True
        ),
        "IMM (CI)": run_imm_fusion(
            radar_csv,
            RES_STONESOUP_IMM,
            verbose=False,
            use_ci=True,
            include_source_measurement_details=False,
        ),
    }

    print("\n[3/3] Ortak radar zaman çizelgesinde metrikler hesaplanıyor...")
    comparison = _metrics_table(gt_df, radar_df, scenarios, max_match_distance)
    target_metrics = _target_metrics_table(gt_df, radar_df, scenarios, max_match_distance)
    comparison.to_csv(STONESOUP_METRICS_CSV)
    target_metrics.to_csv(STONESOUP_TARGET_METRICS_CSV, index=False)

    if print_report:
        _print_comparison("STONE SOUP FÜZYON PERFORMANS KARŞILAŞTIRMASI", comparison)
        _print_target_metrics("STONE SOUP - HEDEF BAZLI METRİKLER", target_metrics)
        print(f"\nÖzet metrikler: {STONESOUP_METRICS_CSV}")
        print(f"Hedef bazlı metrikler: {STONESOUP_TARGET_METRICS_CSV}")
    return comparison


def _print_comparison(title, comparison):
    print("\n" + "=" * 120)
    print(title.center(120))
    print("=" * 120)
    print(comparison.round(3).to_string())
    print("=" * 120)


def _print_target_metrics(title, target_metrics):
    """Print the original per-target comparison tables."""
    display_columns = [
        "precision", "recall", "f1_score", "coverage", "mota", "id_switches",
        "rmse_pos_m", "rmse_x_m", "rmse_y_m", "rmse_z_m",
        "rmse_vel_mps", "rmse_vx_mps", "rmse_vy_mps", "rmse_vz_mps", "nees",
    ]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "coverage": "Coverage",
        "mota": "MOTA",
        "id_switches": "ID Switch",
        "rmse_pos_m": "RMSE Pos(m)",
        "rmse_x_m": "RMSE X",
        "rmse_y_m": "RMSE Y",
        "rmse_z_m": "RMSE Z",
        "rmse_vel_mps": "RMSE Vel(m/s)",
        "rmse_vx_mps": "RMSE VX",
        "rmse_vy_mps": "RMSE VY",
        "rmse_vz_mps": "RMSE VZ",
        "nees": "NEES",
    }
    print("\n" + "#" * 120)
    print(title.center(120))
    print("#" * 120)
    for target, frame in target_metrics.groupby("target", sort=True):
        available = [column for column in display_columns if column in frame.columns]
        table = frame.set_index("algorithm")[available].rename(columns=labels)
        print("\n" + "-" * 120)
        print(f"HEDEF: {target}".center(120))
        print("-" * 120)
        print(table.round(3).to_string())


def run_adsb_benchmark(max_match_distance=500.0, generate_data=False, print_report=True):
    """Run the same three algorithms on the user-generated realistic data."""
    if generate_data or not os.path.exists(GERCEKCI_SENSOR_CSV):
        step1_generate_data()

    if not os.path.exists(GROUND_TRUTH_CSV):
        raise FileNotFoundError(f"Ground truth bulunamadı: {GROUND_TRUTH_CSV}")
    if not os.path.exists(GERCEKCI_SENSOR_CSV):
        raise FileNotFoundError(f"Gerçekçi radar verisi bulunamadı: {GERCEKCI_SENSOR_CSV}")

    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    radar_df = pd.read_csv(GERCEKCI_SENSOR_CSV)
    _validate_input(gt_df, GT_REQUIRED_COLUMNS, GROUND_TRUTH_CSV, "Ground truth")
    _validate_input(radar_df, RADAR_REQUIRED_COLUMNS, GERCEKCI_SENSOR_CSV, "Radar")

    print("\n[KENDİ VERİN] Üç füzyon algoritması çalıştırılıyor...")
    scenarios = {
        "Basic (CV)": run_basic_fusion(GERCEKCI_SENSOR_CSV, RES_REAL_BASIC, False),
        "Advanced (CA+CI)": run_advanced_fusion(
            GERCEKCI_SENSOR_CSV, RES_REAL_ADV, False, True
        ),
        "IMM (CI)": run_imm_fusion(
            GERCEKCI_SENSOR_CSV, RES_REAL_IMM, False, True, False
        ),
    }
    comparison = _metrics_table(gt_df, radar_df, scenarios, max_match_distance)
    target_metrics = _target_metrics_table(gt_df, radar_df, scenarios, max_match_distance)
    comparison.to_csv(ADSB_METRICS_CSV)
    target_metrics.to_csv(ADSB_TARGET_METRICS_CSV, index=False)

    if print_report:
        _print_comparison("KENDİ ÜRETTİĞİN VERİ - FÜZYON PERFORMANSI", comparison)
        _print_target_metrics("KENDİ VERİN - HEDEF BAZLI METRİKLER", target_metrics)
        print(f"\nÖzet metrikler: {ADSB_METRICS_CSV}")
        print(f"Hedef bazlı metrikler: {ADSB_TARGET_METRICS_CSV}")
    return comparison


def run_all_benchmarks(
    gt_csv,
    radar_csv,
    max_match_distance=500.0,
    regenerate_adsb=False,
):
    """Compare both datasets with an identical algorithm/metric pipeline."""
    own_results = run_adsb_benchmark(
        max_match_distance=max_match_distance,
        generate_data=regenerate_adsb,
        print_report=False,
    )
    stonesoup_results = run_stonesoup_benchmark(
        gt_csv,
        radar_csv,
        max_match_distance=max_match_distance,
        print_report=False,
    )
    combined = pd.concat(
        {"Kendi Ürettiğim Veri": own_results, "Stone Soup": stonesoup_results},
        names=["Veri Kaynağı", "Algoritma"],
    )
    combined.to_csv(ALL_DATASETS_METRICS_CSV)
    own_target_metrics = pd.read_csv(ADSB_TARGET_METRICS_CSV)
    stonesoup_target_metrics = pd.read_csv(STONESOUP_TARGET_METRICS_CSV)
    combined_target_metrics = pd.concat(
        [
            own_target_metrics.assign(dataset="Kendi Ürettiğim Veri"),
            stonesoup_target_metrics.assign(dataset="Stone Soup"),
        ],
        ignore_index=True,
    )
    combined_target_metrics.to_csv(ALL_DATASETS_TARGET_METRICS_CSV, index=False)
    _print_comparison("KENDİ VERİN + STONE SOUP ORTAK PERFORMANS KIYASLAMASI", combined)
    _print_target_metrics("KENDİ VERİN - 4 HEDEF İÇİN AYRI METRİKLER", own_target_metrics)
    _print_target_metrics("STONE SOUP - HEDEF BAZLI METRİKLER", stonesoup_target_metrics)
    print(f"\nBirleşik rapor: {ALL_DATASETS_METRICS_CSV}")
    print(f"Birleşik hedef raporu: {ALL_DATASETS_TARGET_METRICS_CSV}")
    print(f"Kendi veri raporu: {ADSB_METRICS_CSV}")
    print(f"Stone Soup raporu: {STONESOUP_METRICS_CSV}")
    return combined


# The original ADS-B workflow is kept for backward compatibility.
def step1_generate_data():
    print("\n[AŞAMA 1] ADS-B tabanlı radar verileri üretiliyor...")
    generate_sensor_csvs(
        gt_csv=GROUND_TRUTH_CSV,
        idealize_csv=IDEALIZE_SENSOR_CSV,
        gercekci_csv=GERCEKCI_SENSOR_CSV,
    )


def step2_run_scenarios():
    print("\n[AŞAMA 2] Eski ADS-B senaryoları çalıştırılıyor...")
    run_basic_fusion(IDEALIZE_SENSOR_CSV, RES_IDEAL_BASIC, False)
    run_advanced_fusion(IDEALIZE_SENSOR_CSV, RES_IDEAL_ADV, False, True)
    run_basic_fusion(GERCEKCI_SENSOR_CSV, RES_REAL_BASIC, False)
    run_advanced_fusion(GERCEKCI_SENSOR_CSV, RES_REAL_ADV, False, True)
    run_advanced_fusion(GERCEKCI_SENSOR_CSV, RES_REAL_ADV_NOCI, False, False)
    run_imm_fusion(GERCEKCI_SENSOR_CSV, RES_REAL_IMM, False, True, False)


def step3_compute_metrics():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    radar_df = pd.read_csv(GERCEKCI_SENSOR_CSV)
    scenarios = {
        "İdealize + Basic": _safe_read_csv(RES_IDEAL_BASIC),
        "İdealize + Advanced": _safe_read_csv(RES_IDEAL_ADV),
        "Gerçekçi + Basic": _safe_read_csv(RES_REAL_BASIC),
        "Gerçekçi + Advanced (CI)": _safe_read_csv(RES_REAL_ADV),
        "Gerçekçi + Advanced (No-CI)": _safe_read_csv(RES_REAL_ADV_NOCI),
        "Gerçekçi + IMM": _safe_read_csv(RES_REAL_IMM),
    }
    result = _metrics_table(gt_df, radar_df, scenarios)
    print(result.round(3).to_string())
    return result


def step4_compute_target_metrics():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    radar_df = pd.read_csv(GERCEKCI_SENSOR_CSV)
    scenarios = {
        "Basic": _safe_read_csv(RES_REAL_BASIC),
        "Advanced": _safe_read_csv(RES_REAL_ADV),
        "IMM": _safe_read_csv(RES_REAL_IMM),
    }
    result = _target_metrics_table(gt_df, radar_df, scenarios)
    print(result.round(3).to_string(index=False))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Radar füzyon performans karşılaştırması")
    parser.add_argument(
        "--source", choices=("both", "stonesoup", "adsb"), default="both",
        help="Veri kaynağı (varsayılan: both)",
    )
    parser.add_argument("--gt-csv", default=STONESOUP_GT_CSV)
    parser.add_argument("--radar-csv", default=STONESOUP_RADAR_CSV)
    parser.add_argument("--max-match-distance", type=float, default=500.0)
    parser.add_argument(
        "--regenerate-adsb",
        action="store_true",
        help="Kendi radar CSV'lerini ground truth'tan yeniden üret",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("RADAR FÜZYON - PERFORMANS KIYASLAMASI".center(70))
    print("=" * 70)
    if args.source == "both":
        run_all_benchmarks(
            args.gt_csv,
            args.radar_csv,
            args.max_match_distance,
            regenerate_adsb=args.regenerate_adsb,
        )
    elif args.source == "stonesoup":
        run_stonesoup_benchmark(args.gt_csv, args.radar_csv, args.max_match_distance)
    else:
        run_adsb_benchmark(args.max_match_distance, generate_data=args.regenerate_adsb)


if __name__ == "__main__":
    main()
