import os
import pandas as pd
import matplotlib.pyplot as plt

from fusion_advanced import run_advanced_fusion
from fusion_basic import run_basic_fusion
from fusion_evaluation import compute_tracking_metrics
from radar_sim import generate_sensor_csvs

# ==========================================
# KONFİGÜRASYON
# ==========================================
GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"
IDEALIZE_SENSOR_CSV = "radar_sensor_tracks_idealize.csv"
GERCEKCI_SENSOR_CSV = "radar_sensor_tracks_gercekci.csv"
RES_IDEAL_BASIC = "res_ideal_basic.csv"
RES_IDEAL_ADV = "res_ideal_adv.csv"
RES_REAL_BASIC = "res_real_basic.csv"
RES_REAL_ADV = "res_real_adv.csv"


def main():
    print("=" * 70)
    print("RADAR FÜZYON BÜTÜNLEŞİK TEST ORKESTRASYONU")
    print("=" * 70)

    print("\n[AŞAMA 1] Simülasyon Verileri Üretiliyor...")
    generate_sensor_csvs(
        gt_csv=GROUND_TRUTH_CSV,
        idealize_csv=IDEALIZE_SENSOR_CSV,
        gercekci_csv=GERCEKCI_SENSOR_CSV,
    )

    print("\n[AŞAMA 2] İdealize Radar Senaryosu Koşuluyor...")
    run_basic_fusion(sensor_csv=IDEALIZE_SENSOR_CSV, output_csv=RES_IDEAL_BASIC)
    run_advanced_fusion(sensor_csv=IDEALIZE_SENSOR_CSV, output_csv=RES_IDEAL_ADV)

    print("\n[AŞAMA 3] Gerçekçi Radar Senaryosu Koşuluyor...")
    run_basic_fusion(sensor_csv=GERCEKCI_SENSOR_CSV, output_csv=RES_REAL_BASIC)
    run_advanced_fusion(sensor_csv=GERCEKCI_SENSOR_CSV, output_csv=RES_REAL_ADV)

    print("\n[AŞAMA 4] Metrikler Hesaplanıyor ve Karşılaştırılıyor...")
    try:
        gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    except FileNotFoundError:
        print(f"HATA: {GROUND_TRUTH_CSV} bulunamadı.")
        return

    dfs = {
        "İdealize + Temel": pd.read_csv(RES_IDEAL_BASIC) if os.path.exists(RES_IDEAL_BASIC) else pd.DataFrame(),
        "İdealize + Gelişmiş": pd.read_csv(RES_IDEAL_ADV) if os.path.exists(RES_IDEAL_ADV) else pd.DataFrame(),
        "Gerçekçi + Temel": pd.read_csv(RES_REAL_BASIC) if os.path.exists(RES_REAL_BASIC) else pd.DataFrame(),
        "Gerçekçi + Gelişmiş": pd.read_csv(RES_REAL_ADV) if os.path.exists(RES_REAL_ADV) else pd.DataFrame(),
    }

    metrics = {name: compute_tracking_metrics(gt_df, df) for name, df in dfs.items()}

    comp_df = pd.DataFrame(metrics).T
    comp_df = comp_df[['precision', 'recall', 'f1_score', 'mota', 'id_switches', 'rmse_pos_m', 'rmse_vel_mps', 'nees']]
    comp_df.columns = ['Precision', 'Recall', 'F1 Score', 'MOTA', 'ID Switch', 'RMSE Pos', 'RMSE Vel', 'NEES']

    print("\n" + "=" * 90)
    print("4'LÜ KOMBİNASYON METRİK KARŞILAŞTIRMASI".center(90))    
    print("=" * 90)
    print(comp_df.round(3).to_string())
    print("=" * 90)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Füzyon Algoritmaları Tüm Senaryolar XY Konum Karşılaştırması", fontsize=16, fontweight="bold")

    plot_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (name, df), (row, col) in zip(dfs.items(), plot_coords):
        ax = axes[row, col]
        ax.set_title(name)

        for cs in gt_df['callsign'].unique() if 'callsign' in gt_df.columns else gt_df['target'].unique():
            sub = gt_df[gt_df.get('callsign', gt_df.get('target')) == cs].sort_values('time')
            ax.plot(sub['x'], sub['y'], 'k--', linewidth=2, alpha=0.6)

        if not df.empty and 'global_track_id' in df.columns:
            for tid in df['global_track_id'].unique():
                tdf = df[df['global_track_id'] == tid].sort_values('time')
                ax.plot(tdf['x'], tdf['y'], linewidth=2, label=f"{tid}")
                ax.scatter(tdf['x'], tdf['y'], s=10)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('tum_senaryolar_xy.png', dpi=150)
    print("\n-> Karşılaştırma grafiği 'tum_senaryolar_xy.png' olarak kaydedildi.")
    plt.show()


if __name__ == '__main__':
    main()
