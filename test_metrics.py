#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.insert(0, '.')
from fusion_evaluation import compute_target_specific_metrics, compute_tracking_metrics

# Test gerçekçi advanced fusion
scenarios = [
    ('res_ideal_basic.csv', 'İdealize + Temel'),
    ('res_ideal_adv.csv', 'İdealize + Gelişmiş CI'),
    ('res_real_basic.csv', 'Gerçekçi + Temel'),
    ('res_real_adv.csv', 'Gerçekçi + Gelişmiş (CI)'),
    ('res_real_adv_noci.csv', 'Gerçekçi + Gelişmiş (No-CI)'),
]

gt_df = pd.read_csv('ground_truth_adsb_multi.csv')
gt_key = 'callsign' if 'callsign' in gt_df.columns else 'target'
targets = sorted(gt_df[gt_key].dropna().unique())

print("\n" + "="*100)
print(" "*20 + "TARGET METRİKLERİ (TÜM HEDEFLER - KARŞILAŞTIRMA)")
print("="*100)
print(f"Hedefler: {targets}")

for scenario_csv, scenario_name in scenarios:
    try:
        fused_df = pd.read_csv(scenario_csv)
        print(f"\n{scenario_name:30s}:")
        for target in targets:
            metrics = compute_target_specific_metrics(gt_df, fused_df, target)
            f1 = metrics.get('f1_score', metrics.get('f1', 0))
            mota = metrics.get('mota', 0)
            rmse_pos = metrics.get('rmse_pos_m', metrics.get('rmse_pos', 0))
            print(f"  {target:12s}: F1={f1:.3f}, MOTA={mota:.3f}, RMSE={rmse_pos:.1f}m")
    except Exception as e:
        print(f"\n{scenario_name:30s}: ERROR - {str(e)[:60]}")

print("\n" + "="*100 + "\n")
