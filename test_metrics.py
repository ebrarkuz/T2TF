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

targets = ["HEDEF_1", "HEDEF_2", "HEDEF_3"]
gt_df = pd.read_csv('ground_truth_adsb_multi.csv')

print("\n" + "="*100)
print(" "*35 + "HEDEF_3 - TARGET METRİKLERİ (KARŞILAŞTIRMA)")
print("="*100)

for scenario_csv, scenario_name in scenarios:
    try:
        fused_df = pd.read_csv(scenario_csv)
        metrics = compute_target_specific_metrics(gt_df, fused_df, 'HEDEF_3')
        f1 = metrics.get('f1_score', metrics.get('f1', 0))
        mota = metrics.get('mota', 0)
        rmse_pos = metrics.get('rmse_pos_m', metrics.get('rmse_pos', 0))
        print(f"\n{scenario_name:30s}: F1={f1:.3f}, MOTA={mota:.3f}, RMSE={rmse_pos:.1f}m")
    except Exception as e:
        print(f"\n{scenario_name:30s}: ERROR - {str(e)[:60]}")

print("\n" + "="*100 + "\n")
