#!/usr/bin/env python3
import sys, pandas as pd
sys.path.insert(0, '.')

from fusion_evaluation import compute_target_specific_metrics

gt_df = pd.read_csv('ground_truth_adsb_multi.csv')

# Test each scenario
scenarios = [
    ('res_ideal_basic.csv', 'Idealize + Basic'),
    ('res_ideal_adv.csv', 'Idealize + Advanced'),
    ('res_real_basic.csv', 'Gercekçi + Basic'),
    ('res_real_adv.csv', 'Gercekçi + Advanced (CI)'),
]

targets = ['HEDEF_2', 'HEDEF_3_MANEVRA', 'PGT549U']

print("\n" + "="*120)
print(" "*35 + "PER-TARGET FUSION METRICS (with Acceleration Model)")
print("="*120 + "\n")

for scenario_file, scenario_name in scenarios:
    print(f"\n{scenario_name}:")
    print("-" * 100)
    try:
        fused_df = pd.read_csv(scenario_file)
        for target in targets:
            metrics = compute_target_specific_metrics(gt_df, fused_df, target)
            f1 = metrics.get('f1_score', 0)
            prec = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            mota = metrics.get('mota', 0)
            rmse_pos = metrics.get('rmse_pos_m', 0)
            tp = metrics.get('total_tp', 0)
            fp = metrics.get('total_fp', 0)
            fn = metrics.get('total_fn', 0)
            print(f"  {target:20s} | F1={f1:.3f} | Prec={prec:.3f} | Recall={recall:.3f} | MOTA={mota:.3f} | TP={tp:4d} FP={fp:4d} FN={fn:4d} | RMSE={rmse_pos:6.1f}m")
    except FileNotFoundError:
        print(f"  (File not found: {scenario_file})")

print("\n" + "="*120)
print("NOTE: Acceleration model implemented in GlobalTrack.propagate()")
print("  - Tracks acceleration from velocity history")
print("  - Low-pass filtered: accel_filt = 0.3*new + 0.7*old")
print("  - Clamped to ±20 m/s² for stability")
print("  - Position predicted as: x = x + v*dt + 0.5*a*dt²")
print("="*120 + "\n")
