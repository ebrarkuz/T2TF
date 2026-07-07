#!/usr/bin/env python3
import sys, pandas as pd
sys.path.insert(0, '.')

# Quick fusion run (skip idealize for speed)
import fusion_advanced

scenarios = [
    ('fusion_advanced', 'gerçekçi', True),  # (module, name, use_ci)
]

print("\n" + "="*100)
print(" "*30 + "QUICK FUSION TEST (GERÇEKÇİ ONLY)")
print("="*100 + "\n")

for mod_name, desc, use_ci in scenarios:
    print(f"Running {desc} fusion (use_ci={use_ci})...")
    if desc == 'gerçekçi':
        fused_df = fusion_advanced.run_advanced_fusion(
            sensor_csv='radar_sensor_tracks_gercekci.csv',
            output_csv='res_quick_test.csv',
            verbose=False,
            use_ci=use_ci
        )
        print(f"  → {len(fused_df)} fused measurements\n")

# Now check HEDEF_3 metrics
from fusion_evaluation import compute_target_specific_metrics
gt_df = pd.read_csv('ground_truth_adsb_multi.csv')
fused_df = pd.read_csv('res_quick_test.csv')

print("="*100)
print(" "*35 + "HEDEF_3_MANEVRA METRICS (QUICK TEST)")
print("="*100 + "\n")

metrics = compute_target_specific_metrics(gt_df, fused_df, 'HEDEF_3_MANEVRA')
print(f"F1 Score:     {metrics.get('f1_score', 0):.3f}")
print(f"Precision:    {metrics.get('precision', 0):.3f}")
print(f"Recall:       {metrics.get('recall', 0):.3f}")
print(f"MOTA:         {metrics.get('mota', 0):.3f}")
print(f"RMSE Pos:     {metrics.get('rmse_pos_m', 0):.1f} m")
print(f"Total TP:     {metrics.get('total_tp', 0)}")
print(f"Total FP:     {metrics.get('total_fp', 0)}")
print(f"Total FN:     {metrics.get('total_fn', 0)}")

print("\n" + "="*100 + "\n")
