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

# Now check per-target metrics
from fusion_evaluation import compute_target_specific_metrics
gt_df = pd.read_csv('ground_truth_adsb_multi.csv')
fused_df = pd.read_csv('res_quick_test.csv')
gt_key = 'callsign' if 'callsign' in gt_df.columns else 'target'
targets = sorted(gt_df[gt_key].dropna().unique())

print("="*100)
print(" "*30 + "PER-TARGET METRICS (QUICK TEST)")
print("="*100 + "\n")
print(f"Targets: {targets}\n")

for target in targets:
    metrics = compute_target_specific_metrics(gt_df, fused_df, target)
    print(f"{target}:")
    print(f"  F1 Score:   {metrics.get('f1_score', 0):.3f}")
    print(f"  Precision:  {metrics.get('precision', 0):.3f}")
    print(f"  Recall:     {metrics.get('recall', 0):.3f}")
    print(f"  MOTA:       {metrics.get('mota', 0):.3f}")
    print(f"  RMSE Pos:   {metrics.get('rmse_pos_m', 0):.1f} m")
    print(f"  Total TP:   {metrics.get('total_tp', 0)}")
    print(f"  Total FP:   {metrics.get('total_fp', 0)}")
    print(f"  Total FN:   {metrics.get('total_fn', 0)}")
    print()

print("\n" + "="*100 + "\n")
