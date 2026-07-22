#!/usr/bin/env python3
import sys, pandas as pd
sys.path.insert(0, '.')

fused_df = pd.read_csv('res_quick_test.csv')
print(f'Fused measurements: {len(fused_df)}')
print(f'Unique times: {len(fused_df["time"].unique())}')
print(f'Time range: {fused_df["time"].min()} to {fused_df["time"].max()}')
print(f'Columns: {list(fused_df.columns)}')

gt_df = pd.read_csv('ground_truth_adsb_multi.csv')
print(f'\nGT measurements: {len(gt_df)}')
print(f'GT columns: {list(gt_df.columns)}')

from fusion_evaluation import compute_target_specific_metrics
gt_key = 'callsign' if 'callsign' in gt_df.columns else 'target'
targets = sorted(gt_df[gt_key].dropna().unique())
print(f'\nComputing metrics for targets: {targets}')

for target in targets:
    metrics = compute_target_specific_metrics(gt_df, fused_df, target)
    print(f'\n{target}:')
    print(f"F1 Score:     {metrics.get('f1_score', 0):.3f}")
    print(f"Precision:    {metrics.get('precision', 0):.3f}")
    print(f"Recall:       {metrics.get('recall', 0):.3f}")
    print(f"MOTA:         {metrics.get('mota', 0):.3f}")
    print(f"RMSE Pos:     {metrics.get('rmse_pos_m', 0):.1f} m")
    print(f"Total TP:     {metrics.get('total_tp', 0)}")
    print(f"Total FP:     {metrics.get('total_fp', 0)}")
    print(f"Total FN:     {metrics.get('total_fn', 0)}")
