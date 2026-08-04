import unittest
from pathlib import Path

import pandas as pd

from fusion_evaluation import compute_tracking_metrics
from stonesoup_benchmark import load_stonesoup_data, run_stonesoup_benchmark


PROJECT_ROOT = Path(__file__).resolve().parent


class StoneSoupIntegrationTests(unittest.TestCase):
    def test_stonesoup_files_match_expected_contract(self):
        gt_df, radar_df = load_stonesoup_data()
        self.assertFalse(gt_df.empty)
        self.assertFalse(radar_df.empty)
        self.assertTrue({"time", "x", "vx", "y", "vy"}.issubset(gt_df.columns))
        self.assertTrue(
            {"time", "sensor", "local_track_id", "x", "vx", "y", "vy"}.issubset(
                radar_df.columns
            )
        )
        gt_ids = set(gt_df["callsign"].astype(str))
        radar_ids = set(
            radar_df.loc[radar_df["callsign_true"] != "CLUTTER", "callsign_true"].astype(str)
        )
        self.assertEqual(radar_ids, gt_ids)

    def test_explicit_timeline_counts_missing_output_as_false_negative(self):
        gt_df = pd.DataFrame(
            [
                {"time": 0.0, "callsign": "T1", "x": 0.0, "vx": 1.0, "y": 0.0, "vy": 0.0},
                {"time": 2.0, "callsign": "T1", "x": 2.0, "vx": 1.0, "y": 0.0, "vy": 0.0},
            ]
        )
        fused_df = pd.DataFrame(
            [{"time": 0.0, "global_track_id": "G1", "x": 0.0, "vx": 1.0, "y": 0.0, "vy": 0.0}]
        )
        result = compute_tracking_metrics(gt_df, fused_df, evaluation_times=[0.0, 1.0, 2.0])
        self.assertEqual(result["total_tp"], 1)
        self.assertEqual(result["total_fn"], 2)
        self.assertAlmostEqual(result["recall"], 1 / 3)

    def test_three_stonesoup_algorithms_run(self):
        result = run_stonesoup_benchmark(
            str(PROJECT_ROOT / "ground_truth_adsb_multi.csv"),
            str(PROJECT_ROOT / "radar_measurements_stonesoup.csv"),
        )

        self.assertEqual(
            list(result.index), ["Basic (CV)", "Advanced (CA+CI)", "IMM (CI)"]
        )
        self.assertTrue(
            {"precision", "recall", "f1_score", "rmse_pos_m"}.issubset(result.columns)
        )


if __name__ == "__main__":
    unittest.main()
