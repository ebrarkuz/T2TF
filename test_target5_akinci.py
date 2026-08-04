import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fusion_imm import run_imm_fusion
from fusion_evaluation import compute_target_specific_metrics
from target5_akinci import (
    TARGET5_CALLSIGN,
    combine_ground_truth,
    combine_sensor_measurements,
    associate_fused_tracks_to_target5,
    generate_target5_akinci_climbing_arc,
    generate_target5_radar_measurements,
    match_fused_to_target5,
)


ROOT = Path(__file__).resolve().parent


class Target5AkinciTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing_gt = pd.read_csv(ROOT / "ground_truth_adsb_multi.csv")
        cls.existing_gt = cls.existing_gt[
            cls.existing_gt["callsign"].astype(str) != TARGET5_CALLSIGN
        ].copy()
        cls.target5 = generate_target5_akinci_climbing_arc(
            other_ground_truth=cls.existing_gt
        )
        cls.sensor = generate_target5_radar_measurements(
            cls.target5, seed=1234, detection_probability=0.9
        )

    def test_generated_data_identity_and_required_fields(self):
        self.assertFalse(self.target5.empty)
        self.assertEqual(set(self.target5["target_id"]), {5})
        self.assertEqual(set(self.target5["callsign"]), {TARGET5_CALLSIGN})
        required = {
            "time", "target_id", "x", "y", "z", "vx", "vy", "vz",
            "speed", "heading", "turn_rate", "climb_rate", "lat", "lon",
        }
        self.assertTrue(required.issubset(self.target5.columns))

    def test_time_altitude_speed_and_spiral_are_physical(self):
        self.assertFalse(self.target5.empty)
        self.assertTrue(self.target5["callsign"].eq("HEDEF_5").all())
        self.assertTrue(self.target5["time"].is_monotonic_increasing)
        self.assertFalse(self.target5["time"].duplicated().any())
        self.assertTrue((np.diff(self.target5["time"]) > 0.0).all())
        self.assertTrue(np.isfinite(
            self.target5[["x", "y", "z", "vx", "vy", "vz"]]
        ).all().all())
        self.assertAlmostEqual(float(self.target5.iloc[0]["z"]), 3000.0)
        self.assertAlmostEqual(float(self.target5.iloc[-1]["z"]), 4880.0, delta=15.0)
        self.assertTrue((self.target5["z"].diff().dropna() > 0.0).all())
        self.assertAlmostEqual(float(self.target5["speed"].mean()), 77.0, delta=0.05)
        self.assertGreater((self.target5["turn_rate"].abs() > 0.0).sum(), 100)
        center_x = self.target5.attrs["center_x_m"]
        center_y = self.target5.attrs["center_y_m"]
        radius = np.hypot(
            self.target5["x"] - center_x, self.target5["y"] - center_y
        )
        self.assertGreater(float(radius.iloc[-1]), float(radius.iloc[0]))
        self.assertAlmostEqual(float(radius.iloc[0]), 500.0, delta=0.1)
        self.assertAlmostEqual(float(radius.iloc[-1]), 1800.0, delta=0.1)
        self.assertAlmostEqual(
            float(self.target5["spiral_theta_rad"].iloc[-1]) / (2.0 * np.pi),
            4.0,
            places=6,
        )
        self.assertAlmostEqual(
            self.target5.attrs["duration_s"],
            self.target5.attrs["total_arc_length_m"] / 77.0,
            places=6,
        )

    def test_minimum_separation_is_preserved(self):
        separations = self.target5.attrs["minimum_separations_m"]
        self.assertEqual(set(separations), {"HEDEF_1", "HEDEF_2", "HEDEF_3", "HEDEF_4"})
        self.assertGreaterEqual(min(separations.values()), 1000.0)

    def test_sensor_measurements_exist_for_every_radar(self):
        self.assertFalse(self.sensor.empty)
        self.assertEqual(set(self.sensor["callsign_true"]), {TARGET5_CALLSIGN})
        self.assertEqual(set(self.sensor["sensor"]), {"RADAR_A", "RADAR_B", "RADAR_C"})
        required = {
            "time", "sensor", "local_track_id", "x", "y", "z", "vx", "vy",
            "vz", "track_quality", "sigma_pos_m", "sigma_vel_mps", "is_clutter",
        }
        self.assertTrue(required.issubset(self.sensor.columns))

    def test_seed_is_reproducible(self):
        repeated = generate_target5_radar_measurements(
            self.target5, seed=1234, detection_probability=0.9
        )
        assert_frame_equal(self.sensor.reset_index(drop=True), repeated.reset_index(drop=True))

    def test_existing_ground_truth_and_sensor_rows_are_unchanged(self):
        combined_gt = combine_ground_truth(self.existing_gt, self.target5)
        recovered_gt = combined_gt[combined_gt["callsign"] != TARGET5_CALLSIGN]
        recovered_gt = recovered_gt[self.existing_gt.columns].reset_index(drop=True)
        expected_gt = self.existing_gt.sort_values(["time", "callsign"], kind="stable")
        expected_gt = expected_gt.reset_index(drop=True)
        assert_frame_equal(expected_gt, recovered_gt, check_dtype=False)

        existing_sensor = pd.read_csv(ROOT / "radar_sensor_tracks_gercekci.csv")
        existing_sensor = existing_sensor[existing_sensor["callsign_true"] != TARGET5_CALLSIGN]
        combined_sensor = combine_sensor_measurements(existing_sensor, self.sensor)
        recovered_sensor = combined_sensor[combined_sensor["callsign_true"] != TARGET5_CALLSIGN]
        recovered_sensor = recovered_sensor[existing_sensor.columns].reset_index(drop=True)
        expected_sensor = existing_sensor.sort_values("time", kind="stable").reset_index(drop=True)
        assert_frame_equal(expected_sensor, recovered_sensor, check_dtype=False)

    def test_imm_fusion_output_and_streamlit_analysis_fields(self):
        sensor_path = ROOT / ".target5_test_sensor.csv"
        fused_path = ROOT / ".target5_test_fused.csv"
        diagnostic_path = ROOT / ".target5_test_diagnostics.csv"
        try:
            self.sensor.to_csv(sensor_path, index=False)
            fused = run_imm_fusion(
                sensor_csv=str(sensor_path), output_csv=str(fused_path), verbose=False,
                diagnostics_csv=str(diagnostic_path),
            )
            self.assertTrue(fused_path.exists())
            self.assertFalse(fused.empty)
            required_fused = {
                "time", "global_track_id", "x", "y", "z", "vx", "vy", "vz",
                "mode_prob_xy_cv", "mode_prob_xy_ca", "mode_prob_xy_ct_left",
                "mode_prob_xy_ct_right", "sigma_x_m", "sigma_y_m", "sigma_z_m",
            }
            self.assertTrue(required_fused.issubset(fused.columns))
            self.assertTrue({"update_used", "is_prediction_only", "time_since_update_s"}.issubset(fused.columns))
            self.assertTrue(diagnostic_path.exists())
            diagnostic = pd.read_csv(diagnostic_path)
            self.assertTrue({"source_d2_xy", "source_gate_xy", "created"}.issubset(diagnostic.columns))
            matched = match_fused_to_target5(self.target5, fused, 400.0)
            self.assertFalse(matched.empty)
            self.assertTrue({"position_error_3d", "velocity_error_3d", "gt_heading"}.issubset(matched.columns))
            associated, summary = associate_fused_tracks_to_target5(
                self.target5, fused, 400.0
            )
            self.assertFalse(summary.empty)
            for _, segment in associated.groupby(["global_track_id", "segment_id"]):
                self.assertTrue(segment["time"].is_monotonic_increasing)
        finally:
            sensor_path.unlink(missing_ok=True)
            fused_path.unlink(missing_ok=True)
            diagnostic_path.unlink(missing_ok=True)

    def test_existing_target_imm_metrics_do_not_change(self):
        sensor_all = pd.read_csv(ROOT / "radar_sensor_tracks_gercekci.csv")
        base = sensor_all[
            (sensor_all["callsign_true"] != TARGET5_CALLSIGN)
            & (sensor_all["time"] <= 152.0)
        ].copy()
        combined = combine_sensor_measurements(base, self.sensor)
        paths = [
            ROOT / ".target5_compat_base_sensor.csv",
            ROOT / ".target5_compat_combined_sensor.csv",
            ROOT / ".target5_compat_base_fused.csv",
            ROOT / ".target5_compat_combined_fused.csv",
        ]
        try:
            base.to_csv(paths[0], index=False)
            combined.to_csv(paths[1], index=False)
            base_fused = run_imm_fusion(str(paths[0]), str(paths[2]), verbose=False)
            combined_fused = run_imm_fusion(str(paths[1]), str(paths[3]), verbose=False)
            evaluation_times = base["time"].unique().tolist()
            for target in ("HEDEF_1", "HEDEF_2", "HEDEF_3", "HEDEF_4"):
                before = compute_target_specific_metrics(
                    self.existing_gt, base_fused, target, 400.0, evaluation_times
                )
                after = compute_target_specific_metrics(
                    self.existing_gt, combined_fused, target, 400.0, evaluation_times
                )
                for metric in ("precision", "recall", "f1_score", "rmse_pos_m"):
                    self.assertAlmostEqual(before[metric], after[metric], places=12)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
