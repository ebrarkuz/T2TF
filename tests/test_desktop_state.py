import unittest

from visualization.desktop_state import (
    DesktopState, format_fused_details, format_radar_details, load_ground_truth,
    split_track_segments,
)


def radar(index=0):
    return {
        "message_type": "radar_measurement", "measurement_id": f"m{index}",
        "sensor_id": "RADAR_A", "source_track_id": "A-1", "timestamp": float(index),
        "sequence_number": index,
        "position": {"x_m": index, "y_m": index + 1, "z_m": index + 2},
        "velocity": {"vx_mps": 3, "vy_mps": 4, "vz_mps": 5},
        "assigned_track_id": "GT-1", "rejection_reason": None,
    }


def fused(track_id="GT-1", timestamp=1.0):
    return {
        "message_type": "fused_track", "track_id": track_id,
        "track_status": "confirmed", "state_timestamp": timestamp,
        "publish_timestamp": timestamp + 0.01,
        "position": {"x_m": timestamp, "y_m": 2 * timestamp, "z_m": 3000},
        "velocity": {"vx_mps": 10, "vy_mps": 11, "vz_mps": 2},
        "fusion_metadata": {
            "filter_name": "Dual-IMM", "update_type": "measurement_update",
            "dominant_model": "XY:CV|Z:CV", "used_measurements": [{
                "measurement_id": "radar-12", "sensor_id": "RADAR_A",
                "measurement_timestamp": timestamp - 0.02, "sequence_number": 12,
            }],
        },
    }


class DesktopStateTests(unittest.TestCase):
    def test_default_buffer_keeps_last_twenty_and_limits_are_changeable(self):
        state = DesktopState()
        for index in range(30):
            state.add_message(radar(index))
        self.assertEqual([m["measurement_id"] for m in state.snapshot()["radar_measurements"]],
                         [f"m{i}" for i in range(10, 30)])
        for limit in (10, 20, 50, 100):
            state.set_radar_limit(limit)
            for index in range(110):
                state.add_message(radar(index))
            self.assertEqual(len(state.snapshot()["radar_measurements"]), limit)

    def test_tracks_are_separate_bounded_and_time_sorted(self):
        state = DesktopState(max_fused_per_track=2)
        state.add_message(fused("A", 3))
        state.add_message(fused("B", 2))
        state.add_message(fused("A", 1))
        snapshot = state.snapshot()["fused_tracks"]
        self.assertEqual(set(snapshot), {"A", "B"})
        self.assertEqual([p["state_timestamp"] for p in snapshot["A"]], [1, 3])

    def test_large_time_gap_splits_segments(self):
        segments = split_track_segments([fused(timestamp=8), fused(timestamp=1), fused(timestamp=2)], 2)
        self.assertEqual([[p["state_timestamp"] for p in s] for s in segments], [[1, 2], [8]])

    def test_hover_details_include_xyz_and_all_used_measurements(self):
        self.assertIn("X / Y / Z: 0 / 1 / 2", format_radar_details(radar()))
        details = format_fused_details(fused())
        for expected in ("X / Y / Z", "radar-12", "RADAR_A", "Sequence: 12", "Delta t: 0.020"):
            self.assertIn(expected, details)

    def test_ground_truth_loads_and_missing_file_is_nonfatal(self):
        loaded = load_ground_truth("data/ground_truth_adsb_multi.csv")
        self.assertTrue(loaded.routes)
        missing = load_ground_truth("data/does-not-exist.csv")
        self.assertEqual(missing.routes, {})
        self.assertIsNotNone(missing.warning)


if __name__ == "__main__":
    unittest.main()
