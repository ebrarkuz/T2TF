import unittest

from visualization.realtime_map import _fused_hover, _radar_hover


class RealtimeMapHoverTests(unittest.TestCase):
    def test_radar_hover_contains_identity_xyz_and_timestamps(self):
        hover = _radar_hover({
            "measurement_id": "m1", "sensor_id": "R1", "sequence_number": 7,
            "timestamp": 10.0, "received_at": 10.1,
            "position": {"x_m": 1, "y_m": 2, "z_m": 3},
            "velocity": {"vx_mps": 4, "vy_mps": 5, "vz_mps": 6},
            "measurement": {}, "used_in_fusion": True, "assigned_track_id": "GT-1",
        })
        for value in ("m1", "R1", "X/Y/Z", "1 / 2 / 3", "10.0", "10.1", "GT-1"):
            self.assertIn(value, hover)

    def test_fused_hover_contains_used_measurement_and_correct_delta(self):
        hover = _fused_hover({
            "track_id": "GT-1", "track_status": "confirmed",
            "state_timestamp": 10.25, "publish_timestamp": 10.3,
            "position": {"x_m": 1, "y_m": 2, "z_m": 3},
            "velocity": {"vx_mps": 4, "vy_mps": 5, "vz_mps": 6},
            "fusion_metadata": {
                "update_type": "measurement_update", "filter_name": "IMM",
                "model_probabilities": {"CV": 0.5},
                "used_measurements": [{
                    "measurement_id": "m1", "sensor_id": "R1",
                    "measurement_timestamp": 10.2, "sequence_number": 7,
                }],
            },
        })
        for value in ("GT-1", "X/Y/Z", "1 / 2 / 3", "m1", "R1", "Δt=0.050s"):
            self.assertIn(value, hover)


if __name__ == "__main__":
    unittest.main()
