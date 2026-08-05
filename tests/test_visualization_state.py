import unittest
from pathlib import Path

from visualization.realtime_state import RealtimeMapState, load_snapshot


class RealtimeMapStateTests(unittest.TestCase):
    def test_default_radar_buffer_keeps_latest_twenty(self):
        state = RealtimeMapState()
        for index in range(30):
            state.add_radar({"measurement_id": f"m{index}"})
        radar = state.snapshot()["radar_measurements"]
        self.assertEqual(len(radar), 20)
        self.assertEqual(radar[0]["measurement_id"], "m10")

    def test_configurable_radar_limits(self):
        for limit in (10, 20, 50, 100):
            state = RealtimeMapState(max_radar_measurements=limit)
            for index in range(limit + 5):
                state.add_radar({"measurement_id": str(index)})
            self.assertEqual(len(state.snapshot()["radar_measurements"]), limit)

    def test_fused_history_is_bounded_per_track_and_not_mixed(self):
        state = RealtimeMapState(max_fused_per_track=3)
        for index in range(5):
            state.add_fused({"track_id": "A", "index": index})
            state.add_fused({"track_id": "B", "index": index})
        snapshot = state.snapshot()["fused_tracks"]
        self.assertEqual([row["index"] for row in snapshot["A"]], [2, 3, 4])
        self.assertEqual([row["index"] for row in snapshot["B"]], [2, 3, 4])

    def test_snapshot_is_copy_and_atomic_file_is_readable(self):
        state = RealtimeMapState()
        state.add_radar({"measurement_id": "original"})
        copied = state.snapshot()
        copied["radar_measurements"][0]["measurement_id"] = "changed"
        self.assertEqual(state.snapshot()["radar_measurements"][0]["measurement_id"], "original")
        path = Path.cwd() / ".test_realtime_map_state.json"
        try:
            state.write_snapshot(path)
            self.assertEqual(load_snapshot(path)["radar_measurements"][0]["measurement_id"], "original")
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
