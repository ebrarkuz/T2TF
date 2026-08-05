import unittest
import pandas as pd

from realtime.message_schema import validate_radar_message
from tools.radar_udp_replay import row_to_message
from visualization.desktop_state import load_ground_truth


class DataAndReplayTests(unittest.TestCase):
    def test_packaged_ground_truth_loads(self):
        ground_truth = load_ground_truth("data/ground_truth_adsb_multi.csv")
        self.assertTrue(ground_truth.routes)
        first_route = next(iter(ground_truth.routes.values()))
        self.assertTrue({"time", "x", "y", "z"}.issubset(first_route[0]))

    def test_csv_row_converts_to_valid_json_schema(self):
        row = pd.read_csv("data/radar_sensor_tracks_gercekci.csv", nrows=1).iloc[0]
        message = validate_radar_message(row_to_message(row, 7, 1760000000.125))
        self.assertEqual(message["sequence_number"], 7)
        self.assertEqual(message["coordinate_frame"], "ENU")
        self.assertEqual(message["source_track_id"], str(row["local_track_id"]))
        self.assertTrue(message["measurement_id"])


if __name__ == "__main__":
    unittest.main()
