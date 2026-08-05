import time
import unittest
from realtime.fusion_runtime import FusionRuntime
from realtime.message_schema import encode_message, validate_fused_message
from realtime.runtime_config import RuntimeConfig
from tests.test_message_schema import radar_message


class RealtimeFusionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FusionRuntime(RuntimeConfig(
            radar_udp_host="127.0.0.1",
            radar_udp_port=0,
            fused_udp_host="127.0.0.1",
            fused_udp_port=9,
            max_packet_age_s=10.0,
        ))
        self.runtime.started_at = time.time()

    def tearDown(self):
        self.runtime.publisher.close()
        self.runtime.telemetry_publisher.close()

    def test_measurement_identity_is_stored_during_fusion(self):
        now = time.time()
        message = radar_message(timestamp=now, measurement_id="identity-1")
        output = self.runtime.process_datagram(encode_message(message), now, publish=False)
        self.assertEqual(len(output), 1)
        validate_fused_message(output[0])
        used = output[0]["fusion_metadata"]["used_measurements"]
        self.assertEqual(used[0]["measurement_id"], "identity-1")
        self.assertEqual(used[0]["sensor_id"], "RADAR_A")
        self.assertEqual(used[0]["sequence_number"], 1)

    def test_other_tracks_publish_prediction_only(self):
        now = time.time()
        first = radar_message(timestamp=now, measurement_id="a1", sensor_id="RADAR_A", source_track_id="A")
        self.runtime.process_datagram(encode_message(first), now, publish=False)
        second = radar_message(
            timestamp=now + 0.01, measurement_id="b1", sensor_id="RADAR_B",
            source_track_id="B", sequence_number=1,
            position={"x_m": 100000.0, "y_m": 100000.0, "z_m": 3000.0},
        )
        output = self.runtime.process_datagram(encode_message(second), now + 0.01, publish=False)
        prediction_messages = [
            item for item in output
            if item["fusion_metadata"]["update_type"] == "prediction_only"
        ]
        self.assertTrue(prediction_messages)
        self.assertEqual(prediction_messages[0]["fusion_metadata"]["used_measurements"], [])

    def test_duplicate_does_not_reenter_fusion(self):
        now = time.time()
        payload = encode_message(radar_message(timestamp=now))
        self.assertTrue(self.runtime.process_datagram(payload, now, publish=False))
        self.assertEqual(self.runtime.process_datagram(payload, now, publish=False), [])
        self.assertEqual(self.runtime.counters.rejected_packets, 1)

    def test_multiple_used_measurements_are_preserved_in_output_schema(self):
        now = time.time()
        self.runtime.process_datagram(
            encode_message(radar_message(timestamp=now)), now, publish=False
        )
        used = [
            {"measurement_id": "m1", "sensor_id": "R1", "measurement_timestamp": now, "sequence_number": 1},
            {"measurement_id": "m2", "sensor_id": "R2", "measurement_timestamp": now, "sequence_number": 2},
        ]
        state = self.runtime.fusion.process_measurement(self.runtime._measurement_from_message(
            radar_message(timestamp=now + 0.01, measurement_id="identity-2", sequence_number=2)
        ))[0]
        message = self.runtime._fused_message(state, now, used)
        self.assertEqual(message["fusion_metadata"]["used_measurements"], used)


if __name__ == "__main__":
    unittest.main()
