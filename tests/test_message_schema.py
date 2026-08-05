import json
import math
import unittest

from realtime.message_schema import (
    RadarMessageValidator,
    encode_message,
    validate_radar_message,
)


def radar_message(**changes):
    message = {
        "schema_version": 1,
        "message_type": "radar_measurement",
        "measurement_id": "radar_0001",
        "sensor_id": "RADAR_A",
        "source_track_id": "RADAR_A-T1",
        "target_hint": None,
        "timestamp": 1000.0,
        "sequence_number": 1,
        "coordinate_frame": "ENU",
        "position": {"x_m": 10.0, "y_m": 20.0, "z_m": 3000.0},
        "velocity": {"vx_mps": 40.0, "vy_mps": 5.0, "vz_mps": 1.0},
        "measurement": {},
        "covariance": None,
        "quality": {},
    }
    message.update(changes)
    return message


class MessageSchemaTests(unittest.TestCase):
    def test_valid_radar_json_is_accepted(self):
        result = RadarMessageValidator(clock=lambda: 1000.5).validate_packet(
            encode_message(radar_message())
        )
        self.assertTrue(result.accepted)

    def test_missing_timestamp_is_rejected(self):
        message = radar_message()
        message.pop("timestamp")
        result = RadarMessageValidator(clock=lambda: 1000.0).validate_packet(
            json.dumps(message).encode()
        )
        self.assertFalse(result.accepted)
        self.assertIn("timestamp", result.reason)

    def test_nan_coordinate_is_rejected(self):
        message = radar_message()
        message["position"]["x_m"] = math.nan
        with self.assertRaises(ValueError):
            encode_message(message)
        result = RadarMessageValidator(clock=lambda: 1000.0).validate_packet(
            json.dumps(message).encode()
        )
        self.assertFalse(result.accepted)

    def test_duplicate_measurement_is_rejected(self):
        validator = RadarMessageValidator(clock=lambda: 1000.1)
        payload = encode_message(radar_message())
        self.assertTrue(validator.validate_packet(payload).accepted)
        duplicate = validator.validate_packet(payload)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate_measurement")

    def test_unsupported_schema_version_is_rejected(self):
        result = RadarMessageValidator(clock=lambda: 1000.0).validate_packet(
            encode_message(radar_message(schema_version=2))
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "unsupported_schema_version")

    def test_regressed_sequence_is_rejected(self):
        validator = RadarMessageValidator(clock=lambda: 1000.1)
        self.assertTrue(validator.validate_packet(encode_message(radar_message(sequence_number=5))).accepted)
        second = radar_message(measurement_id="radar_0002", sequence_number=4)
        self.assertEqual(validator.validate_packet(encode_message(second)).reason, "sequence_number_regressed")


if __name__ == "__main__":
    unittest.main()

