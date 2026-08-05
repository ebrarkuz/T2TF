import socket
import time
import unittest
from realtime.fusion_runtime import FusionRuntime
from realtime.message_schema import decode_json_packet, encode_message, validate_fused_message
from realtime.runtime_config import RuntimeConfig
from tests.test_message_schema import radar_message


class UdpPipelineTests(unittest.TestCase):
    def test_udp_receiver_to_selected_fusion_to_sender_and_ports_close(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(5.0)
        fused_port = listener.getsockname()[1]
        telemetry_listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        telemetry_listener.bind(("127.0.0.1", 0))
        telemetry_listener.settimeout(5.0)
        telemetry_port = telemetry_listener.getsockname()[1]
        try:
            service = FusionRuntime(RuntimeConfig(
                radar_udp_host="127.0.0.1", radar_udp_port=0,
                fused_udp_host="127.0.0.1", fused_udp_port=fused_port,
                telemetry_udp_host="127.0.0.1", telemetry_udp_port=telemetry_port,
                max_packet_age_s=10.0,
            ))
            service.start()
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            now = time.time()
            sender.sendto(
                encode_message(radar_message(timestamp=now, measurement_id="udp-e2e-1")),
                ("127.0.0.1", service.receiver.bound_port),
            )
            payload, _ = listener.recvfrom(65508)
            fused = validate_fused_message(decode_json_packet(payload))
            self.assertEqual(
                fused["fusion_metadata"]["used_measurements"][0]["measurement_id"],
                "udp-e2e-1",
            )
            self.assertTrue(str(fused["track_id"]).startswith("GT-"))
            telemetry_types = set()
            deadline = time.time() + 5.0
            while time.time() < deadline and not {"radar_measurement", "fused_track"}.issubset(telemetry_types):
                telemetry_payload, _ = telemetry_listener.recvfrom(65508)
                telemetry_types.add(decode_json_packet(telemetry_payload)["message_type"])
            self.assertTrue({"radar_measurement", "fused_track"}.issubset(telemetry_types))
            sender.close()
            service.stop()
            self.assertFalse(service.worker.is_alive())
            self.assertFalse(service.receiver.thread.is_alive())
        finally:
            telemetry_listener.close()
        listener.close()


if __name__ == "__main__":
    unittest.main()
