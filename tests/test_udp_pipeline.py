import socket
import time
import unittest
from pathlib import Path

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
        snapshot_path = Path.cwd() / ".test_udp_pipeline_state.json"
        try:
            service = FusionRuntime(RuntimeConfig(
                radar_udp_host="127.0.0.1", radar_udp_port=0,
                fused_udp_host="127.0.0.1", fused_udp_port=fused_port,
                state_snapshot_path=str(snapshot_path),
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
            sender.close()
            service.stop()
            self.assertFalse(service.worker.is_alive())
            self.assertFalse(service.receiver.thread.is_alive())
        finally:
            snapshot_path.unlink(missing_ok=True)
            snapshot_path.with_suffix(snapshot_path.suffix + ".tmp").unlink(missing_ok=True)
        listener.close()


if __name__ == "__main__":
    unittest.main()
