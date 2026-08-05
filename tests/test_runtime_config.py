import os
import unittest
from pathlib import Path
from unittest.mock import patch

from realtime.runtime_config import RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_yaml_profile_and_environment_overrides(self):
        content = """
network:
  radar_port: 7001
  fused_port: 8001
fusion:
  algorithm: basic_cv
  profiles:
    basic_cv:
      gate_threshold: 21.0
    dual_imm:
      CONFIRM_HITS: 5
visualization:
  max_visible_radar_measurements: 20
"""
        path = Path.cwd() / ".test_runtime_config.yaml"
        try:
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, {"RADAR_UDP_PORT": "7777", "FUSION_ALGORITHM": "dual_imm"}, clear=False):
                config = RuntimeConfig.load(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(config.radar_udp_port, 7777)
        self.assertEqual(config.fusion_algorithm, "dual_imm")
        self.assertEqual(config.fusion_parameters["CONFIRM_HITS"], 5)


if __name__ == "__main__":
    unittest.main()
