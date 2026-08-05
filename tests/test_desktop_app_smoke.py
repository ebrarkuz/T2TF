import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from realtime.runtime_config import RuntimeConfig
from visualization.desktop_app import DesktopWindow


class DesktopAppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_starts_offscreen_and_worker_stops_on_close(self):
        window = DesktopWindow(RuntimeConfig(
            telemetry_udp_host="127.0.0.1", telemetry_udp_port=0,
            visualization_refresh_ms=100,
        ))
        window.show()
        deadline = time.time() + 3.0
        while window.udp_worker.bound_port is None and time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNotNone(window.udp_worker.bound_port)
        window.close()
        self.app.processEvents()
        self.assertFalse(window.udp_thread.isRunning())


if __name__ == "__main__":
    unittest.main()
