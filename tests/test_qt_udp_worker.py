import os
import socket
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QThread

from realtime.message_schema import encode_message
from visualization.udp_visualization_receiver import UdpVisualizationWorker


class QtUdpWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def _wait(self, predicate, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_worker_receives_off_ui_thread_and_stops_cleanly(self):
        worker = UdpVisualizationWorker("127.0.0.1", 0)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        messages = []
        worker.message_received.connect(messages.append)
        start = time.perf_counter()
        thread.start()
        self.assertLess(time.perf_counter() - start, 0.1)
        self.assertTrue(self._wait(lambda: worker.bound_port is not None))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(encode_message({
            "schema_version": 1, "message_type": "runtime_status",
            "publish_timestamp": time.time(), "status": {"active_track_count": 1},
        }), ("127.0.0.1", worker.bound_port))
        self.assertTrue(self._wait(lambda: len(messages) == 1))
        worker.stop()
        self.assertTrue(self._wait(lambda: not thread.isRunning()))
        sender.close()


if __name__ == "__main__":
    unittest.main()
