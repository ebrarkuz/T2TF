"""Qt ana thread'ini bloklamayan UDP telemetri worker'i."""

from __future__ import annotations

import socket
import threading

from PySide6.QtCore import QObject, Signal, Slot

from realtime.message_schema import MessageValidationError, decode_json_packet


class UdpVisualizationWorker(QObject):
    message_received = Signal(dict)
    connection_changed = Signal(bool, str)
    finished = Signal()

    def __init__(self, host: str, port: int, max_packet_bytes: int = 65507):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.max_packet_bytes = max_packet_bytes
        self.bound_port: int | None = None
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None

    @Slot()
    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        try:
            sock.settimeout(0.2)
            sock.bind((self.host, self.port))
            self.bound_port = int(sock.getsockname()[1])
            self.connection_changed.emit(True, f"{self.host}:{self.bound_port}")
            while not self._stop_event.is_set():
                try:
                    payload, _address = sock.recvfrom(self.max_packet_bytes + 1)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    message = decode_json_packet(payload, self.max_packet_bytes)
                except MessageValidationError:
                    continue
                if message.get("message_type") in {"radar_measurement", "fused_track", "runtime_status"}:
                    self.message_received.emit(message)
        except OSError as exc:
            self.connection_changed.emit(False, str(exc))
        finally:
            try:
                sock.close()
            except OSError:
                pass
            self._socket = None
            self.connection_changed.emit(False, "kapali")
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop_event.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
