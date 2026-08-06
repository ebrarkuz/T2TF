"""Bounded queue kullanan UDP radar alıcısı."""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReceivedDatagram:
    payload: bytes
    address: tuple[str, int]
    received_at: float
    received_at_monotonic: float


class UdpRadarReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        output_queue: queue.Queue,
        *,
        max_packet_bytes: int = 65507,
        overflow_policy: str = "drop_oldest",
    ):
        self.host = host
        self.port = port
        self.output_queue = output_queue
        self.max_packet_bytes = max_packet_bytes
        self.overflow_policy = overflow_policy
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._bound_port = port
        self.received_count = 0
        self.overflow_count = 0

    @property
    def bound_port(self) -> int:
        if self.socket:
            try:
                return self.socket.getsockname()[1]
            except OSError:
                pass
        return self._bound_port

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self._bound_port = self.socket.getsockname()[1]
        self.socket.settimeout(0.2)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="udp-radar-receiver", daemon=True)
        self.thread.start()
        LOG.info("Radar UDP receiver bind edildi: %s:%s", self.host, self.bound_port)

    def _enqueue(self, datagram: ReceivedDatagram) -> None:
        try:
            self.output_queue.put_nowait(datagram)
            return
        except queue.Full:
            self.overflow_count += 1
            LOG.warning("Radar queue overflow; policy=%s", self.overflow_policy)
        if self.overflow_policy == "drop_oldest":
            try:
                self.output_queue.get_nowait()
                self.output_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(datagram)
            except queue.Full:
                pass

    def _run(self) -> None:
        LOG.info("UDP radar receiver başladı")
        while not self.stop_event.is_set():
            try:
                payload, address = self.socket.recvfrom(self.max_packet_bytes + 1)
            except socket.timeout:
                continue
            except OSError:
                break
            self.received_count += 1
            self._enqueue(ReceivedDatagram(payload, address, time.time(), time.monotonic()))
            LOG.debug("Radar paketi alındı: bytes=%d source=%s", len(payload), address)

    def stop(self) -> None:
        self.stop_event.set()
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        LOG.info("UDP radar receiver durduruldu")
