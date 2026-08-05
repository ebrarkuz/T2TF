"""Bounded kuyruk kullanan genel amacli asenkron JSON/UDP yayimlayici."""

from __future__ import annotations

import logging
import queue
import socket
import threading
from typing import Any

from .message_schema import encode_message


LOG = logging.getLogger(__name__)


class UdpJsonPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        max_queue_size: int = 1000,
        overflow_policy: str = "drop_oldest",
        name: str = "json",
    ):
        self.host = host
        self.port = port
        self.name = name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self.overflow_policy = overflow_policy
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.published_count = 0
        self.overflow_count = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name=f"udp-{self.name}-publisher", daemon=True
        )
        self.thread.start()

    def publish(self, message: dict[str, Any]) -> bool:
        try:
            self.queue.put_nowait(message)
            return True
        except queue.Full:
            self.overflow_count += 1
            LOG.warning("%s publish queue overflow; policy=%s", self.name, self.overflow_policy)
        if self.overflow_policy == "drop_oldest":
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(message)
                return True
            except queue.Full:
                pass
        return False

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                message = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.socket.sendto(encode_message(message), (self.host, self.port))
                self.published_count += 1
            except OSError:
                if not self.stop_event.is_set():
                    LOG.exception("UDP %s publish hatasi", self.name)
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        try:
            self.socket.close()
        except OSError:
            pass
