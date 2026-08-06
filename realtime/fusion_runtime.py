"""Mevcut dual-IMM merkezini UDP akışına bağlayan gerçek zamanlı servis."""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from fusion import create_fusion_algorithm
from fusion.measurement import TQ_MAX, TQ_MIN, measurement_covariance
from .message_schema import RadarMessageValidator
from .micro_batch import BufferedMeasurement, MeasurementMicroBatcher
from .runtime_config import DEFAULT_CONFIG_PATH, RuntimeConfig
from .udp_fused_publisher import UdpFusedPublisher
from .udp_json_publisher import UdpJsonPublisher
from .udp_radar_receiver import ReceivedDatagram, UdpRadarReceiver


LOG = logging.getLogger(__name__)


@dataclass
class RuntimeCounters:
    total_packets: int = 0
    valid_packets: int = 0
    rejected_packets: int = 0
    queue_dropped_packets: int = 0
    published_fused_messages: int = 0
    last_radar_packet_time: float | None = None
    last_fused_publish_time: float | None = None
    max_queue_depth: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    processed_micro_batches: int = 0
    micro_batch_measurements: int = 0
    micro_batch_max_size: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)


class FusionRuntime:
    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig.load()
        self.config.validate()
        self.input_queue: queue.Queue[ReceivedDatagram] = queue.Queue(
            maxsize=self.config.radar_queue_maxsize
        )
        self.receiver = UdpRadarReceiver(
            self.config.radar_udp_host,
            self.config.radar_udp_port,
            self.input_queue,
            max_packet_bytes=self.config.max_udp_packet_bytes,
            overflow_policy=self.config.queue_overflow_policy,
        )
        self.publisher = UdpFusedPublisher(
            self.config.fused_udp_host,
            self.config.fused_udp_port,
            self.config.fused_queue_maxsize,
            self.config.queue_overflow_policy,
        )
        self.telemetry_publisher = UdpJsonPublisher(
            self.config.telemetry_udp_host,
            self.config.telemetry_udp_port,
            self.config.fused_queue_maxsize,
            self.config.queue_overflow_policy,
            name="visualization-telemetry",
        )
        self.validator = RadarMessageValidator(
            duplicate_cache_size=self.config.duplicate_cache_size,
            max_packet_age_s=self.config.max_packet_age_s,
            max_out_of_order_s=self.config.max_out_of_order_s,
            drop_late_packets=self.config.drop_late_packets,
            max_packet_bytes=self.config.max_udp_packet_bytes,
        )
        self.fusion = create_fusion_algorithm(
            self.config.fusion_algorithm, self.config.fusion_parameters
        )
        self.micro_batcher = MeasurementMicroBatcher(
            window_ms=self.config.micro_batch_window_ms,
            timestamp_tolerance_ms=self.config.timestamp_group_tolerance_ms,
            max_batch_size=self.config.max_micro_batch_size,
            max_wait_ms=self.config.max_micro_batch_wait_ms,
        )
        self.counters = RuntimeCounters()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.started_at: float | None = None
        self._last_status_publish = 0.0
        self._counter_lock = threading.RLock()

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.fusion.reset()
        self.started_at = time.time()
        self.stop_event.clear()
        self.receiver.start()
        self.publisher.start()
        self.telemetry_publisher.start()
        self.worker = threading.Thread(target=self._worker_loop, name="fusion-worker", daemon=True)
        self.worker.start()
        LOG.info(
            "Gerçek zamanlı %s servisi başladı; radar=%s:%s fused=%s:%s",
            self.config.fusion_algorithm,
            self.config.radar_udp_host,
            self.receiver.bound_port,
            self.config.fused_udp_host,
            self.config.fused_udp_port,
        )

    def _worker_loop(self) -> None:
        poll_timeout = min(
            0.2,
            self.config.micro_batch_window_ms / 1000.0,
            self.config.max_micro_batch_wait_ms / 1000.0,
        ) if self.config.micro_batch_enabled else 0.2
        while not self.stop_event.is_set() or not self.input_queue.empty():
            try:
                datagram = self.input_queue.get(timeout=max(0.005, poll_timeout))
            except queue.Empty:
                self._flush_ready_micro_batches(time.monotonic())
                self._publish_runtime_status_if_due()
                continue
            try:
                self.process_datagram(
                    datagram.payload,
                    datagram.received_at,
                    received_at_monotonic=datagram.received_at_monotonic,
                )
            except Exception:
                LOG.exception("Radar paketi işlenirken beklenmeyen hata")
            finally:
                self.input_queue.task_done()
        self._flush_ready_micro_batches(force=True)

    def _measurement_from_message(self, message: dict[str, Any]) -> dict[str, Any]:
        position, velocity = message["position"], message["velocity"]
        state = np.array([
            position["x_m"], velocity["vx_mps"],
            position["y_m"], velocity["vy_mps"],
            position["z_m"], velocity["vz_mps"],
        ], dtype=float).reshape(6, 1)
        if message.get("covariance") is not None:
            covariance = np.asarray(message["covariance"], dtype=float)
        else:
            quality = message.get("quality") or {}
            detection_probability = quality.get("detection_probability")
            default_tq = 8.0 if detection_probability is None else 1.0 + 14.0 * float(detection_probability)
            tq = float(np.clip(message.get("track_quality", default_tq), TQ_MIN, TQ_MAX))
            covariance = measurement_covariance({
                "track_quality": tq,
                "sigma_pos_m": message.get("sigma_pos_m", np.nan),
                "sigma_vel_mps": message.get("sigma_vel_mps", np.nan),
            })
        tq = float(np.clip(message.get("track_quality", 8.0), TQ_MIN, TQ_MAX))
        source_track_id = (
            message.get("source_track_id")
            or message["measurement_id"]
        )
        return {
            "timestamp": float(message["timestamp"]),
            "state": state,
            "cov": covariance,
            "tq": tq,
            "src": (message["sensor_id"], str(source_track_id)),
            "measurement_id": message["measurement_id"],
            "sequence_number": message["sequence_number"],
            "measurement_timestamp": message["timestamp"],
        }

    def _fused_message(
        self,
        track: dict[str, Any],
        state_timestamp: float,
        used_measurements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        publish_timestamp = time.time()
        state = track["state"]
        return {
            "schema_version": 1,
            "message_type": "fused_track",
            "publish_timestamp": publish_timestamp,
            "state_timestamp": float(state_timestamp),
            "track_id": track["track_id"],
            "track_status": track["track_status"],
            "coordinate_frame": "ENU",
            "position": {
                "x_m": state["x_m"], "y_m": state["y_m"], "z_m": state["z_m"],
            },
            "velocity": {
                "vx_mps": state["vx_mps"], "vy_mps": state["vy_mps"], "vz_mps": state["vz_mps"],
            },
            "covariance": {
                "position_variance": track["position_variance"],
                "velocity_variance": track["velocity_variance"],
            },
            "fusion_metadata": {
                "update_type": "measurement_update" if used_measurements else "prediction_only",
                "filter_name": track["filter_name"],
                "dominant_model": track["dominant_model"],
                "model_probabilities": track["model_probabilities"],
                "used_measurements": used_measurements,
            },
        }

    def process_datagram(
        self,
        payload: bytes,
        received_at: float | None = None,
        *,
        received_at_monotonic: float | None = None,
        publish: bool = True,
    ) -> list[dict[str, Any]]:
        received_at = time.time() if received_at is None else float(received_at)
        received_at_monotonic = (
            time.monotonic() if received_at_monotonic is None
            else float(received_at_monotonic)
        )
        started = time.perf_counter()
        with self._counter_lock:
            self.counters.total_packets += 1
            self.counters.last_radar_packet_time = received_at
        result = self.validator.validate_packet(payload)
        if not result.accepted:
            with self._counter_lock:
                self.counters.rejected_packets += 1
                reasons = dict(self.counters.rejected_reasons)
                reasons[str(result.reason)] = reasons.get(str(result.reason), 0) + 1
                self.counters.rejected_reasons = reasons
            LOG.warning("Radar paketi reddedildi: %s", result.reason)
            self._publish_runtime_status_if_due()
            return []

        message = result.message
        with self._counter_lock:
            self.counters.valid_packets += 1
        measurement = self._measurement_from_message(message)
        entry = BufferedMeasurement(
            message=message,
            measurement=measurement,
            received_at=received_at,
            received_at_monotonic=received_at_monotonic,
        )
        if not self.config.micro_batch_enabled:
            outputs = self._process_measurement_batch([entry], publish=publish)
        else:
            self.micro_batcher.add(entry)
            outputs = []
            for batch in self.micro_batcher.pop_ready_batches(received_at_monotonic):
                outputs.extend(self._process_measurement_batch(batch, publish=publish))

        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._counter_lock:
            self.counters.latency_sum_ms += latency_ms
            self.counters.latency_max_ms = max(self.counters.latency_max_ms, latency_ms)
            self.counters.max_queue_depth = max(
                self.counters.max_queue_depth, self.input_queue.qsize()
            )
        self._publish_runtime_status_if_due()
        return outputs

    def _process_measurement_batch(
        self,
        entries: list[BufferedMeasurement],
        *,
        publish: bool = True,
    ) -> list[dict[str, Any]]:
        if not entries:
            return []
        entries = sorted(entries, key=lambda item: item.measurement["timestamp"])
        measurements = [item.measurement for item in entries]
        batch_time = max(item["timestamp"] for item in measurements)
        span_ms = (batch_time - min(item["timestamp"] for item in measurements)) * 1000.0
        sensor_ids = sorted({str(item.message["sensor_id"]) for item in entries})
        LOG.info(
            "Micro-batch: size=%d start=%.6f end=%.6f span_ms=%.3f sensors=%s",
            len(entries), min(item["timestamp"] for item in measurements), batch_time,
            span_ms, ",".join(sensor_ids),
        )
        with self._counter_lock:
            self.counters.processed_micro_batches += 1
            self.counters.micro_batch_measurements += len(entries)
            self.counters.micro_batch_max_size = max(
                self.counters.micro_batch_max_size, len(entries)
            )

        previous_tracks = {
            str(track.id): str(getattr(track, "status", getattr(track, "state_status", "")))
            for track in self.fusion.tracks
        }
        track_snapshots = self.fusion.process_batch(measurements)
        current_tracks = {t["track_id"]: t["track_status"].upper() for t in track_snapshots}
        for track_id in current_tracks.keys() - previous_tracks.keys():
            LOG.info("Track oluşturuldu: %s", track_id)
        for track_id, status in current_tracks.items():
            if status == "CONFIRMED" and previous_tracks.get(track_id) != "CONFIRMED":
                LOG.info("Track confirmed oldu: %s", track_id)
        for track_id in previous_tracks.keys() - current_tracks.keys():
            LOG.info("Track silindi: %s", track_id)
        assignments = dict(getattr(self.fusion, "last_batch_assignments", {}))
        used_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            message = entry.message
            assignment = assignments.get(str(message["measurement_id"]), {})
            assigned_track_id = assignment.get("track_id")
            accepted_stage = assignment.get("stage", "unassigned")
            radar_record = dict(message)
            radar_record.update({
                "received_at": entry.received_at,
                "used_in_fusion": assigned_track_id is not None,
                "assigned_track_id": assigned_track_id,
                "association_stage": accepted_stage,
                "rejection_reason": None if assigned_track_id is not None else "association_rejected",
            })
            if publish:
                self.telemetry_publisher.publish(radar_record)
            if assigned_track_id is not None:
                used_by_track[str(assigned_track_id)].append({
                    "measurement_id": message["measurement_id"],
                    "sensor_id": message["sensor_id"],
                    "measurement_timestamp": message["timestamp"],
                    "sequence_number": message["sequence_number"],
                    "received_at": entry.received_at,
                })

        fused_messages = []
        for track in track_snapshots:
            used = used_by_track.get(track["track_id"], [])
            fused_message = self._fused_message(track, batch_time, used)
            fused_messages.append(fused_message)
            if publish:
                if self.publisher.publish(fused_message):
                    with self._counter_lock:
                        self.counters.published_fused_messages += 1
                        self.counters.last_fused_publish_time = fused_message["publish_timestamp"]
                self.telemetry_publisher.publish(fused_message)

        return fused_messages

    def _flush_ready_micro_batches(
        self,
        current_monotonic_time: float | None = None,
        *,
        force: bool = False,
        publish: bool = True,
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if current_monotonic_time is None else float(current_monotonic_time)
        outputs = []
        for batch in self.micro_batcher.pop_ready_batches(now, force=force):
            outputs.extend(self._process_measurement_batch(batch, publish=publish))
        return outputs

    def _diagnostic_snapshot(self) -> dict[str, Any]:
        with self._counter_lock:
            counters = asdict(self.counters)
        elapsed = max(time.time() - self.started_at, 1e-9) if self.started_at else 0.0
        counters.update({
            "service_running": not self.stop_event.is_set(),
            "radar_udp_host": self.config.radar_udp_host,
            "radar_udp_port": self.receiver.bound_port,
            "fused_udp_host": self.config.fused_udp_host,
            "fused_udp_port": self.config.fused_udp_port,
            "packets_per_second": counters["total_packets"] / elapsed if elapsed else 0.0,
            "active_track_count": len(self.fusion.tracks),
            "confirmed_track_count": sum(
                getattr(t, "status", getattr(t, "state_status", "")) == "CONFIRMED"
                for t in self.fusion.tracks
            ),
            "fusion_algorithm": self.config.fusion_algorithm,
            "micro_batch_enabled": self.config.micro_batch_enabled,
            "micro_batch_pending_measurements": len(self.micro_batcher),
            "micro_batch_window_ms": self.config.micro_batch_window_ms,
            "timestamp_group_tolerance_ms": self.config.timestamp_group_tolerance_ms,
            "track_deletion_requires_measurement_tick": True,
            "radar_queue_depth": self.input_queue.qsize(),
            "radar_queue_capacity": self.config.radar_queue_maxsize,
            "receiver_overflow_count": self.receiver.overflow_count,
            "total_received_packets": self.receiver.received_count,
            "fused_queue_depth": self.publisher.queue.qsize(),
            "fused_queue_capacity": self.config.fused_queue_maxsize,
            "fused_queue_overflow_count": self.publisher.overflow_count,
            "telemetry_udp_host": self.config.telemetry_udp_host,
            "telemetry_udp_port": self.config.telemetry_udp_port,
            "telemetry_queue_depth": self.telemetry_publisher.queue.qsize(),
            "telemetry_queue_overflow_count": self.telemetry_publisher.overflow_count,
            "average_fusion_latency_ms": (
                counters["latency_sum_ms"] / counters["valid_packets"]
                if counters["valid_packets"] else 0.0
            ),
        })
        return counters

    def _publish_runtime_status_if_due(self, force: bool = False) -> None:
        now = time.monotonic()
        interval = self.config.visualization_refresh_ms / 1000.0
        if force or now - self._last_status_publish >= interval:
            self.telemetry_publisher.publish({
                "schema_version": 1,
                "message_type": "runtime_status",
                "publish_timestamp": time.time(),
                "status": self._diagnostic_snapshot(),
            })
            self._last_status_publish = now

    def stop(self) -> None:
        self.stop_event.set()
        self.receiver.stop()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3.0)
        self._flush_ready_micro_batches(force=True)
        self.counters.queue_dropped_packets = self.receiver.overflow_count
        self._publish_runtime_status_if_due(force=True)
        self.publisher.close()
        self.telemetry_publisher.close()
        LOG.info("Füzyon servisi durduruldu: %s", self._diagnostic_snapshot())


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerçek zamanlı seçilebilir UDP füzyon servisi")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = FusionRuntime(RuntimeConfig.load(args.config))
    shutdown = threading.Event()

    def request_shutdown(*_args):
        shutdown.set()

    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signal_name, request_shutdown)
        except (ValueError, OSError):
            pass
    service.start()
    try:
        while not shutdown.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


if __name__ == "__main__":
    main()
