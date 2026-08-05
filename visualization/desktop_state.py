"""Masaustu goruntuleyici icin bounded, thread-safe durum ve saf donusumler."""

from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RADAR_LIMIT_OPTIONS = (10, 20, 50, 100)


class DesktopState:
    def __init__(self, max_radar_measurements: int = 20, max_fused_per_track: int = 500):
        self._lock = threading.RLock()
        self._radar_limit = int(max_radar_measurements)
        self._fused_limit = int(max_fused_per_track)
        self._radar: deque[dict[str, Any]] = deque(maxlen=self._radar_limit)
        self._fused: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._fused_limit)
        )
        self._status: dict[str, Any] = {}
        self._radar_count = 0
        self._fused_count = 0
        self._last_datagram_wall_time: float | None = None
        self._last_radar_time: float | None = None
        self._last_fused_time: float | None = None

    def add_message(self, message: dict[str, Any]) -> bool:
        message_type = message.get("message_type")
        with self._lock:
            self._last_datagram_wall_time = time.time()
            if message_type == "radar_measurement":
                self._radar.append(copy.deepcopy(message))
                self._radar_count += 1
                self._last_radar_time = float(message.get("timestamp", 0.0))
                return True
            if message_type == "fused_track" and message.get("track_id"):
                self._fused[str(message["track_id"])].append(copy.deepcopy(message))
                self._fused_count += 1
                self._last_fused_time = float(message.get("publish_timestamp", 0.0))
                return True
            if message_type == "runtime_status" and isinstance(message.get("status"), dict):
                self._status = copy.deepcopy(message["status"])
                return True
        return False

    def set_radar_limit(self, limit: int) -> None:
        if limit not in RADAR_LIMIT_OPTIONS:
            raise ValueError(f"Radar limiti {RADAR_LIMIT_OPTIONS} degerlerinden biri olmali")
        with self._lock:
            self._radar_limit = limit
            self._radar = deque(list(self._radar)[-limit:], maxlen=limit)

    def set_fused_limit(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Fused nokta limiti pozitif olmali")
        with self._lock:
            self._fused_limit = int(limit)
            for track_id, points in list(self._fused.items()):
                self._fused[track_id] = deque(list(points)[-limit:], maxlen=limit)

    def clear_visual_history(self) -> None:
        with self._lock:
            self._radar.clear()
            self._fused.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "radar_measurements": copy.deepcopy(list(self._radar)),
                "fused_tracks": {
                    track_id: sorted(copy.deepcopy(list(points)), key=state_timestamp)
                    for track_id, points in self._fused.items()
                },
                "runtime_status": copy.deepcopy(self._status),
                "received_radar_count": self._radar_count,
                "received_fused_count": self._fused_count,
                "last_radar_time": self._last_radar_time,
                "last_fused_time": self._last_fused_time,
                "last_datagram_wall_time": self._last_datagram_wall_time,
            }


def state_timestamp(point: dict[str, Any]) -> float:
    return float(point.get("state_timestamp", 0.0))


def split_track_segments(
    points: list[dict[str, Any]], max_gap_s: float
) -> list[list[dict[str, Any]]]:
    ordered = sorted(points, key=state_timestamp)
    segments: list[list[dict[str, Any]]] = []
    for point in ordered:
        if not segments or state_timestamp(point) - state_timestamp(segments[-1][-1]) > max_gap_s:
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def format_radar_details(message: dict[str, Any]) -> str:
    position, velocity = message.get("position", {}), message.get("velocity", {})
    return "\n".join([
        "Tur: Radar",
        f"Measurement ID: {message.get('measurement_id', 'N/A')}",
        f"Sensor ID: {message.get('sensor_id', 'N/A')}",
        f"Source track ID: {message.get('source_track_id', 'N/A')}",
        f"Timestamp: {message.get('timestamp', 'N/A')}",
        f"Sequence number: {message.get('sequence_number', 'N/A')}",
        f"X / Y / Z: {position.get('x_m', 'N/A')} / {position.get('y_m', 'N/A')} / {position.get('z_m', 'N/A')} m",
        f"VX / VY / VZ: {velocity.get('vx_mps', 'N/A')} / {velocity.get('vy_mps', 'N/A')} / {velocity.get('vz_mps', 'N/A')} m/s",
        f"Fused track: {message.get('assigned_track_id', 'N/A')}",
        f"Ret nedeni: {message.get('rejection_reason') or 'N/A'}",
    ])


def format_fused_details(message: dict[str, Any]) -> str:
    position, velocity = message.get("position", {}), message.get("velocity", {})
    metadata = message.get("fusion_metadata", {})
    used = metadata.get("used_measurements", [])
    lines = [
        "Tur: Fused",
        f"Track ID: {message.get('track_id', 'N/A')}",
        f"Track status: {message.get('track_status', 'N/A')}",
        f"State timestamp: {message.get('state_timestamp', 'N/A')}",
        f"Publish timestamp: {message.get('publish_timestamp', 'N/A')}",
        f"X / Y / Z: {position.get('x_m', 'N/A')} / {position.get('y_m', 'N/A')} / {position.get('z_m', 'N/A')} m",
        f"VX / VY / VZ: {velocity.get('vx_mps', 'N/A')} / {velocity.get('vy_mps', 'N/A')} / {velocity.get('vz_mps', 'N/A')} m/s",
        f"Algoritma: {metadata.get('filter_name', 'N/A')}",
        f"Update type: {metadata.get('update_type', 'N/A')}",
        f"Dominant model: {metadata.get('dominant_model', 'N/A')}",
        f"Kullanilan radar sayisi: {len(used)}",
    ]
    state_time = float(message.get("state_timestamp", 0.0))
    for index, item in enumerate(used, 1):
        radar_time = float(item.get("measurement_timestamp", 0.0))
        lines.extend([
            "",
            f"Radar {index}: {item.get('measurement_id', 'N/A')}",
            f"Sensor: {item.get('sensor_id', 'N/A')}",
            f"Radar time: {radar_time}",
            f"Sequence: {item.get('sequence_number', 'N/A')}",
            f"Delta t: {state_time - radar_time:.3f} s",
        ])
    return "\n".join(lines)


@dataclass(frozen=True)
class GroundTruthData:
    routes: dict[str, list[dict[str, Any]]]
    warning: str | None = None


def load_ground_truth(path: str | Path) -> GroundTruthData:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        return GroundTruthData({}, f"Ground truth yuklenemedi: {exc}")
    required = {"time", "x", "y"}
    if frame.empty or not required.issubset(frame.columns):
        return GroundTruthData({}, "Ground truth bos veya time/x/y kolonlari eksik")
    group_key = next((key for key in ("callsign", "target_id", "target_name") if key in frame), None)
    if group_key is None:
        frame = frame.assign(_route="GROUND_TRUTH")
        group_key = "_route"
    routes = {
        str(name): group.sort_values("time", kind="stable").to_dict("records")
        for name, group in frame.groupby(group_key, sort=False)
    }
    return GroundTruthData(routes)
