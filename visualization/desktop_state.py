"""Masaustu goruntuleyici icin kalici, bounded ve thread-safe gorsel state."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


RADAR_LIMIT_OPTIONS = (10, 20, 50, 100)


@dataclass
class TrackVisualState:
    track_id: str
    points: deque[dict[str, Any]] = field(default_factory=deque)
    status: str = "active"
    last_update_timestamp: float | None = None
    last_received_wall_time: float | None = None
    sensor_ids: set[str] = field(default_factory=set)
    filter_status: str = "tentative"
    ever_confirmed: bool = False
    first_seen: float | None = None
    last_seen: float | None = None
    delete_reason: str | None = None


class DesktopState:
    """Fuzyon state'inden bagimsiz radar tamponu ve track rota arsivi."""

    def __init__(
        self,
        max_radar_measurements_per_sensor: int = 20,
        max_fused_per_track: int = 2000,
        track_archive_timeout_s: float = 5.0,
        preserve_archived_tracks: bool = True,
        full_visual_history: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        self._lock = threading.RLock()
        self._radar_limit = int(max_radar_measurements_per_sensor)
        self._fused_limit = int(max_fused_per_track)
        self._archive_timeout_s = float(track_archive_timeout_s)
        self._preserve_archived = bool(preserve_archived_tracks)
        self._full_visual_history = bool(full_visual_history)
        self._clock = clock
        self._radar_by_sensor: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._radar_limit)
        )
        self.active_tracks: dict[str, TrackVisualState] = {}
        self.archived_tracks: dict[str, TrackVisualState] = {}
        self._status: dict[str, Any] = {}
        self._radar_count = 0
        self._fused_count = 0
        self._seen_track_ids: set[str] = set()
        self._last_datagram_wall_time: float | None = None
        self._last_radar_time: float | None = None
        self._last_fused_time: float | None = None

    def add_message(self, message: dict[str, Any]) -> bool:
        message_type = message.get("message_type")
        now = self._clock()
        with self._lock:
            self._last_datagram_wall_time = now
            if message_type == "radar_measurement":
                sensor_id = str(message.get("sensor_id") or "UNKNOWN_SENSOR")
                self._radar_by_sensor[sensor_id].append(copy.deepcopy(message))
                self._radar_count += 1
                self._last_radar_time = float(message.get("timestamp", 0.0))
                return True
            if message_type == "fused_track" and message.get("track_id"):
                self._add_fused(message, now)
                return True
            if message_type == "runtime_status" and isinstance(message.get("status"), dict):
                self._status = copy.deepcopy(message["status"])
                return True
        return False

    def _add_fused(self, message: dict[str, Any], now: float) -> None:
        track_id = str(message["track_id"])
        track = self.active_tracks.get(track_id)
        if track is None:
            track = self.archived_tracks.pop(track_id, None)
        if track is None:
            track = TrackVisualState(track_id=track_id)
        track.status = "active"
        point = copy.deepcopy(message)
        track.filter_status = str(message.get("track_status", "tentative")).lower()
        track.ever_confirmed = track.ever_confirmed or track.filter_status == "confirmed"
        track.points.append(point)
        track.last_update_timestamp = state_timestamp(point)
        track.first_seen = track.first_seen if track.first_seen is not None else track.last_update_timestamp
        track.last_seen = track.last_update_timestamp
        track.delete_reason = None
        track.last_received_wall_time = now
        track.sensor_ids.update(
            str(item.get("sensor_id"))
            for item in point.get("fusion_metadata", {}).get("used_measurements", [])
            if item.get("sensor_id")
        )
        self._compact_track(track)
        self.active_tracks[track_id] = track
        self._seen_track_ids.add(track_id)
        self._fused_count += 1
        self._last_fused_time = float(message.get("publish_timestamp", 0.0))

    def archive_inactive(self, now: float | None = None) -> list[str]:
        now = self._clock() if now is None else float(now)
        archived = []
        with self._lock:
            for track_id, track in list(self.active_tracks.items()):
                last_seen = track.last_received_wall_time
                if last_seen is None or now - last_seen <= self._archive_timeout_s:
                    continue
                self.active_tracks.pop(track_id)
                track.status = "archived"
                track.delete_reason = "no_fused_update_timeout"
                if self._preserve_archived:
                    self.archived_tracks[track_id] = track
                archived.append(track_id)
        return archived

    def _compact_track(self, track: TrackVisualState) -> None:
        if self._full_visual_history or len(track.points) <= self._fused_limit:
            return
        points = list(track.points)
        recent_count = max(2, self._fused_limit // 2)
        recent = points[-recent_count:]
        older = points[:-recent_count]
        remaining = max(1, self._fused_limit - len(recent))
        if len(older) <= remaining:
            sampled = older
        else:
            important_indices = {0}
            previous_status = None
            for index, point in enumerate(older):
                metadata = point.get("fusion_metadata", {})
                status = point.get("track_status")
                if metadata.get("used_measurements") or (
                    previous_status is not None and status != previous_status
                ):
                    important_indices.add(index)
                previous_status = status
            candidates = sorted(important_indices)
            if len(candidates) > remaining:
                selected = {
                    candidates[round(index * (len(candidates) - 1) / max(1, remaining - 1))]
                    for index in range(remaining)
                }
            else:
                selected = set(candidates)
                ordinary = [index for index in range(len(older)) if index not in selected]
                slots = remaining - len(selected)
                if ordinary and slots:
                    selected.update(
                        ordinary[round(index * (len(ordinary) - 1) / max(1, slots - 1))]
                        for index in range(slots)
                    )
            sampled = [older[index] for index in sorted(selected)]
        compacted = (sampled + recent)[-self._fused_limit:]
        if points and compacted and compacted[0] is not points[0]:
            compacted[0] = points[0]
        track.points = deque(compacted)

    def set_radar_limit(self, limit: int) -> None:
        if limit not in RADAR_LIMIT_OPTIONS:
            raise ValueError(f"Radar limiti {RADAR_LIMIT_OPTIONS} degerlerinden biri olmali")
        with self._lock:
            self._radar_limit = limit
            for sensor_id, points in list(self._radar_by_sensor.items()):
                self._radar_by_sensor[sensor_id] = deque(list(points)[-limit:], maxlen=limit)

    def set_fused_limit(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Fused nokta limiti pozitif olmali")
        with self._lock:
            self._fused_limit = int(limit)
            for track in (*self.active_tracks.values(), *self.archived_tracks.values()):
                self._compact_track(track)

    def clear_visual_history(self) -> None:
        with self._lock:
            self._radar_by_sensor.clear()
            self.active_tracks.clear()
            self.archived_tracks.clear()
            self._seen_track_ids.clear()

    @staticmethod
    def _track_snapshot(track: TrackVisualState) -> dict[str, Any]:
        return {
            "track_id": track.track_id,
            "status": track.status,
            "points": sorted(copy.deepcopy(list(track.points)), key=state_timestamp),
            "last_update_timestamp": track.last_update_timestamp,
            "sensor_ids": sorted(track.sensor_ids),
            "filter_status": track.filter_status,
            "visual_state": track.status,
            "point_count": len(track.points),
            "ever_confirmed": track.ever_confirmed,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "delete_reason": track.delete_reason,
        }

    def snapshot(self) -> dict[str, Any]:
        self.archive_inactive()
        with self._lock:
            radar_by_sensor = {
                sensor: copy.deepcopy(list(points))
                for sensor, points in self._radar_by_sensor.items()
            }
            active = {
                track_id: self._track_snapshot(track)
                for track_id, track in self.active_tracks.items()
            }
            archived = {
                track_id: self._track_snapshot(track)
                for track_id, track in self.archived_tracks.items()
            }
            return {
                "radar_by_sensor": radar_by_sensor,
                "radar_measurements": [
                    item for points in radar_by_sensor.values() for item in points
                ],
                "active_tracks": active,
                "archived_tracks": archived,
                "fused_tracks": {
                    track_id: data["points"] for track_id, data in {**archived, **active}.items()
                },
                "runtime_status": copy.deepcopy(self._status),
                "received_radar_count": self._radar_count,
                "received_fused_count": self._fused_count,
                "active_track_count": len(active),
                "archived_track_count": len(archived),
                "total_seen_track_count": len(self._seen_track_ids),
                "last_radar_time": self._last_radar_time,
                "last_fused_time": self._last_fused_time,
                "last_datagram_wall_time": self._last_datagram_wall_time,
            }


def state_timestamp(point: dict[str, Any]) -> float:
    return float(point.get("state_timestamp", 0.0))


def sample_track_points(
    points: list[dict[str, Any]], interval_s: float = 1.0
) -> list[dict[str, Any]]:
    """Her zaman araligi icin son fused noktayi sec; rota cizgisi uretme."""
    if interval_s <= 0:
        return sorted(points, key=state_timestamp)
    buckets: dict[int, dict[str, Any]] = {}
    for point in sorted(points, key=state_timestamp):
        bucket = math.floor(state_timestamp(point) / interval_s)
        buckets[bucket] = point
    return [buckets[key] for key in sorted(buckets)]


def select_visual_tracks(
    active_tracks: dict[str, dict[str, Any]],
    archived_tracks: dict[str, dict[str, Any]],
    mode: str = "Normal gorunum",
    minimum_visible_tentative_points: int = 2,
) -> dict[str, dict[str, Any]]:
    tracks = {**archived_tracks, **active_tracks}
    if mode == "Confirmed only":
        return {key: value for key, value in tracks.items() if value["ever_confirmed"]}
    if mode == "Active hypotheses":
        return {key: value for key, value in tracks.items() if value["visual_state"] == "active"}
    if mode == "Tum trackler / debug":
        return tracks
    return {
        key: value for key, value in tracks.items()
        if value["ever_confirmed"]
        or value["visual_state"] == "active"
        or (
            value["filter_status"] == "tentative"
            and value["point_count"] >= minimum_visible_tentative_points
        )
    }


def format_radar_details(message: dict[str, Any]) -> str:
    position, velocity = message.get("position", {}), message.get("velocity", {})
    return "\n".join([
        "Tur: Radar",
        f"Sensor ID: {message.get('sensor_id', 'N/A')}",
        f"Measurement ID: {message.get('measurement_id', 'N/A')}",
        f"Source Track ID: {message.get('source_track_id', 'N/A')}",
        f"Timestamp: {message.get('timestamp', 'N/A')}",
        f"Sequence Number: {message.get('sequence_number', 'N/A')}",
        f"X / Y / Z: {position.get('x_m', 'N/A')} / {position.get('y_m', 'N/A')} / {position.get('z_m', 'N/A')} m",
        f"VX / VY / VZ: {velocity.get('vx_mps', 'N/A')} / {velocity.get('vy_mps', 'N/A')} / {velocity.get('vz_mps', 'N/A')} m/s",
        f"Global Track: {message.get('assigned_track_id', 'N/A')}",
        f"Ret nedeni: {message.get('rejection_reason') or 'N/A'}",
    ])


def format_fused_details(message: dict[str, Any], visual_status: str = "active") -> str:
    position, velocity = message.get("position", {}), message.get("velocity", {})
    metadata = message.get("fusion_metadata", {})
    used = metadata.get("used_measurements", [])
    lines = [
        "Tur: Fused",
        f"Global Track ID: {message.get('track_id', 'N/A')}",
        f"Track status: {message.get('track_status', 'N/A')}",
        f"Visual state: {visual_status.capitalize()}",
        f"State timestamp: {message.get('state_timestamp', 'N/A')}",
        f"Publish timestamp: {message.get('publish_timestamp', 'N/A')}",
        f"X / Y / Z: {position.get('x_m', 'N/A')} / {position.get('y_m', 'N/A')} / {position.get('z_m', 'N/A')} m",
        f"VX / VY / VZ: {velocity.get('vx_mps', 'N/A')} / {velocity.get('vy_mps', 'N/A')} / {velocity.get('vz_mps', 'N/A')} m/s",
        f"Algoritma: {metadata.get('filter_name', 'N/A')}",
        f"Dominant model: {metadata.get('dominant_model', 'N/A')}",
        f"Update type: {metadata.get('update_type', 'N/A')}",
        f"Kullanilan radar sayisi: {len(used)}",
    ]
    state_time = float(message.get("state_timestamp", 0.0))
    for index, item in enumerate(used, 1):
        radar_time = float(item.get("measurement_timestamp", 0.0))
        lines.extend([
            "", f"Radar {index}",
            f"Sensor ID: {item.get('sensor_id', 'N/A')}",
            f"Measurement ID: {item.get('measurement_id', 'N/A')}",
            f"Radar timestamp: {radar_time}",
            f"Sequence number: {item.get('sequence_number', 'N/A')}",
            f"Delta t: {state_time - radar_time:.3f} s",
        ])
    return "\n".join(lines)


def format_ground_truth_details(name: str, point: dict[str, Any]) -> str:
    return "\n".join([
        "Tur: Ground Truth", f"Hedef: {name}", f"Time: {point.get('time', 'N/A')}",
        f"X / Y / Z: {point.get('x', 'N/A')} / {point.get('y', 'N/A')} / {point.get('z', 'N/A')} m",
        f"VX / VY / VZ: {point.get('vx', 'N/A')} / {point.get('vy', 'N/A')} / {point.get('vz', 'N/A')} m/s",
    ])


@dataclass(frozen=True)
class GroundTruthData:
    routes: dict[str, list[dict[str, Any]]]
    warning: str | None = None


def load_ground_truth(path: str | Path) -> GroundTruthData:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        return GroundTruthData({}, f"Ground truth yuklenemedi: {exc}")
    if frame.empty or not {"time", "x", "y"}.issubset(frame.columns):
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
