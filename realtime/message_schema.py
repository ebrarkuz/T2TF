"""UDP radar/fused JSON şemaları ve katı doğrulama yardımcıları."""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = 1
SUPPORTED_FRAMES = {"ENU"}


class MessageValidationError(ValueError):
    """Paketin neden reddedildiğini taşıyan beklenen doğrulama hatası."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise MessageValidationError(f"{field} sayısal olmalı")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MessageValidationError(f"{field} sayısal olmalı") from exc
    if not math.isfinite(number):
        raise MessageValidationError(f"{field} sonlu olmalı")
    return number


def decode_json_packet(payload: bytes, max_bytes: int = 65507) -> dict[str, Any]:
    if len(payload) > max_bytes:
        raise MessageValidationError("packet_too_large")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageValidationError("invalid_json") from exc
    if not isinstance(decoded, dict):
        raise MessageValidationError("JSON kökü object olmalı")
    return decoded


def validate_radar_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("schema_version") != SCHEMA_VERSION:
        raise MessageValidationError("unsupported_schema_version")
    if message.get("message_type") != "radar_measurement":
        raise MessageValidationError("invalid_message_type")
    measurement_id = str(message.get("measurement_id", "")).strip()
    sensor_id = str(message.get("sensor_id", "")).strip()
    if not measurement_id:
        raise MessageValidationError("missing_measurement_id")
    if not sensor_id:
        raise MessageValidationError("missing_sensor_id")
    timestamp = _finite_number(message.get("timestamp"), "timestamp")
    sequence_number = message.get("sequence_number")
    if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
        raise MessageValidationError("sequence_number integer olmalı")
    frame = str(message.get("coordinate_frame", "")).upper()
    if frame not in SUPPORTED_FRAMES:
        raise MessageValidationError("unsupported_coordinate_frame")

    position = message.get("position")
    velocity = message.get("velocity")
    if not isinstance(position, dict) or not isinstance(velocity, dict):
        raise MessageValidationError("Cartesian position ve velocity gerekli")
    normalized = dict(message)
    normalized["measurement_id"] = measurement_id
    normalized["sensor_id"] = sensor_id
    normalized["timestamp"] = timestamp
    normalized["coordinate_frame"] = frame
    normalized["position"] = {
        key: _finite_number(position.get(key), f"position.{key}")
        for key in ("x_m", "y_m", "z_m")
    }
    normalized["velocity"] = {
        key: _finite_number(velocity.get(key), f"velocity.{key}")
        for key in ("vx_mps", "vy_mps", "vz_mps")
    }
    covariance = message.get("covariance")
    if covariance is not None:
        if not isinstance(covariance, list) or len(covariance) != 6:
            raise MessageValidationError("covariance 6x6 liste veya null olmalı")
        normalized_covariance = []
        for i, row in enumerate(covariance):
            if not isinstance(row, list) or len(row) != 6:
                raise MessageValidationError("covariance 6x6 olmalı")
            normalized_covariance.append([
                _finite_number(value, f"covariance[{i}][{j}]")
                for j, value in enumerate(row)
            ])
        normalized["covariance"] = normalized_covariance
    return normalized


def validate_fused_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("schema_version") != SCHEMA_VERSION:
        raise MessageValidationError("unsupported_schema_version")
    if message.get("message_type") != "fused_track":
        raise MessageValidationError("invalid_message_type")
    if not str(message.get("track_id", "")).strip():
        raise MessageValidationError("missing_track_id")
    _finite_number(message.get("publish_timestamp"), "publish_timestamp")
    _finite_number(message.get("state_timestamp"), "state_timestamp")
    for section, fields in (
        ("position", ("x_m", "y_m", "z_m")),
        ("velocity", ("vx_mps", "vy_mps", "vz_mps")),
    ):
        data = message.get(section)
        if not isinstance(data, dict):
            raise MessageValidationError(f"missing_{section}")
        for field in fields:
            _finite_number(data.get(field), f"{section}.{field}")
    metadata = message.get("fusion_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("used_measurements"), list):
        raise MessageValidationError("invalid_fusion_metadata")
    return message


@dataclass
class ValidationResult:
    accepted: bool
    message: dict[str, Any] | None = None
    reason: str | None = None


class RadarMessageValidator:
    """Duplicate, yaş ve sıralama politikasını şema doğrulamasına ekler."""

    def __init__(
        self,
        *,
        duplicate_cache_size: int = 10000,
        max_packet_age_s: float = 2.0,
        max_out_of_order_s: float = 0.25,
        drop_late_packets: bool = True,
        max_packet_bytes: int = 65507,
        clock=time.time,
    ):
        self.duplicate_cache_size = duplicate_cache_size
        self.max_packet_age_s = max_packet_age_s
        self.max_out_of_order_s = max_out_of_order_s
        self.drop_late_packets = drop_late_packets
        self.max_packet_bytes = max_packet_bytes
        self.clock = clock
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._last_sequence: dict[str, int] = {}
        self._last_timestamp: dict[str, float] = {}

    def validate_packet(self, payload: bytes) -> ValidationResult:
        try:
            message = validate_radar_message(
                decode_json_packet(payload, self.max_packet_bytes)
            )
            measurement_id = message["measurement_id"]
            if measurement_id in self._seen:
                raise MessageValidationError("duplicate_measurement")
            age = self.clock() - message["timestamp"]
            if self.drop_late_packets and age > self.max_packet_age_s:
                raise MessageValidationError("packet_too_old")
            sensor = message["sensor_id"]
            last_time = self._last_timestamp.get(sensor)
            if (
                self.drop_late_packets and last_time is not None
                and message["timestamp"] < last_time - self.max_out_of_order_s
            ):
                raise MessageValidationError("out_of_order_timestamp")
            last_sequence = self._last_sequence.get(sensor)
            if last_sequence is not None and message["sequence_number"] < last_sequence:
                raise MessageValidationError("sequence_number_regressed")
        except MessageValidationError as exc:
            return ValidationResult(False, reason=str(exc))

        self._seen[measurement_id] = None
        self._seen.move_to_end(measurement_id)
        while len(self._seen) > self.duplicate_cache_size:
            self._seen.popitem(last=False)
        self._last_sequence[sensor] = message["sequence_number"]
        self._last_timestamp[sensor] = max(
            message["timestamp"], self._last_timestamp.get(sensor, -math.inf)
        )
        return ValidationResult(True, message=message)


def encode_message(message: dict[str, Any]) -> bytes:
    return json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
