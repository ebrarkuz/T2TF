"""Timestamp tabanli, bounded radar micro-batch tamponu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BufferedMeasurement:
    message: dict[str, Any]
    measurement: dict[str, Any]
    received_at: float
    received_at_monotonic: float


@dataclass
class MeasurementGroup:
    entries: list[BufferedMeasurement] = field(default_factory=list)
    first_received_monotonic: float = 0.0

    @property
    def start_timestamp(self) -> float:
        return min(item.measurement["timestamp"] for item in self.entries)

    @property
    def end_timestamp(self) -> float:
        return max(item.measurement["timestamp"] for item in self.entries)


class MeasurementMicroBatcher:
    def __init__(
        self,
        *,
        window_ms: float = 40.0,
        timestamp_tolerance_ms: float = 20.0,
        max_batch_size: int = 100,
        max_wait_ms: float = 75.0,
    ):
        if window_ms <= 0 or timestamp_tolerance_ms < 0 or max_batch_size <= 0 or max_wait_ms <= 0:
            raise ValueError("Micro-batch esikleri gecersiz")
        self.window_s = float(window_ms) / 1000.0
        self.tolerance_s = float(timestamp_tolerance_ms) / 1000.0
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self.groups: list[MeasurementGroup] = []

    def __len__(self) -> int:
        return sum(len(group.entries) for group in self.groups)

    def add(self, entry: BufferedMeasurement) -> None:
        timestamp = float(entry.measurement["timestamp"])
        compatible = [
            group for group in self.groups
            if max(group.end_timestamp, timestamp) - min(group.start_timestamp, timestamp)
            <= self.tolerance_s
        ]
        if compatible:
            group = min(compatible, key=lambda item: abs(timestamp - item.end_timestamp))
            group.entries.append(entry)
        else:
            self.groups.append(MeasurementGroup([entry], entry.received_at_monotonic))
        self.groups.sort(key=lambda item: item.start_timestamp)

    def pop_ready_batches(self, current_monotonic_time: float, *, force: bool = False):
        ready: list[list[BufferedMeasurement]] = []
        while self.groups:
            group = self.groups[0]
            age = float(current_monotonic_time) - group.first_received_monotonic
            newer_group_exists = len(self.groups) > 1
            should_flush = (
                force
                or len(group.entries) >= self.max_batch_size
                or newer_group_exists
                or age >= self.window_s
                or age >= self.max_wait_s
            )
            if not should_flush:
                break
            self.groups.pop(0)
            ready.append(sorted(group.entries, key=lambda item: (
                float(item.measurement["timestamp"]),
                str(item.measurement["src"][0]),
                str(item.measurement["src"][1]),
                str(item.measurement["measurement_id"]),
            )))
        return ready
