"""Füzyondan bağımsız, bounded ve thread-safe görsel durum deposu."""

from __future__ import annotations

import copy
import json
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


class RealtimeMapState:
    def __init__(self, max_radar_measurements: int = 20, max_fused_per_track: int = 500):
        self.max_radar_measurements = max_radar_measurements
        self.max_fused_per_track = max_fused_per_track
        self._radar = deque(maxlen=max_radar_measurements)
        self._fused: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_fused_per_track)
        )
        self._diagnostics: dict[str, Any] = {}
        self._lock = threading.RLock()

    def add_radar(self, measurement: dict[str, Any]) -> None:
        with self._lock:
            self._radar.append(copy.deepcopy(measurement))

    def add_fused(self, fused_state: dict[str, Any]) -> None:
        with self._lock:
            self._fused[str(fused_state["track_id"])].append(copy.deepcopy(fused_state))

    def set_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        with self._lock:
            self._diagnostics = copy.deepcopy(diagnostics)

    def clear_visual_history(self) -> None:
        with self._lock:
            self._radar.clear()
            self._fused.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "radar_measurements": copy.deepcopy(list(self._radar)),
                "fused_tracks": {
                    track_id: copy.deepcopy(list(points))
                    for track_id, points in self._fused.items()
                },
                "diagnostics": copy.deepcopy(self._diagnostics),
            }

    def write_snapshot(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)


def load_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"radar_measurements": [], "fused_tracks": {}, "diagnostics": {}}
