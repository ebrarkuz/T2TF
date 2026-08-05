"""YAML ve environment override destekli urun konfigurasyonu."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    radar_udp_host: str = "0.0.0.0"
    radar_udp_port: int = 7777
    fused_udp_host: str = "127.0.0.1"
    fused_udp_port: int = 8888
    fusion_algorithm: str = "dual_imm"
    fusion_parameters: dict[str, Any] = field(default_factory=dict)
    ground_truth_path: str = "data/ground_truth_adsb_multi.csv"
    example_radar_path: str = "data/radar_sensor_tracks_gercekci.csv"
    state_snapshot_path: str = "runtime/realtime_state.json"
    max_visible_radar_measurements: int = 20
    max_fused_points_per_track: int = 500
    radar_queue_maxsize: int = 1000
    fused_queue_maxsize: int = 1000
    visualization_refresh_ms: int = 250
    max_packet_age_s: float = 2.0
    max_out_of_order_s: float = 0.25
    max_line_gap_s: float = 1.0
    max_udp_packet_bytes: int = 65507
    duplicate_cache_size: int = 10000
    drop_late_packets: bool = True
    queue_overflow_policy: str = "drop_oldest"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RuntimeConfig":
        config_path = Path(path or os.getenv("FUSION_CONFIG", DEFAULT_CONFIG_PATH))
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {config_path}") from exc
        network, fusion = raw.get("network", {}), raw.get("fusion", {})
        visualization, data = raw.get("visualization", {}), raw.get("data", {})
        runtime = raw.get("runtime", {})
        selected_algorithm = os.getenv("FUSION_ALGORITHM", str(fusion.get("algorithm", "dual_imm")))
        profiles = fusion.get("profiles", {}) or {}
        selected_parameters = profiles.get(selected_algorithm, fusion.get("parameters", {})) or {}
        cfg = cls(
            radar_udp_host=os.getenv("RADAR_UDP_HOST", str(network.get("radar_host", "0.0.0.0"))),
            radar_udp_port=int(os.getenv("RADAR_UDP_PORT", network.get("radar_port", 7777))),
            fused_udp_host=os.getenv("FUSED_UDP_HOST", str(network.get("fused_host", "127.0.0.1"))),
            fused_udp_port=int(os.getenv("FUSED_UDP_PORT", network.get("fused_port", 8888))),
            fusion_algorithm=selected_algorithm,
            fusion_parameters=dict(selected_parameters),
            ground_truth_path=os.getenv("GROUND_TRUTH_PATH", str(data.get("ground_truth_csv", "data/ground_truth_adsb_multi.csv"))),
            example_radar_path=os.getenv("EXAMPLE_RADAR_PATH", str(data.get("example_radar_csv", "data/radar_sensor_tracks_gercekci.csv"))),
            state_snapshot_path=os.getenv("REALTIME_STATE_PATH", str(runtime.get("state_snapshot", "runtime/realtime_state.json"))),
            max_visible_radar_measurements=int(os.getenv("MAX_VISIBLE_RADAR_MEASUREMENTS", visualization.get("max_visible_radar_measurements", 20))),
            max_fused_points_per_track=int(os.getenv("MAX_FUSED_POINTS_PER_TRACK", visualization.get("max_fused_points_per_track", 500))),
            visualization_refresh_ms=int(os.getenv("VISUALIZATION_REFRESH_MS", visualization.get("refresh_interval_ms", 250))),
            radar_queue_maxsize=int(os.getenv("RADAR_QUEUE_MAXSIZE", runtime.get("radar_queue_maxsize", 1000))),
            fused_queue_maxsize=int(os.getenv("FUSED_QUEUE_MAXSIZE", runtime.get("fused_queue_maxsize", 1000))),
            max_packet_age_s=float(os.getenv("MAX_PACKET_AGE_S", runtime.get("max_packet_age_s", 2.0))),
            max_out_of_order_s=float(os.getenv("MAX_OUT_OF_ORDER_S", runtime.get("max_out_of_order_s", 0.25))),
            max_line_gap_s=float(os.getenv("MAX_LINE_GAP_S", visualization.get("max_line_gap_s", 1.0))),
            max_udp_packet_bytes=int(os.getenv("MAX_UDP_PACKET_BYTES", runtime.get("max_udp_packet_bytes", 65507))),
            duplicate_cache_size=int(os.getenv("DUPLICATE_CACHE_SIZE", runtime.get("duplicate_cache_size", 10000))),
            drop_late_packets=_env_bool("DROP_LATE_PACKETS", bool(runtime.get("drop_late_packets", True))),
            queue_overflow_policy=os.getenv("QUEUE_OVERFLOW_POLICY", str(runtime.get("queue_overflow_policy", "drop_oldest"))),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0 <= self.radar_udp_port <= 65535 or not 0 <= self.fused_udp_port <= 65535:
            raise ValueError("UDP portlari 0..65535 araliginda olmali")
        for name in ("max_visible_radar_measurements", "max_fused_points_per_track",
                     "radar_queue_maxsize", "fused_queue_maxsize", "duplicate_cache_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} sifirdan buyuk olmali")
        if self.queue_overflow_policy not in {"drop_oldest", "reject_new"}:
            raise ValueError("QUEUE_OVERFLOW_POLICY drop_oldest veya reject_new olmali")

    @property
    def snapshot_path(self) -> Path:
        return Path(self.state_snapshot_path)
