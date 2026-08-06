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
    telemetry_udp_host: str = "127.0.0.1"
    telemetry_udp_port: int = 8890
    fusion_algorithm: str = "dual_imm"
    fusion_parameters: dict[str, Any] = field(default_factory=dict)
    ground_truth_path: str = "data/ground_truth_adsb_multi.csv"
    example_radar_path: str = "data/radar_sensor_tracks_gercekci.csv"
    max_visible_radar_measurements: int = 20
    max_visible_radar_measurements_per_sensor: int = 20
    max_fused_points_per_track: int = 2000
    track_archive_timeout_s: float = 5.0
    preserve_archived_tracks: bool = True
    full_visual_history: bool = False
    fused_display_interval_s: float = 1.0
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
    micro_batch_enabled: bool = True
    micro_batch_window_ms: float = 40.0
    timestamp_group_tolerance_ms: float = 5.0
    max_micro_batch_size: int = 100
    max_micro_batch_wait_ms: float = 75.0
    minimum_visible_tentative_points: int = 2

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
            telemetry_udp_host=os.getenv("TELEMETRY_UDP_HOST", str(visualization.get("telemetry_host", "127.0.0.1"))),
            telemetry_udp_port=int(os.getenv("TELEMETRY_UDP_PORT", visualization.get("telemetry_port", 8890))),
            fusion_algorithm=selected_algorithm,
            fusion_parameters=dict(selected_parameters),
            ground_truth_path=os.getenv("GROUND_TRUTH_PATH", str(data.get("ground_truth_csv", "data/ground_truth_adsb_multi.csv"))),
            example_radar_path=os.getenv("EXAMPLE_RADAR_PATH", str(data.get("example_radar_csv", "data/radar_sensor_tracks_gercekci.csv"))),
            max_visible_radar_measurements=int(os.getenv("MAX_VISIBLE_RADAR_MEASUREMENTS", visualization.get("max_visible_radar_measurements", 20))),
            max_visible_radar_measurements_per_sensor=int(os.getenv("MAX_VISIBLE_RADAR_MEASUREMENTS_PER_SENSOR", visualization.get("max_visible_radar_measurements_per_sensor", 20))),
            max_fused_points_per_track=int(os.getenv("MAX_FUSED_POINTS_PER_TRACK", visualization.get("max_fused_points_per_track", 2000))),
            track_archive_timeout_s=float(os.getenv("TRACK_ARCHIVE_TIMEOUT_S", visualization.get("track_archive_timeout_s", 5.0))),
            preserve_archived_tracks=_env_bool("PRESERVE_ARCHIVED_TRACKS", bool(visualization.get("preserve_archived_tracks", True))),
            full_visual_history=_env_bool("FULL_VISUAL_HISTORY", bool(visualization.get("full_visual_history", False))),
            fused_display_interval_s=float(os.getenv("FUSED_DISPLAY_INTERVAL_S", visualization.get("fused_display_interval_s", 1.0))),
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
            micro_batch_enabled=_env_bool("MICRO_BATCH_ENABLED", bool(runtime.get("micro_batch_enabled", True))),
            micro_batch_window_ms=float(os.getenv("MICRO_BATCH_WINDOW_MS", runtime.get("micro_batch_window_ms", 40.0))),
            timestamp_group_tolerance_ms=float(os.getenv("TIMESTAMP_GROUP_TOLERANCE_MS", runtime.get("timestamp_group_tolerance_ms", 5.0))),
            max_micro_batch_size=int(os.getenv("MAX_MICRO_BATCH_SIZE", runtime.get("max_micro_batch_size", 100))),
            max_micro_batch_wait_ms=float(os.getenv("MAX_MICRO_BATCH_WAIT_MS", runtime.get("max_micro_batch_wait_ms", 75.0))),
            minimum_visible_tentative_points=int(os.getenv("MINIMUM_VISIBLE_TENTATIVE_POINTS", visualization.get("minimum_visible_tentative_points", 2))),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not all(0 <= port <= 65535 for port in (
            self.radar_udp_port, self.fused_udp_port, self.telemetry_udp_port
        )):
            raise ValueError("UDP portlari 0..65535 araliginda olmali")
        for name in ("max_visible_radar_measurements", "max_visible_radar_measurements_per_sensor", "max_fused_points_per_track",
                     "radar_queue_maxsize", "fused_queue_maxsize", "duplicate_cache_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} sifirdan buyuk olmali")
        if self.track_archive_timeout_s <= 0:
            raise ValueError("track_archive_timeout_s sifirdan buyuk olmali")
        if self.fused_display_interval_s <= 0:
            raise ValueError("fused_display_interval_s sifirdan buyuk olmali")
        if self.minimum_visible_tentative_points < 1:
            raise ValueError("minimum_visible_tentative_points en az 1 olmali")
        if (
            self.micro_batch_window_ms <= 0
            or self.timestamp_group_tolerance_ms < 0
            or self.max_micro_batch_size <= 0
            or self.max_micro_batch_wait_ms <= 0
        ):
            raise ValueError("Micro-batch konfigurasyonu gecersiz")
        if self.queue_overflow_policy not in {"drop_oldest", "reject_new"}:
            raise ValueError("QUEUE_OVERFLOW_POLICY drop_oldest veya reject_new olmali")
