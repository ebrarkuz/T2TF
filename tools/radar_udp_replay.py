"""Mevcut radar CSV/JSON verisini sürümlü UDP mesajları olarak tekrar oynatır."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime.message_schema import encode_message


def row_to_message(row: pd.Series, sequence: int, timestamp: float) -> dict:
    if isinstance(row.get("position"), dict) and isinstance(row.get("velocity"), dict):
        message = row.to_dict()
        message.update({
            "schema_version": 1,
            "message_type": "radar_measurement",
            "measurement_id": f"{str(message.get('sensor_id', 'radar')).lower()}_{sequence:012d}",
            "timestamp": float(timestamp),
            "sequence_number": int(sequence),
            "coordinate_frame": "ENU",
        })
        message.setdefault("measurement", {})
        message.setdefault("covariance", None)
        message.setdefault("quality", {})
        return message
    sensor = str(row.get("sensor", row.get("sensor_id", "RADAR_UNKNOWN")))
    local_track = row.get("local_track_id")
    return {
        "schema_version": 1,
        "message_type": "radar_measurement",
        "measurement_id": f"{sensor.lower()}_{sequence:012d}",
        "sensor_id": sensor,
        "source_track_id": None if pd.isna(local_track) else str(local_track),
        "target_hint": None if pd.isna(row.get("callsign_true")) else str(row.get("callsign_true")),
        "timestamp": float(timestamp),
        "sequence_number": int(sequence),
        "coordinate_frame": "ENU",
        "position": {
            "x_m": float(row["x"]), "y_m": float(row["y"]), "z_m": float(row.get("z", 0.0)),
        },
        "velocity": {
            "vx_mps": float(row["vx"]), "vy_mps": float(row["vy"]), "vz_mps": float(row.get("vz", 0.0)),
        },
        "measurement": {
            "range_m": None if pd.isna(row.get("range")) else float(row.get("range")),
            "azimuth_rad": None if pd.isna(row.get("azimuth")) else float(row.get("azimuth")),
            "elevation_rad": None if pd.isna(row.get("elevation")) else float(row.get("elevation")),
            "radial_velocity_mps": None if pd.isna(row.get("radial_velocity")) else float(row.get("radial_velocity")),
        },
        "covariance": None,
        "track_quality": float(row.get("track_quality", 8.0)),
        "sigma_pos_m": float(row.get("sigma_pos_m", 100.0)),
        "sigma_vel_mps": float(row.get("sigma_vel_mps", 5.0)),
        "quality": {"snr_db": None, "detection_probability": None},
    }


def load_rows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).sort_values("time", kind="stable", ignore_index=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    frame = pd.DataFrame(data)
    time_column = "time" if "time" in frame else "timestamp"
    return frame.sort_values(time_column, kind="stable", ignore_index=True)


def replay(path: Path, host: str, port: int, speed: float, loop: bool) -> None:
    if speed < 0.0:
        raise ValueError("--speed negatif olamaz")
    rows = load_rows(path)
    if rows.empty:
        raise ValueError("Replay girişi boş")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    try:
        while True:
            time_column = "time" if "time" in rows else "timestamp"
            base_source_time = float(rows.iloc[0][time_column])
            base_wall_time = time.time()
            previous_source_time = base_source_time
            for _, row in rows.iterrows():
                source_time = float(row[time_column])
                if speed > 0.0:
                    time.sleep(max(0.0, (source_time - previous_source_time) / speed))
                replay_timestamp = (
                    base_wall_time + (source_time - base_source_time) / speed
                    if speed > 0.0 else time.time()
                )
                message = row_to_message(row, sequence, replay_timestamp)
                sock.sendto(encode_message(message), (host, port))
                sequence += 1
                previous_source_time = source_time
            if not loop:
                break
    finally:
        sock.close()
    print(f"Gönderilen radar ölçümü: {sequence}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Radar CSV/JSON UDP replay")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    replay(args.input, args.host, args.port, args.speed, args.loop)


if __name__ == "__main__":
    main()
