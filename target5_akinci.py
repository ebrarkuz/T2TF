"""AKINCI-benzeri Hedef 5 yörüngesi ve mevcut radar modelleriyle entegrasyonu."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq


TARGET5_CALLSIGN = "HEDEF_5"
TARGET5_NAME = "AKINCI Benzeri Dışa Açılan Spiral Tırmanış Rotası"
GROUND_TRUTH_BASE_COLUMNS = [
    "time", "x", "y", "z", "vx", "vy", "vz", "callsign",
    "lat", "lon", "alt_m", "time_unix",
]


@dataclass(frozen=True)
class Target5Parameters:
    sample_rate_hz: float = 20.0
    total_speed_mps: float = 77.0
    turn_count: int = 4
    radius_start_m: float = 500.0
    radius_end_m: float = 1800.0
    climb_rate_mps: float = 5.0
    center_x_m: float = 100000.0
    center_y_m: float = 100000.0
    start_altitude_m: float = 3000.0
    turn_direction: str = "ccw"
    target_id: int = 5
    min_separation_m: float = 1000.0


def _validate_parameters(params: Target5Parameters) -> None:
    if params.sample_rate_hz <= 0.0 or params.sample_rate_hz > 100.0:
        raise ValueError("sample_rate_hz 0–100 Hz aralığında olmalıdır")
    if params.turn_count < 1 or params.turn_count > 10:
        raise ValueError("turn_count 1–10 aralığında olmalıdır")
    if params.radius_start_m <= 0.0:
        raise ValueError("radius_start_m sıfırdan büyük olmalıdır")
    if params.radius_end_m <= params.radius_start_m:
        raise ValueError("radius_end_m, radius_start_m değerinden büyük olmalıdır")
    if abs(params.climb_rate_mps) >= params.total_speed_mps:
        raise ValueError("climb_rate_mps toplam hızdan küçük olmalıdır")
    if params.turn_direction.lower() not in {"cw", "ccw"}:
        raise ValueError("turn_direction yalnızca 'cw' veya 'ccw' olabilir")
    if params.target_id != 5:
        raise ValueError("Hedef 5 üreticisinde target_id=5 olmalıdır")
    if params.min_separation_m <= 0.0:
        raise ValueError("min_separation_m sıfırdan büyük olmalıdır")


def _reference_coordinates(reference_gt: Optional[pd.DataFrame]) -> tuple[float, float, float, float]:
    if reference_gt is None or reference_gt.empty:
        return 50.0, 14.0, 0.0, 0.0
    ref_lat = float(reference_gt["lat"].median()) if "lat" in reference_gt else 50.0
    ref_lon = float(reference_gt["lon"].median()) if "lon" in reference_gt else 14.0
    if {"alt_m", "z"}.issubset(reference_gt.columns):
        ref_alt = float((reference_gt["alt_m"] - reference_gt["z"]).median())
    else:
        ref_alt = 0.0
    time_unix_0 = (
        float((reference_gt["time_unix"] - reference_gt["time"]).median())
        if {"time_unix", "time"}.issubset(reference_gt.columns)
        else 0.0
    )
    return ref_lat, ref_lon, ref_alt, time_unix_0


def _enu_to_latlon(x: np.ndarray, y: np.ndarray, ref_lat: float, ref_lon: float):
    earth_radius_m = 6371000.0
    lat = ref_lat + np.degrees(y / earth_radius_m)
    lon = ref_lon + np.degrees(x / (earth_radius_m * math.cos(math.radians(ref_lat))))
    return lat, lon


def target_separations(target5_df: pd.DataFrame, other_gt: pd.DataFrame) -> Dict[str, float]:
    """Hedef 5'in her mevcut hedefe eşzamanlı minimum 3B ayrımını hesapla."""
    if other_gt is None or other_gt.empty:
        return {}
    target_key = "callsign" if "callsign" in other_gt.columns else "target"
    result: Dict[str, float] = {}
    times = target5_df["time"].to_numpy(float)
    target_positions = target5_df[["x", "y", "z"]].to_numpy(float)
    for target, group in other_gt.groupby(target_key):
        if str(target) == TARGET5_CALLSIGN:
            continue
        group = group.sort_values("time").drop_duplicates("time")
        overlap = (times >= float(group["time"].min())) & (times <= float(group["time"].max()))
        if not overlap.any():
            continue
        interpolated = np.column_stack([
            np.interp(times[overlap], group["time"], group[axis])
            for axis in ("x", "y", "z")
        ])
        distances = np.linalg.norm(target_positions[overlap] - interpolated, axis=1)
        result[str(target)] = float(distances.min())
    return result


def generate_target5_akinci_climbing_arc(
    sample_rate_hz: float = 20.0,
    total_speed_mps: float = 77.0,
    turn_count: int = 4,
    radius_start_m: float = 500.0,
    radius_end_m: float = 1800.0,
    climb_rate_mps: float = 5.0,
    start_position: tuple[float, float, float] = (100000.0, 100000.0, 3000.0),
    turn_direction: str = "ccw",
    target_id: int = 5,
    other_ground_truth: Optional[pd.DataFrame] = None,
    min_separation_m: float = 1000.0,
) -> pd.DataFrame:
    """Dışa açılan dört turluk spiral helisi yaklaşık sabit 3B hızda üretir.

    ``start_position`` içindeki ilk iki değer spiral merkezidir; üçüncü değer
    başlangıç irtifasıdır. Süre, tam eğrinin 3B yay uzunluğundan hesaplanır.
    """
    params = Target5Parameters(
        sample_rate_hz=sample_rate_hz,
        total_speed_mps=total_speed_mps,
        turn_count=int(turn_count),
        radius_start_m=radius_start_m,
        radius_end_m=radius_end_m,
        climb_rate_mps=climb_rate_mps,
        center_x_m=float(start_position[0]),
        center_y_m=float(start_position[1]),
        start_altitude_m=float(start_position[2]),
        turn_direction=turn_direction,
        target_id=target_id,
        min_separation_m=min_separation_m,
    )
    _validate_parameters(params)

    raw_point_count = max(100000, int(turn_count * 25000))
    total_angle_rad = 2.0 * np.pi * turn_count
    theta = np.linspace(0.0, total_angle_rad, raw_point_count)
    direction_sign = 1.0 if turn_direction.lower() == "ccw" else -1.0
    radius_growth_per_rad = (radius_end_m - radius_start_m) / total_angle_rad
    radius = radius_start_m + radius_growth_per_rad * theta
    raw_x_offset = radius * np.cos(theta)
    raw_y_offset = direction_sign * radius * np.sin(theta)
    horizontal_segment_lengths = np.hypot(
        np.diff(raw_x_offset), np.diff(raw_y_offset)
    )
    horizontal_arc_length_m = float(horizontal_segment_lengths.sum())
    def total_length_for_duration(candidate_duration_s: float) -> float:
        dz_per_segment = climb_rate_mps * candidate_duration_s / (raw_point_count - 1)
        return float(np.sqrt(horizontal_segment_lengths**2 + dz_per_segment**2).sum())

    lower_duration_s = horizontal_arc_length_m / total_speed_mps
    upper_duration_s = horizontal_arc_length_m / math.sqrt(
        total_speed_mps**2 - climb_rate_mps**2
    )
    while total_length_for_duration(upper_duration_s) > total_speed_mps * upper_duration_s:
        upper_duration_s *= 1.05
    duration_s = brentq(
        lambda candidate_duration: (
            total_length_for_duration(candidate_duration)
            - total_speed_mps * candidate_duration
        ),
        lower_duration_s,
        upper_duration_s,
    )
    raw_time = np.linspace(0.0, duration_s, raw_point_count)
    raw_z_offset = climb_rate_mps * raw_time

    time_values = np.arange(0.0, duration_s, 1.0 / sample_rate_hz, dtype=float)
    if time_values.size == 0 or not math.isclose(time_values[-1], duration_s, abs_tol=1e-10):
        time_values = np.append(time_values, duration_s)
    desired_distance = total_speed_mps * time_values

    candidate_specs = [
        (params.center_x_m, params.center_y_m, direction_sign, params.start_altitude_m),
        (params.center_x_m + 50000.0, params.center_y_m, direction_sign, params.start_altitude_m),
        (params.center_x_m - 50000.0, params.center_y_m, direction_sign, params.start_altitude_m),
        (params.center_x_m, params.center_y_m + 50000.0, direction_sign, params.start_altitude_m),
        (params.center_x_m, params.center_y_m - 50000.0, direction_sign, params.start_altitude_m),
        (params.center_x_m, params.center_y_m, direction_sign, params.start_altitude_m + 3000.0),
        (params.center_x_m, params.center_y_m, -direction_sign, params.start_altitude_m),
    ]
    ref_lat, ref_lon, ref_alt, time_unix_0 = _reference_coordinates(other_ground_truth)
    result = None
    separations = {}
    for candidate_x, candidate_y, candidate_direction, candidate_z in candidate_specs:
        candidate_y_offset = raw_y_offset * (candidate_direction / direction_sign)
        raw_x = candidate_x + raw_x_offset
        raw_y = candidate_y + candidate_y_offset
        raw_z = candidate_z + raw_z_offset
        segment_lengths = np.sqrt(
            np.diff(raw_x) ** 2 + np.diff(raw_y) ** 2 + np.diff(raw_z) ** 2
        )
        arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_arc_length_m = float(arc_length[-1])
        if total_arc_length_m + 1e-6 < desired_distance[-1]:
            raise ValueError(
                "Spiral ham eğrisi istenen son mesafeden kısa; son nokta tekrarlanmadı"
            )
        x = np.interp(desired_distance, arc_length, raw_x)
        y = np.interp(desired_distance, arc_length, raw_y)
        z = np.interp(desired_distance, arc_length, raw_z)
        sampled_radius = np.interp(desired_distance, arc_length, radius)
        sampled_theta = np.interp(desired_distance, arc_length, theta)
        vx = np.gradient(x, time_values, edge_order=2)
        vy = np.gradient(y, time_values, edge_order=2)
        vz = np.gradient(z, time_values, edge_order=2)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        heading_rad = np.unwrap(np.arctan2(vy, vx))
        turn_rate = np.gradient(heading_rad, time_values, edge_order=2)
        lat, lon = _enu_to_latlon(x, y, ref_lat, ref_lon)
        candidate = pd.DataFrame({
            "time": time_values, "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz, "callsign": TARGET5_CALLSIGN,
            "lat": lat, "lon": lon, "alt_m": ref_alt + z,
            "time_unix": time_unix_0 + time_values, "target_id": target_id,
            "target_name": TARGET5_NAME, "speed": speed,
            "heading": np.degrees(heading_rad), "turn_rate": turn_rate,
            "climb_rate": vz,
            "spiral_center_x_m": candidate_x,
            "spiral_center_y_m": candidate_y,
            "spiral_radius_m": sampled_radius,
            "spiral_theta_rad": sampled_theta,
        })
        separations = target_separations(candidate, other_ground_truth)
        if not separations or min(separations.values()) >= min_separation_m:
            result = candidate
            break

    if result is None:
        details = ", ".join(f"{target}={distance:.1f} m" for target, distance in separations.items())
        raise ValueError(f"Hedef 5 için güvenli spiral rota bulunamadı: {details}")
    numeric = result[["time", "x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Hedef 5 yörüngesinde NaN veya infinity oluştu")
    if not result["time"].is_monotonic_increasing or result["time"].duplicated().any():
        raise ValueError("Hedef 5 timestamp değerleri sıralı ve benzersiz değil")
    result.attrs["minimum_separations_m"] = separations
    result.attrs["turn_count"] = int(turn_count)
    result.attrs["loop_count"] = int(turn_count)
    result.attrs["center_x_m"] = float(candidate_x)
    result.attrs["center_y_m"] = float(candidate_y)
    result.attrs["radius_start_m"] = float(radius_start_m)
    result.attrs["radius_end_m"] = float(radius_end_m)
    result.attrs["duration_s"] = float(duration_s)
    result.attrs["horizontal_arc_length_m"] = float(horizontal_arc_length_m)
    result.attrs["total_arc_length_m"] = float(total_arc_length_m)
    result.attrs["start_altitude_m"] = float(result.iloc[0]["z"])
    result.attrs["end_altitude_m"] = float(result.iloc[-1]["z"])
    result.attrs["altitude_gain_m"] = float(result.iloc[-1]["z"] - result.iloc[0]["z"])
    result.attrs["average_speed_mps"] = float(result["speed"].mean())
    return result


def combine_ground_truth(existing_gt: pd.DataFrame, target5_gt: pd.DataFrame) -> pd.DataFrame:
    """Mevcut hedef satırlarını değiştirmeden Hedef 5'i ekle/değiştir."""
    target_key = "callsign" if "callsign" in existing_gt.columns else "target"
    base = existing_gt[existing_gt[target_key].astype(str) != TARGET5_CALLSIGN].copy()
    return pd.concat([base, target5_gt], ignore_index=True, sort=False).sort_values(
        ["time", target_key], kind="stable", ignore_index=True
    )


def generate_target5_radar_measurements(
    target5_gt: pd.DataFrame,
    *,
    realistic: bool = True,
    seed: int = 42,
    noise_scale: float = 1.0,
    detection_probability: Optional[float] = None,
) -> pd.DataFrame:
    """Mevcut radar modelleriyle yalnızca Hedef 5 ölçümlerini üretir."""
    if noise_scale <= 0.0:
        raise ValueError("noise_scale sıfırdan büyük olmalıdır")
    if detection_probability is not None and not 0.0 <= detection_probability <= 1.0:
        raise ValueError("detection_probability 0–1 aralığında olmalıdır")

    import radar_sim

    frames = []
    random_state = np.random.get_state()
    try:
        for index, radar_config in enumerate(radar_sim.RADARS):
            radar = dict(radar_config)
            if detection_probability is not None:
                radar["pd"] = float(detection_probability)
                radar["pd_min"] = min(float(radar.get("pd_min", 0.0)), radar["pd"])
            np.random.seed(int(seed) + index * 1009)
            if realistic:
                measurements = radar_sim.simulate_radar_gercekci(
                    radar, target5_gt, global_static_clutter=[], ghost_tracks=[]
                )
            else:
                measurements = radar_sim.simulate_radar_idealize(radar, target5_gt)
            measurements = measurements[
                measurements["callsign_true"].astype(str) == TARGET5_CALLSIGN
            ].copy()

            if noise_scale != 1.0 and not measurements.empty:
                truth_time = measurements["time"].to_numpy(float) - float(radar.get("delay_s", 0.0))
                for axis in ("x", "y", "z", "vx", "vy", "vz"):
                    truth = np.interp(truth_time, target5_gt["time"], target5_gt[axis])
                    measurements[axis] = truth + noise_scale * (measurements[axis].to_numpy(float) - truth)
                measurements["sigma_pos_m"] *= noise_scale
                measurements["sigma_vel_mps"] *= noise_scale
            frames.append(measurements)
    finally:
        np.random.set_state(random_state)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("time", kind="stable", ignore_index=True)


def combine_sensor_measurements(existing_sensor: pd.DataFrame, target5_sensor: pd.DataFrame) -> pd.DataFrame:
    """Eski sensör satırlarını koruyup Hedef 5'i mevcut radar taramalarına ekler.

    Her radar tüm hedefleri aynı scan anında gördüğünden, yeni ölçümler o radarın
    en yakın mevcut timestamp'ine hizalanır. Böylece yeni hedef eski track'lere
    yapay ek propagation zamanları oluşturmaz.
    """
    if "callsign_true" not in existing_sensor:
        raise ValueError("Sensör verisinde callsign_true kolonu bulunamadı")
    base = existing_sensor[existing_sensor["callsign_true"].astype(str) != TARGET5_CALLSIGN].copy()
    aligned = target5_sensor.copy()
    for sensor_name, indices in aligned.groupby("sensor").groups.items():
        scan_times = np.sort(
            base.loc[base["sensor"] == sensor_name, "time"].dropna().unique().astype(float)
        )
        if scan_times.size == 0:
            continue
        values = aligned.loc[indices, "time"].to_numpy(float)
        right = np.searchsorted(scan_times, values, side="left").clip(0, scan_times.size - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(values - scan_times[left]) <= np.abs(scan_times[right] - values)
        nearest = np.where(choose_left, scan_times[left], scan_times[right])
        aligned.loc[indices, "time"] = nearest
    aligned = aligned.drop_duplicates(["sensor", "time", "callsign_true"], keep="first")
    return pd.concat([base, aligned], ignore_index=True, sort=False).sort_values(
        "time", kind="stable", ignore_index=True
    )


def write_target5_scenario(
    project_dir: str | Path,
    params: Target5Parameters = Target5Parameters(),
    *,
    seed: int = 42,
    noise_scale: float = 1.0,
    detection_probability: Optional[float] = None,
    update_canonical_files: bool = False,
) -> dict[str, Path]:
    """Birleşik GT/radar senaryosu yazar; varsayılan olarak yeni dosyalar kullanır."""
    root = Path(project_dir)
    canonical_gt_path = root / "ground_truth_adsb_multi.csv"
    canonical_sensor_path = root / "radar_sensor_tracks_gercekci.csv"
    existing_gt = pd.read_csv(canonical_gt_path)
    existing_sensor = pd.read_csv(canonical_sensor_path)
    other_gt = existing_gt[existing_gt["callsign"].astype(str) != TARGET5_CALLSIGN]
    target5_gt = generate_target5_akinci_climbing_arc(
        sample_rate_hz=params.sample_rate_hz,
        total_speed_mps=params.total_speed_mps,
        turn_count=params.turn_count,
        radius_start_m=params.radius_start_m,
        radius_end_m=params.radius_end_m,
        climb_rate_mps=params.climb_rate_mps,
        start_position=(params.center_x_m, params.center_y_m, params.start_altitude_m),
        turn_direction=params.turn_direction,
        target_id=params.target_id,
        other_ground_truth=other_gt,
        min_separation_m=params.min_separation_m,
    )
    target5_sensor = generate_target5_radar_measurements(
        target5_gt,
        realistic=True,
        seed=seed,
        noise_scale=noise_scale,
        detection_probability=detection_probability,
    )
    combined_gt = combine_ground_truth(existing_gt, target5_gt)
    if update_canonical_files:
        sensor_base = existing_sensor
    else:
        # Hedef 5 analiz dosyası yalnızca ortak zaman penceresini taşır; böylece
        # Streamlit'e yüzlerce MB gereksiz geçmiş gönderilmez.
        sensor_base = existing_sensor[
            existing_sensor["time"] <= float(target5_gt["time"].max()) + 2.0
        ].copy()
    combined_sensor = combine_sensor_measurements(sensor_base, target5_sensor)

    gt_path = canonical_gt_path if update_canonical_files else root / "ground_truth_adsb_multi_target5.csv"
    sensor_path = canonical_sensor_path if update_canonical_files else root / "radar_sensor_tracks_gercekci_target5.csv"
    combined_gt.to_csv(gt_path, index=False)
    combined_sensor.to_csv(sensor_path, index=False)
    return {
        "ground_truth": gt_path,
        "sensor": sensor_path,
        "target5_ground_truth": target5_gt,
        "target5_sensor": target5_sensor,
    }


def update_project_target5_data(
    project_dir: str | Path,
    params: Target5Parameters = Target5Parameters(),
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Kanonik GT ile idealize/gerçekçi radar CSV'lerine yalnızca Hedef 5'i ekler."""
    root = Path(project_dir)
    gt_path = root / "ground_truth_adsb_multi.csv"
    realistic_path = root / "radar_sensor_tracks_gercekci.csv"
    idealized_path = root / "radar_sensor_tracks_idealize.csv"
    existing_gt = pd.read_csv(gt_path)
    other_gt = existing_gt[existing_gt["callsign"].astype(str) != TARGET5_CALLSIGN].copy()
    target5_gt = generate_target5_akinci_climbing_arc(
        sample_rate_hz=params.sample_rate_hz,
        total_speed_mps=params.total_speed_mps,
        turn_count=params.turn_count,
        radius_start_m=params.radius_start_m,
        radius_end_m=params.radius_end_m,
        climb_rate_mps=params.climb_rate_mps,
        start_position=(params.center_x_m, params.center_y_m, params.start_altitude_m),
        turn_direction=params.turn_direction,
        target_id=params.target_id,
        other_ground_truth=other_gt,
        min_separation_m=params.min_separation_m,
    )
    combined_gt = combine_ground_truth(other_gt, target5_gt)

    sensor_outputs = {}
    for path, realistic in ((realistic_path, True), (idealized_path, False)):
        existing_sensor = pd.read_csv(path)
        old_rows = existing_sensor[
            existing_sensor["callsign_true"].astype(str) != TARGET5_CALLSIGN
        ].copy()
        target5_sensor = generate_target5_radar_measurements(
            target5_gt, realistic=realistic, seed=seed
        )
        combined_sensor = combine_sensor_measurements(old_rows, target5_sensor)
        # Birleştirme eski hedef değerlerini değiştirmemelidir.
        recovered = combined_sensor[
            combined_sensor["callsign_true"].astype(str) != TARGET5_CALLSIGN
        ][old_rows.columns].sort_values("time", kind="stable", ignore_index=True)
        expected = old_rows.sort_values("time", kind="stable", ignore_index=True)
        pd.testing.assert_frame_equal(expected, recovered, check_dtype=False)
        combined_sensor.to_csv(path, index=False)
        sensor_outputs["realistic" if realistic else "idealized"] = target5_sensor

    combined_gt.to_csv(gt_path, index=False)
    return {
        "ground_truth": target5_gt,
        "realistic_sensor": sensor_outputs["realistic"],
        "idealized_sensor": sensor_outputs["idealized"],
        "minimum_separations_m": target5_gt.attrs["minimum_separations_m"],
        "loop_count": target5_gt.attrs["turn_count"],
        "average_speed_mps": target5_gt.attrs["average_speed_mps"],
    }


def associate_fused_tracks_to_target5(
    target5_gt: pd.DataFrame,
    fused_df: pd.DataFrame,
    max_match_distance_m: float = 400.0,
    max_allowed_gap_s: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track'leri tüm zaman örtüşmesi üzerinden Hedef 5 ile ilişkilendirir.

    Her global ID bağımsız değerlendirilir. Dönen ilk tablo yalnızca gate içindeki
    satırları içerir ve ``segment_id`` ile yapay çizgi bağlantıları engellenir.
    """
    if target5_gt.empty or fused_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    gt = target5_gt.sort_values("time").drop_duplicates("time")
    gt_for_merge = gt[["time", "x", "y", "z", "vx", "vy", "vz"]].rename(
        columns={axis: f"gt_{axis}" for axis in ("x", "y", "z", "vx", "vy", "vz")}
    )
    median_dt = float(np.median(np.diff(gt["time"]))) if len(gt) > 1 else 0.05
    time_tolerance_s = max(0.075, 1.5 * median_dt)
    associated_parts = []
    summaries = []
    for track_id, track_df in fused_df.groupby("global_track_id", sort=False):
        track_df = track_df.sort_values("time").copy()
        track_df = track_df[
            (track_df["time"] >= float(gt["time"].min()))
            & (track_df["time"] <= float(gt["time"].max()))
        ]
        if track_df.empty:
            continue
        merged = pd.merge_asof(
            track_df,
            gt_for_merge,
            on="time",
            direction="nearest",
            tolerance=time_tolerance_s,
        )
        merged["time_match_missing"] = merged["gt_x"].isna()
        for axis in ("x", "y", "z", "vx", "vy", "vz"):
            merged[f"error_{axis}"] = merged[axis] - merged[f"gt_{axis}"]
        merged["position_error_3d"] = np.sqrt(
            merged["error_x"] ** 2 + merged["error_y"] ** 2 + merged["error_z"] ** 2
        )
        merged["velocity_error_3d"] = np.sqrt(
            merged["error_vx"] ** 2 + merged["error_vy"] ** 2 + merged["error_vz"] ** 2
        )
        merged["gt_speed"] = np.sqrt(
            merged["gt_vx"] ** 2 + merged["gt_vy"] ** 2 + merged["gt_vz"] ** 2
        )
        merged["estimated_speed"] = np.sqrt(
            merged["vx"] ** 2 + merged["vy"] ** 2 + merged["vz"] ** 2
        )
        merged["gt_heading"] = np.degrees(np.arctan2(merged["gt_vy"], merged["gt_vx"]))
        merged["estimated_heading"] = np.degrees(np.arctan2(merged["vy"], merged["vx"]))
        inside = merged["position_error_3d"].le(max_match_distance_m)
        matched_part = merged[inside].copy()
        if len(matched_part) < 2:
            continue
        matched_part["global_track_id"] = matched_part["global_track_id"].astype(str)
        matched_part["segment_id"] = (
            matched_part["time"].diff().fillna(0.0) > max_allowed_gap_s
        ).cumsum().astype(int)
        associated_parts.append(matched_part)
        summaries.append({
            "track_id": str(track_id),
            "start_time": float(matched_part["time"].min()),
            "end_time": float(matched_part["time"].max()),
            "point_count": int(len(matched_part)),
            "duration": float(matched_part["time"].max() - matched_part["time"].min()),
            "matched_target_id": TARGET5_CALLSIGN,
            "mean_3d_error": float(matched_part["position_error_3d"].mean()),
            "max_3d_error": float(matched_part["position_error_3d"].max()),
            "prediction_only_count": int(matched_part.get("is_prediction_only", pd.Series(False, index=matched_part.index)).fillna(False).astype(bool).sum()),
            "time_match_rejected": int(merged["time_match_missing"].sum()),
            "max_time_gap_s": float(matched_part["time"].diff().max()) if len(matched_part) > 1 else 0.0,
        })
    associated = (
        pd.concat(associated_parts, ignore_index=True)
        if associated_parts else pd.DataFrame()
    )
    summary = pd.DataFrame(summaries).sort_values(
        ["point_count", "duration"], ascending=False, ignore_index=True
    ) if summaries else pd.DataFrame()
    return associated, summary


def match_fused_to_target5(
    target5_gt: pd.DataFrame,
    fused_df: pd.DataFrame,
    max_match_distance_m: float = 400.0,
) -> pd.DataFrame:
    """Metrikler için her timestamp'teki en yakın ilişkili Hedef 5 state'ini seçer."""
    associated, _ = associate_fused_tracks_to_target5(
        target5_gt, fused_df, max_match_distance_m=max_match_distance_m
    )
    if associated.empty:
        return associated
    return associated.sort_values(
        ["time", "position_error_3d"]
    ).drop_duplicates("time", keep="first").sort_values("time", ignore_index=True)
