import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.groundtruth import GroundTruthState

GROUND_TRUTH_CSV = "ground_truth_adsb_multi.csv"
OUTPUT_RADAR_CSV = "radar_measurements_stonesoup.csv"

rng = np.random.default_rng(42)

RADARS = [
    {
        "name": "RADAR_A",
        "revisit_s": 2.5,
        "delay_s": 0.3,
        "pd": 0.95,
        "sigma_pos": 80.0,
        "sigma_vel": 2.0,
        "sigma_angle_rad": np.deg2rad(0.05),
        "clutter_per_scan": 0.04,
    },
    {
        "name": "RADAR_B",
        "revisit_s": 5.0,
        "delay_s": 0.8,
        "pd": 0.88,
        "sigma_pos": 150.0,
        "sigma_vel": 4.0,
        "sigma_angle_rad": np.deg2rad(0.08),
        "clutter_per_scan": 0.06,
    },
    {
        "name": "RADAR_C",
        "revisit_s": 10.0,
        "delay_s": 1.5,
        "pd": 0.80,
        "sigma_pos": 300.0,
        "sigma_vel": 7.0,
        "sigma_angle_rad": np.deg2rad(0.12),
        "clutter_per_scan": 0.08,
    },
]


def build_stonesoup_models(radar, seed):
    """Create the Stone Soup radar and velocity measurement models."""
    radar_sensor = RadarElevationBearingRangeRate(
        ndim_state=6,
        position_mapping=(0, 2, 4),
        velocity_mapping=(1, 3, 5),
        noise_covar=CovarianceMatrix(np.diag([
            radar["sigma_angle_rad"] ** 2,
            radar["sigma_angle_rad"] ** 2,
            radar["sigma_pos"] ** 2,
            radar["sigma_vel"] ** 2,
        ])),
        position=StateVector([0.0, 0.0, 0.0]),
        seed=seed,
    )
    velocity_model = LinearGaussian(
        ndim_state=6,
        mapping=(1, 3, 5),
        noise_covar=CovarianceMatrix(np.eye(3) * radar["sigma_vel"] ** 2),
        seed=seed + 1000,
    )
    return radar_sensor, velocity_model


def stonesoup_measurement(truth, timestamp, radar_sensor, velocity_model, callsign):
    """Measure one Cartesian truth state through Stone Soup models."""
    state = GroundTruthState(
        StateVector([
            truth["x"], truth["vx"],
            truth["y"], truth["vy"],
            truth["z"], truth["vz"],
        ]),
        timestamp=timestamp,
        metadata={"callsign": callsign},
    )
    detection = next(iter(radar_sensor.measure({state}, noise=True)))
    elevation, bearing, measured_range, _ = map(float, detection.state_vector[:, 0])

    cos_elevation = np.cos(elevation)
    x = measured_range * cos_elevation * np.cos(bearing)
    y = measured_range * cos_elevation * np.sin(bearing)
    z = measured_range * np.sin(elevation)
    vx, vy, vz = map(float, velocity_model.function(state, noise=True)[:, 0])
    return {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}


def interpolate_target(target_df, t):
    target_df = target_df.sort_values("time")

    return {
        "x": np.interp(t, target_df["time"], target_df["x"]),
        "y": np.interp(t, target_df["time"], target_df["y"]),
        "z": np.interp(t, target_df["time"], target_df["z"]),
        "vx": np.interp(t, target_df["time"], target_df["vx"]),
        "vy": np.interp(t, target_df["time"], target_df["vy"]),
        "vz": np.interp(t, target_df["time"], target_df["vz"]),
    }


def generate_radar_data(gt_df):
    records = []
    epoch = datetime(2000, 1, 1)

    t_min = float(gt_df["time"].min())
    t_max = float(gt_df["time"].max())

    callsigns = gt_df["callsign"].unique()

    for radar_index, radar in enumerate(RADARS):
        radar_sensor, velocity_model = build_stonesoup_models(radar, 42 + radar_index)
        scan_time = t_min

        while scan_time <= t_max:
            for callsign in callsigns:
                target_df = gt_df[gt_df["callsign"] == callsign]

                target_start = float(target_df["time"].min())
                target_end = float(target_df["time"].max())

                if not target_start <= scan_time <= target_end:
                    continue

                if rng.random() > radar["pd"]:
                    continue

                truth = interpolate_target(target_df, scan_time)
                measured = stonesoup_measurement(
                    truth,
                    epoch + timedelta(seconds=float(scan_time)),
                    radar_sensor,
                    velocity_model,
                    callsign,
                )

                records.append({
                    "time": round(scan_time, 3),
                    "arrival_time": round(scan_time + radar["delay_s"], 3),
                    "sensor": radar["name"],
                    "local_track_id": f"{radar['name']}-{callsign}",
                    "callsign_true": callsign,

                    **measured,

                    "track_quality": 10,
                    "sigma_pos_m": radar["sigma_pos"],
                    "sigma_vel_mps": radar["sigma_vel"],
                    "is_clutter": False,
                })

            clutter_count = rng.poisson(radar["clutter_per_scan"])

            for clutter_index in range(clutter_count):
                records.append({
                    "time": round(scan_time, 3),
                    "arrival_time": round(scan_time + radar["delay_s"], 3),
                    "sensor": radar["name"],
                    "local_track_id":
                        f"{radar['name']}-CL-{int(scan_time)}-{clutter_index}",
                    "callsign_true": "CLUTTER",

                    "x": rng.uniform(gt_df["x"].min(), gt_df["x"].max()),
                    "y": rng.uniform(gt_df["y"].min(), gt_df["y"].max()),
                    "z": rng.uniform(gt_df["z"].min(), gt_df["z"].max()),

                    "vx": rng.normal(0, 15),
                    "vy": rng.normal(0, 15),
                    "vz": rng.normal(0, 5),

                    "track_quality": 2,
                    "sigma_pos_m": 1000.0,
                    "sigma_vel_mps": 15.0,
                    "is_clutter": True,
                })

            scan_time += radar["revisit_s"]

    return (
        pd.DataFrame(records)
        .sort_values("time")
        .reset_index(drop=True)
    )


def main():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)

    required_columns = {
        "time", "x", "y", "z",
        "vx", "vy", "vz", "callsign"
    }

    missing = required_columns - set(gt_df.columns)

    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    radar_df = generate_radar_data(gt_df)
    radar_df.to_csv(OUTPUT_RADAR_CSV, index=False)

    print(f"Hedef sayısı: {gt_df['callsign'].nunique()}")
    print(f"Radar ölçüm sayısı: {len(radar_df)}")
    print(f"Gerçek ölçüm: {(radar_df['is_clutter'] == False).sum()}")
    print(f"Clutter: {(radar_df['is_clutter'] == True).sum()}")
    print(f"Kaydedildi: {OUTPUT_RADAR_CSV}")


if __name__ == "__main__":
    main()
