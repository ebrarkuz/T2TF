"""Read-only association diagnostics for one labelled simulation target.

This module does not alter fusion_advanced.py. It runs the existing FusionCenter
and observes accepted updates/track creations while recording pre-association
NIS, adaptive gate and velocity residuals.
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd

import fusion_advanced as advanced


H_MEASUREMENT = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
], dtype=float)


def _measurement_from_row(row):
    state = np.array([
        row["x"], row["vx"], row["y"], row["vy"],
        row.get("z", 0.0), row.get("vz", 0.0),
    ], dtype=float).reshape(6, 1)
    return {
        "state": state,
        "cov": advanced.measurement_cov_from_row(row),
        "tq": row.get("track_quality", advanced.TQ_MAX),
        "src": (row["sensor"], row["local_track_id"]),
        "label": str(row.get("callsign_true", "UNKNOWN")),
    }


def _candidate_metrics(track, measurement):
    innovation = H_MEASUREMENT @ track.state - measurement["state"]
    innovation_cov = H_MEASUREMENT @ track.cov @ H_MEASUREMENT.T + measurement["cov"]
    try:
        solved = np.linalg.solve(innovation_cov, innovation)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(innovation_cov) @ innovation
    d2 = float((innovation.T @ solved).item())
    _, logdet = np.linalg.slogdet(innovation_cov)
    position_innovation = float(np.linalg.norm(innovation[[0, 2, 4], 0]))
    velocity_innovation = float(np.linalg.norm(innovation[[1, 3, 5], 0]))
    return {
        "predicted_global_track_id": track.id,
        "position_innovation_m": position_innovation,
        "velocity_innovation_mps": velocity_innovation,
        "d2": d2,
        "base_gate": advanced.BASE_GATE_CHI2_6DOF,
        "effective_gate": track.get_effective_gate(),
        "q_scale": track.q_scale,
        "maneuver_score": track.maneuver_score,
        "last_nis": track.last_nis,
        "assignment_cost": d2 + max(0.0, float(logdet)),
    }


def diagnose(sensor_csv, target="HEDEF_4", output_csv="hedef4_association_debug.csv"):
    sensor_df = pd.read_csv(sensor_csv).sort_values("time")
    if "callsign_true" not in sensor_df:
        raise ValueError("Debug için callsign_true kolonu gerekli")

    center = advanced.FusionCenter(use_ci=True)
    advanced.GlobalTrack._cnt = 0
    records = []
    track_labels = defaultdict(set)
    deleted_target_tracks = set()
    deleted_target_previous_status = {}
    accepted_events = []
    creation_events = []
    current_labels = {}

    original_update = advanced.GlobalTrack.update
    original_init = advanced.GlobalTrack.__init__

    def traced_update(track, meas_state, meas_cov, tq, src):
        original_update(track, meas_state, meas_cov, tq, src)
        label = current_labels.get(src, "UNKNOWN")
        track_labels[track.id].add(label)
        accepted_events.append((src, track.id, label))

    def traced_init(track, t, state, cov, tq, src, use_ci=True):
        original_init(track, t, state, cov, tq, src, use_ci=use_ci)
        label = current_labels.get(src, "UNKNOWN")
        track_labels[track.id].add(label)
        creation_events.append((src, track.id, label))

    advanced.GlobalTrack.update = traced_update
    advanced.GlobalTrack.__init__ = traced_init
    try:
        for time_value, group in sensor_df.groupby("time", sort=True):
            measurements = [_measurement_from_row(row) for _, row in group.iterrows()]
            current_labels = {measurement["src"]: measurement["label"] for measurement in measurements}

            # Predict once here for diagnostics. FusionCenter sees dt=0 and does
            # not propagate a second time, so production behaviour is unchanged.
            status_before_prediction = {track.id: track.status for track in center.tracks}
            for track in center.tracks:
                track.propagate(time_value)
            for track in center.tracks:
                if track.status == "DELETED" and target in track_labels[track.id]:
                    deleted_target_tracks.add(track.id)
                    deleted_target_previous_status.setdefault(
                        track.id, status_before_prediction.get(track.id, "UNKNOWN")
                    )

            active_tracks = [track for track in center.tracks if track.status != "DELETED"]
            active_by_id = {track.id: track for track in active_tracks}
            pre_rows = []
            for measurement in measurements:
                if measurement["label"] != target:
                    continue
                mapped_id = center.src_map.get(measurement["src"])
                source_map_found = mapped_id in active_by_id
                candidates = [
                    _candidate_metrics(track, measurement) for track in active_tracks
                ]
                mapped_candidate = next(
                    (candidate for candidate in candidates
                     if candidate["predicted_global_track_id"] == mapped_id),
                    None,
                )
                best_candidate = mapped_candidate or (
                    min(candidates, key=lambda item: item["assignment_cost"])
                    if candidates else None
                )
                pre_rows.append({
                    "time": float(time_value),
                    "sensor": measurement["src"][0],
                    "local_track_id": measurement["src"][1],
                    "src": measurement["src"],
                    "source_map_found": source_map_found,
                    "active_track_count": len(active_tracks),
                    "_candidates_by_id": {
                        candidate["predicted_global_track_id"]: candidate
                        for candidate in candidates
                    },
                    **(best_candidate or {
                        "predicted_global_track_id": None,
                        "position_innovation_m": np.nan,
                        "velocity_innovation_mps": np.nan,
                        "d2": np.nan,
                        "base_gate": advanced.BASE_GATE_CHI2_6DOF,
                        "effective_gate": advanced.BASE_GATE_CHI2_6DOF,
                        "q_scale": np.nan,
                        "maneuver_score": np.nan,
                        "last_nis": np.nan,
                        "assignment_cost": np.nan,
                    }),
                })

            accepted_events.clear()
            creation_events.clear()
            center.process_batch(time_value, measurements)
            accepted_by_src = {src: track_id for src, track_id, label in accepted_events if label == target}
            created_by_src = {src: track_id for src, track_id, label in creation_events if label == target}

            for row in pre_rows:
                src = row.pop("src")
                candidates_by_id = row.pop("_candidates_by_id")
                accepted_track = accepted_by_src.get(src)
                created_track = created_by_src.get(src)
                accepted = accepted_track is not None
                reasons = []
                if accepted:
                    accepted_metrics = candidates_by_id.get(accepted_track)
                    if accepted_metrics is not None:
                        row.update(accepted_metrics)
                    result = "ACCEPTED"
                else:
                    if not row["source_map_found"]:
                        reasons.append("SOURCE_MAP_NOT_FOUND")
                    if row["active_track_count"] == 0:
                        reasons.append("NO_ACTIVE_TRACK")
                    elif not math.isfinite(row["d2"]) or row["d2"] >= row["effective_gate"]:
                        reasons.append("MAHALANOBIS_GATE_FAIL")
                    elif row["velocity_innovation_mps"] > advanced.MAX_ASSOC_VEL_DIFF_MPS:
                        reasons.append("VELOCITY_GATE_FAIL")
                    else:
                        reasons.append("HUNGARIAN_NOT_SELECTED")
                    if created_track is not None:
                        reasons.append("NEW_TRACK_CREATED")
                    result = ";".join(reasons)
                row.update({
                    "accepted_global_track_id": accepted_track,
                    "created_global_track_id": created_track,
                    "measurement_accepted": accepted,
                    "result": result,
                })
                records.append(row)
    finally:
        advanced.GlobalTrack.update = original_update
        advanced.GlobalTrack.__init__ = original_init

    debug_df = pd.DataFrame(records)
    debug_df.to_csv(output_csv, index=False)
    accepted_count = int(debug_df["measurement_accepted"].sum()) if len(debug_df) else 0
    gate_rejected = int(debug_df["result"].str.contains("MAHALANOBIS_GATE_FAIL").sum()) if len(debug_df) else 0
    velocity_rejected = int(debug_df["result"].str.contains("VELOCITY_GATE_FAIL").sum()) if len(debug_df) else 0
    hungarian_rejected = int(debug_df["result"].str.contains("HUNGARIAN_NOT_SELECTED").sum()) if len(debug_df) else 0
    creations = int(debug_df["created_global_track_id"].notna().sum()) if len(debug_df) else 0
    unique_ids = set(debug_df["accepted_global_track_id"].dropna()) | set(debug_df["created_global_track_id"].dropna())
    summary = {
        "total_measurements": len(debug_df),
        "accepted": accepted_count,
        "rejected": len(debug_df) - accepted_count,
        "gate_inside": int(((debug_df["d2"] < debug_df["effective_gate"]) & debug_df["d2"].notna()).sum()) if len(debug_df) else 0,
        "gate_rejected": gate_rejected,
        "velocity_rejected": velocity_rejected,
        "hungarian_rejected": hungarian_rejected,
        "new_tracks": creations,
        "track_recreations": max(0, creations - 1),
        "track_deletions": len(deleted_target_tracks),
        "confirmed_track_deletions": sum(
            status == "CONFIRMED" for status in deleted_target_previous_status.values()
        ),
        "tentative_track_deletions": sum(
            status == "TENTATIVE" for status in deleted_target_previous_status.values()
        ),
        "unique_global_track_ids": len(unique_ids),
        "max_q_scale": float(debug_df["q_scale"].max()) if len(debug_df) else np.nan,
        "max_maneuver_score": float(debug_df["maneuver_score"].max()) if len(debug_df) else np.nan,
        "max_effective_gate": float(debug_df["effective_gate"].max()) if len(debug_df) else np.nan,
    }
    return debug_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-csv", default="radar_sensor_tracks_gercekci.csv")
    parser.add_argument("--target", default="HEDEF_4")
    parser.add_argument("--output-csv", default="hedef4_association_debug.csv")
    args = parser.parse_args()
    debug_df, summary = diagnose(args.sensor_csv, args.target, args.output_csv)
    print("HEDEF_4 DIAGNOSIS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    columns = [
        "time", "d2", "effective_gate", "velocity_innovation_mps",
        "q_scale", "maneuver_score", "result",
    ]
    print("\nTime-ordered sample:")
    print(debug_df[columns].head(30).to_string(index=False))
    print(f"\nDebug CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
