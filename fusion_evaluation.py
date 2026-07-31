import numpy as np
from scipy.optimize import linear_sum_assignment


def interpolate_gt_state(sub, t):
    if len(sub) < 2 or t < sub["time"].iloc[0] or t > sub["time"].iloc[-1]:
        return None

    x = np.interp(t, sub["time"], sub["x"])
    y = np.interp(t, sub["time"], sub["y"])
    z = np.interp(t, sub["time"], sub["z"]) if "z" in sub.columns else 0.0
    vx = np.interp(t, sub["time"], sub["vx"])
    vy = np.interp(t, sub["time"], sub["vy"])
    vz = np.interp(t, sub["time"], sub["vz"]) if "vz" in sub.columns else 0.0

    return {
        "track_id": sub["callsign"].iloc[0] if "callsign" in sub.columns else sub["target"].iloc[0],
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }


def frame_matches(fused_frame, gt_states, max_distance=0.0):
    if fused_frame.empty or len(gt_states) == 0:
        return [], list(range(len(fused_frame))), list(range(len(gt_states)))

    fused_pos = fused_frame[["x", "y", "z"]].to_numpy() if "z" in fused_frame.columns else np.column_stack([
        fused_frame["x"].to_numpy(),
        fused_frame["y"].to_numpy(),
        np.zeros(len(fused_frame)),
    ])
    gt_pos = np.array([[s["x"], s["y"], s.get("z", 0.0)] for s in gt_states])
    cost = np.linalg.norm(fused_pos[:, None, :] - gt_pos[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    matched_fused = set()
    matched_gt = set()

    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] <= max_distance:
            matches.append((ri, ci))
            matched_fused.add(ri)
            matched_gt.add(ci)

    unmatched_fused = [i for i in range(len(fused_frame)) if i not in matched_fused]
    unmatched_gt = [i for i in range(len(gt_states)) if i not in matched_gt]
    return matches, unmatched_fused, unmatched_gt


def compute_tracking_metrics(
    gt_df,
    fused_df,
    max_match_distance=500.0,
    evaluation_times=None,
):
    """Compute metrics on fused timestamps or on an explicit sensor timeline.

    Supplying ``evaluation_times`` prevents a sparse algorithm from improving
    its recall simply by omitting timestamps at which it produced no track.
    """
    if fused_df.empty:
        if evaluation_times is not None:
            total_gt = 0
            gt_key = "callsign" if "callsign" in gt_df.columns else "target"
            for t in sorted(set(evaluation_times)):
                total_gt += sum(
                    interpolate_gt_state(sub.sort_values("time"), t) is not None
                    for _, sub in gt_df.groupby(gt_key)
                )
        else:
            total_gt = 0
        return {
            "precision": 0.0,
            "id_precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "id_f1": 0.0,
            "mota": 0.0,
            "id_switches": 0,
            "total_tp": 0,
            "total_tp_id": 0,
            "total_fp": 0,
            "total_fn": total_gt,
            "total_gt": total_gt,
            "rmse_pos_m": np.nan,
            "rmse_x_m": np.nan,
            "rmse_y_m": np.nan,
            "rmse_z_m": np.nan,
            "rmse_vel_mps": np.nan,
            "rmse_vx_mps": np.nan,
            "rmse_vy_mps": np.nan,
            "rmse_vz_mps": np.nan,
            "nees": np.nan,
            "nees_ideal": 6.0,
        }

    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    gt_tracks = {cs: sub.sort_values("time") for cs, sub in gt_df.groupby(gt_key)}

    total_tp = total_fp = total_fn = total_gt = 0
    total_tp_id = 0   # ID-tutarlı TP: aynı hedefi ardışık karelerde aynı track ID'siyle bulma
    id_switches = 0
    prev_assign = {}
    matched_rows = []
    fused_frames = {
        t: frame.reset_index(drop=True)
        for t, frame in fused_df.groupby("time", sort=False)
    }
    empty_frame = fused_df.iloc[0:0].copy()

    timeline = (
        sorted(set(evaluation_times))
        if evaluation_times is not None
        else sorted(fused_df["time"].unique())
    )
    for t in timeline:
        fused_frame = fused_frames.get(t, empty_frame)
        gt_states = [
            state
            for _, sub in gt_tracks.items()
            if (state := interpolate_gt_state(sub, t)) is not None
        ]

        if not gt_states:
            total_fp += len(fused_frame)
            for fid in fused_frame["global_track_id"].unique():
                prev_assign[fid] = None
            continue

        matches, unmatched_fused, unmatched_gt = frame_matches(
            fused_frame, gt_states, max_distance=max_match_distance
        )
        total_tp += len(matches)
        total_fp += len(unmatched_fused)
        total_fn += len(unmatched_gt)
        total_gt += len(gt_states)

        for ri, ci in matches:
            fused_row = fused_frame.iloc[ri]
            gt_state = gt_states[ci]
            fused_id = fused_row["global_track_id"]
            gt_id = gt_state["track_id"]
            is_id_switch = (
                fused_id in prev_assign
                and prev_assign[fused_id] is not None
                and prev_assign[fused_id] != gt_id
            )
            if is_id_switch:
                id_switches += 1
            else:
                total_tp_id += 1  # Konum VE kimlik doğru
            prev_assign[fused_id] = gt_id
            matched_rows.append((fused_row, gt_state))

        for ri in unmatched_fused:
            fused_id = fused_frame.iloc[ri]["global_track_id"]
            prev_assign[fused_id] = None

    detection_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    id_precision = total_tp_id / (total_tp_id + total_fp + id_switches) if (total_tp_id + total_fp + id_switches) > 0 else 0.0
    precision = detection_precision  
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    id_f1 = 2 * id_precision * recall / (id_precision + recall) if id_precision + recall > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)

    rmse_pos = rmse_x = rmse_y = rmse_z = np.nan
    rmse_vel = rmse_vx = rmse_vy = rmse_vz = nees_mean = np.nan
    
    if matched_rows:
        errors_pos, errors_x, errors_y, errors_z = [], [], [], []
        errors_vel, errors_vx, errors_vy, errors_vz = [], [], [], []
        nees_vals = []

        for fused_row, gt_state in matched_rows:
            dx = fused_row["x"] - gt_state["x"]
            dy = fused_row["y"] - gt_state["y"]
            dz = fused_row.get("z", 0.0) - gt_state.get("z", 0.0)
            dvx = fused_row["vx"] - gt_state["vx"]
            dvy = fused_row["vy"] - gt_state["vy"]
            dvz = fused_row.get("vz", 0.0) - gt_state.get("vz", 0.0)
            
            # Toplam RMSE listeleri
            errors_pos.append(dx**2 + dy**2 + dz**2)
            errors_vel.append(dvx**2 + dvy**2 + dvz**2)
            
            # Eksen bazlı RMSE listeleri
            errors_x.append(dx**2)
            errors_y.append(dy**2)
            errors_z.append(dz**2)
            errors_vx.append(dvx**2)
            errors_vy.append(dvy**2)
            errors_vz.append(dvz**2)

            cov = np.diag([
                fused_row.get("sigma_x_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vx_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_y_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vy_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_z_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vz_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
            ])
            try:
                invP = np.linalg.inv(cov)
                e = np.array([dx, dvx, dy, dvy, dz, dvz])
                nees_vals.append(float(e.T @ invP @ e))
            except np.linalg.LinAlgError:
                pass

        rmse_pos = float(np.sqrt(np.mean(errors_pos))) if errors_pos else np.nan
        rmse_x = float(np.sqrt(np.mean(errors_x))) if errors_x else np.nan
        rmse_y = float(np.sqrt(np.mean(errors_y))) if errors_y else np.nan
        rmse_z = float(np.sqrt(np.mean(errors_z))) if errors_z else np.nan
        
        rmse_vel = float(np.sqrt(np.mean(errors_vel))) if errors_vel else np.nan
        rmse_vx = float(np.sqrt(np.mean(errors_vx))) if errors_vx else np.nan
        rmse_vy = float(np.sqrt(np.mean(errors_vy))) if errors_vy else np.nan
        rmse_vz = float(np.sqrt(np.mean(errors_vz))) if errors_vz else np.nan
        
        nees_mean = float(np.mean(nees_vals)) if nees_vals else np.nan

    return {
        "precision": precision,           
        "id_precision": id_precision,      
        "recall": recall,
        "f1_score": f1_score,              
        "id_f1": id_f1,                    
        "mota": mota,
        "id_switches": id_switches,
        "total_tp": total_tp,
        "total_tp_id": total_tp_id,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "rmse_pos_m": rmse_pos,
        "rmse_x_m": rmse_x,
        "rmse_y_m": rmse_y,
        "rmse_z_m": rmse_z,
        "rmse_vel_mps": rmse_vel,
        "rmse_vx_mps": rmse_vx,
        "rmse_vy_mps": rmse_vy,
        "rmse_vz_mps": rmse_vz,
        "nees": nees_mean,
        "nees_ideal": 6.0,
    }


def compute_target_specific_metrics(
    gt_df,
    fused_df,
    target_callsign,
    max_match_distance=500.0,
    evaluation_times=None,
    prepared_fused_frames=None,
):
    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    gt_target = gt_df[gt_df[gt_key] == target_callsign]
    
    if gt_target.empty:
        return {
            "callsign": target_callsign,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mota": 0.0,
            "id_switches": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_gt": 0,
            "rmse_pos_m": np.nan,
            "rmse_x_m": np.nan,
            "rmse_y_m": np.nan,
            "rmse_z_m": np.nan,
            "rmse_vel_mps": np.nan,
            "rmse_vx_mps": np.nan,
            "rmse_vy_mps": np.nan,
            "rmse_vz_mps": np.nan,
            "nees": np.nan,
        }
    
    gt_target = gt_target.sort_values("time")
    
    total_tp = total_fp = total_fn = total_gt = 0
    id_switches = 0
    prev_assign = {}
    matched_rows = []
    fused_frames = prepared_fused_frames
    if fused_frames is None:
        fused_frames = {
            t: frame.reset_index(drop=True)
            for t, frame in fused_df.groupby("time", sort=False)
        }
    empty_frame = fused_df.iloc[0:0].copy()
    
    timeline = (
        sorted(set(evaluation_times))
        if evaluation_times is not None
        else sorted(fused_df["time"].unique())
    )
    for t in timeline:
        fused_frame = fused_frames.get(t, empty_frame)
        
        if t < gt_target["time"].iloc[0] or t > gt_target["time"].iloc[-1]:
            continue
        
        gt_state = interpolate_gt_state(gt_target, t)
        if gt_state is None:
            continue
        
        gt_states = [gt_state]
        total_gt += 1

        dist_arr = np.sqrt(
            (fused_frame["x"].to_numpy() - gt_state["x"]) ** 2 +
            (fused_frame["y"].to_numpy() - gt_state["y"]) ** 2 +
            ((fused_frame["z"].to_numpy() if "z" in fused_frame.columns else np.zeros(len(fused_frame))) - gt_state.get("z", 0.0)) ** 2
        )
        fused_candidates = fused_frame[dist_arr <= max_match_distance].reset_index(drop=True)

        if fused_candidates.empty:
            total_fn += 1
            continue

        matches, unmatched_fused, unmatched_gt = frame_matches(
            fused_candidates, gt_states, max_distance=max_match_distance
        )

        total_tp += len(matches)
        total_fp += len(unmatched_fused)  
        total_fn += len(unmatched_gt)
        
        for ri, ci in matches:
            fused_row = fused_candidates.iloc[ri]
            fused_id = fused_row["global_track_id"]

            if fused_id in prev_assign and prev_assign[fused_id] != target_callsign:
                id_switches += 1
            prev_assign[fused_id] = target_callsign
            matched_rows.append((fused_row, gt_state))
    
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    coverage = total_tp / total_gt if total_gt > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)
    
    rmse_pos = rmse_x = rmse_y = rmse_z = np.nan
    rmse_vel = rmse_vx = rmse_vy = rmse_vz = nees_mean = np.nan
    
    if matched_rows:
        errors_pos, errors_x, errors_y, errors_z = [], [], [], []
        errors_vel, errors_vx, errors_vy, errors_vz = [], [], [], []
        nees_vals = []
        
        for fused_row, gt_state in matched_rows:
            dx = fused_row["x"] - gt_state["x"]
            dy = fused_row["y"] - gt_state["y"]
            dz = fused_row.get("z", 0.0) - gt_state.get("z", 0.0)
            dvx = fused_row["vx"] - gt_state["vx"]
            dvy = fused_row["vy"] - gt_state["vy"]
            dvz = fused_row.get("vz", 0.0) - gt_state.get("vz", 0.0)
            
            # Toplam RMSE
            errors_pos.append(dx**2 + dy**2 + dz**2)
            errors_vel.append(dvx**2 + dvy**2 + dvz**2)
            
            # Eksen bazlı RMSE
            errors_x.append(dx**2)
            errors_y.append(dy**2)
            errors_z.append(dz**2)
            errors_vx.append(dvx**2)
            errors_vy.append(dvy**2)
            errors_vz.append(dvz**2)
            
            cov = np.diag([
                fused_row.get("sigma_x_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vx_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_y_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vy_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_z_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vz_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
            ])
            try:
                invP = np.linalg.inv(cov)
                e = np.array([dx, dvx, dy, dvy, dz, dvz])
                nees_vals.append(float(e.T @ invP @ e))
            except np.linalg.LinAlgError:
                pass
        
        rmse_pos = float(np.sqrt(np.mean(errors_pos))) if errors_pos else np.nan
        rmse_x = float(np.sqrt(np.mean(errors_x))) if errors_x else np.nan
        rmse_y = float(np.sqrt(np.mean(errors_y))) if errors_y else np.nan
        rmse_z = float(np.sqrt(np.mean(errors_z))) if errors_z else np.nan
        
        rmse_vel = float(np.sqrt(np.mean(errors_vel))) if errors_vel else np.nan
        rmse_vx = float(np.sqrt(np.mean(errors_vx))) if errors_vx else np.nan
        rmse_vy = float(np.sqrt(np.mean(errors_vy))) if errors_vy else np.nan
        rmse_vz = float(np.sqrt(np.mean(errors_vz))) if errors_vz else np.nan
        
        nees_mean = float(np.mean(nees_vals)) if nees_vals else np.nan
    
    return {
        "callsign": target_callsign,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "coverage": coverage,
        "mota": mota,
        "id_switches": id_switches,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "rmse_pos_m": rmse_pos,
        "rmse_x_m": rmse_x,
        "rmse_y_m": rmse_y,
        "rmse_z_m": rmse_z,
        "rmse_vel_mps": rmse_vel,
        "rmse_vx_mps": rmse_vx,
        "rmse_vy_mps": rmse_vy,
        "rmse_vz_mps": rmse_vz,
        "nees": nees_mean,
    }
