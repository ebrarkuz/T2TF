import numpy as np
from scipy.optimize import linear_sum_assignment


def interpolate_gt_state(sub, t):
    if len(sub) < 2 or t < sub["time"].iloc[0] or t > sub["time"].iloc[-1]:
        return None

    x = np.interp(t, sub["time"], sub["x"])
    y = np.interp(t, sub["time"], sub["y"])
    vx = np.interp(t, sub["time"], sub["vx"])
    vy = np.interp(t, sub["time"], sub["vy"])

    return {
        "track_id": sub["callsign"].iloc[0] if "callsign" in sub.columns else sub["target"].iloc[0],
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
    }


def frame_matches(fused_frame, gt_states, max_distance=00.0):
    if fused_frame.empty or len(gt_states) == 0:
        return [], list(range(len(fused_frame))), list(range(len(gt_states)))

    fused_pos = fused_frame[["x", "y"]].to_numpy()
    gt_pos = np.array([[s["x"], s["y"]] for s in gt_states])
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


def compute_tracking_metrics(gt_df, fused_df, max_match_distance=400.0):
    if fused_df.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mota": 0.0,
            "id_switches": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_gt": 0,
            "rmse_pos_m": 0.0,
            "rmse_vel_mps": 0.0,
            "nees": 0.0,
            "nees_ideal": 4.0,
        }

    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    gt_tracks = {cs: sub.sort_values("time") for cs, sub in gt_df.groupby(gt_key)}

    total_tp = total_fp = total_fn = total_gt = 0
    total_tp_id = 0   # ID-tutarlı TP: aynı hedefi ardışık karelerde aynı track ID'siyle bulma
    id_switches = 0
    prev_assign = {}
    matched_rows = []

    for t in sorted(fused_df["time"].unique()):
        fused_frame = fused_df[fused_df["time"] == t].reset_index(drop=True)
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

    # detection_precision: Sadece uzaysal konum doğruluğu (ID bilgisini görmezden gelir)
    detection_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    # id_precision: Konum VE kimlik tutarlılığını birlikte ölçer (ID switch'leri FP gibi cezalandırır)
    id_precision = total_tp_id / (total_tp_id + total_fp + id_switches) if (total_tp_id + total_fp + id_switches) > 0 else 0.0
    precision = detection_precision  # Geriye dönük uyumluluk için korunur
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    id_f1 = 2 * id_precision * recall / (id_precision + recall) if id_precision + recall > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)

    rmse_pos = rmse_vel = nees_mean = np.nan
    if matched_rows:
        errors_pos = []
        errors_vel = []
        nees_vals = []

        for fused_row, gt_state in matched_rows:
            dx = fused_row["x"] - gt_state["x"]
            dy = fused_row["y"] - gt_state["y"]
            dvx = fused_row["vx"] - gt_state["vx"]
            dvy = fused_row["vy"] - gt_state["vy"]
            errors_pos.append(dx**2 + dy**2)
            errors_vel.append(dvx**2 + dvy**2)

            cov = np.diag([
                fused_row.get("sigma_x_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vx_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_y_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vy_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
            ])
            try:
                invP = np.linalg.inv(cov)
                e = np.array([dx, dvx, dy, dvy])
                nees_vals.append(float(e.T @ invP @ e))
            except np.linalg.LinAlgError:
                pass

        rmse_pos = float(np.sqrt(np.mean(errors_pos))) if errors_pos else np.nan
        rmse_vel = float(np.sqrt(np.mean(errors_vel))) if errors_vel else np.nan
        nees_mean = float(np.mean(nees_vals)) if nees_vals else np.nan

    return {
        "precision": precision,           # Uzaysal detection precision (ID agnostik)
        "id_precision": id_precision,      # ID-tutarlı precision (daha katı)
        "recall": recall,
        "f1_score": f1_score,              # detection_precision bazlı
        "id_f1": id_f1,                    # id_precision bazlı (daha katı)
        "mota": mota,
        "id_switches": id_switches,
        "total_tp": total_tp,
        "total_tp_id": total_tp_id,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "rmse_pos_m": rmse_pos,
        "rmse_vel_mps": rmse_vel,
        "nees": nees_mean,
        "nees_ideal": 4.0,
    }


def compute_target_specific_metrics(gt_df, fused_df, target_callsign, max_match_distance=400.0):
    """
    Belirli bir hedef (target) için metrikleri hesapla.
    
    Parametreler:
        gt_df: Ground truth DataFrame
        fused_df: Fused track DataFrame
        target_callsign: Hedefin callsign'ı (örn: 'HEDEF_1', 'HEDEF_2', 'HEDEF_3')
        max_match_distance: Eşleştirme için maksimum mesafe (metre)
    
    Dönüş: Metrikler dictionary
    """
    
    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    
    # Hedefin ground truth kaydını al
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
            "rmse_vel_mps": np.nan,
            "nees": np.nan,
        }
    
    gt_target = gt_target.sort_values("time")
    
    total_tp = total_fp = total_fn = total_gt = 0
    id_switches = 0
    prev_assign = {}
    matched_rows = []
    
    # Her zaman adımında eşleştir
    for t in sorted(fused_df["time"].unique()):
        fused_frame = fused_df[fused_df["time"] == t].reset_index(drop=True)
        
        # Bu zaman adımında hedefin gerçek konumunu interpolate et
        if t < gt_target["time"].iloc[0] or t > gt_target["time"].iloc[-1]:
            continue
        
        gt_state = interpolate_gt_state(gt_target, t)
        if gt_state is None:
            # GT verisi bu an için interpolate edilemiyorsa atla (FP yazmak hatalıdır)
            continue
        
        gt_states = [gt_state]
        total_gt += 1

        # Uzaysal ön-filtre: yalnızca bu hedefe yakın detections değerlendirilir.
        # Aksi hâlde başka hedeflerin/clutter'ların track'leri FP olarak sayılır.
        dist_arr = np.sqrt(
            (fused_frame["x"].to_numpy() - gt_state["x"]) ** 2 +
            (fused_frame["y"].to_numpy() - gt_state["y"]) ** 2
        )
        fused_candidates = fused_frame[dist_arr <= max_match_distance].reset_index(drop=True)

        if fused_candidates.empty:
            total_fn += 1
            continue

        matches, unmatched_fused, unmatched_gt = frame_matches(
            fused_candidates, gt_states, max_distance=max_match_distance
        )

        total_tp += len(matches)
        total_fp += len(unmatched_fused)  # Hedefe yakın ama eşleşemeyen detections
        total_fn += len(unmatched_gt)
        
        # ID switch kontrol (fused_candidates üzerinden)
        for ri, ci in matches:
            fused_row = fused_candidates.iloc[ri]
            fused_id = fused_row["global_track_id"]

            if fused_id in prev_assign and prev_assign[fused_id] != target_callsign:
                id_switches += 1
            prev_assign[fused_id] = target_callsign
            matched_rows.append((fused_row, gt_state))
    
    # precision: Hedefe yakın detections içinde doğru eşleşme oranı
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    # coverage: Hedefin aktif olduğu zaman adımlarında kaçında bir detection bulundu
    coverage = total_tp / total_gt if total_gt > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)
    
    rmse_pos = rmse_vel = nees_mean = np.nan
    if matched_rows:
        errors_pos = []
        errors_vel = []
        nees_vals = []
        
        for fused_row, gt_state in matched_rows:
            dx = fused_row["x"] - gt_state["x"]
            dy = fused_row["y"] - gt_state["y"]
            dvx = fused_row["vx"] - gt_state["vx"]
            dvy = fused_row["vy"] - gt_state["vy"]
            errors_pos.append(dx**2 + dy**2)
            errors_vel.append(dvx**2 + dvy**2)
            
            cov = np.diag([
                fused_row.get("sigma_x_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vx_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
                fused_row.get("sigma_y_m", fused_row.get("pos_sigma_m", 1.0)) ** 2,
                fused_row.get("sigma_vy_mps", fused_row.get("vel_sigma_mps", 1.0)) ** 2,
            ])
            try:
                invP = np.linalg.inv(cov)
                e = np.array([dx, dvx, dy, dvy])
                nees_vals.append(float(e.T @ invP @ e))
            except np.linalg.LinAlgError:
                pass
        
        rmse_pos = float(np.sqrt(np.mean(errors_pos))) if errors_pos else np.nan
        rmse_vel = float(np.sqrt(np.mean(errors_vel))) if errors_vel else np.nan
        nees_mean = float(np.mean(nees_vals)) if nees_vals else np.nan
    
    return {
        "callsign": target_callsign,
        "precision": precision,    # Hedefe yakın FP'lere karşı TP oranı
        "recall": recall,
        "f1_score": f1_score,
        "coverage": coverage,      # Hedefin kaç zaman adımında bulunabildiği (0–1)
        "mota": mota,
        "id_switches": id_switches,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "rmse_pos_m": rmse_pos,
        "rmse_vel_mps": rmse_vel,
        "nees": nees_mean,
    }
