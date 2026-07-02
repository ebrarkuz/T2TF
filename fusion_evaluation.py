import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

GT_CSV_CANDIDATES = ["ground_truth_adsb4.csv", "ground_truth_adsb.csv"]
OUTPUT_FUSED_CSV = "fused_tracks.csv"
OUTPUT_BASIC_CSV = "basic_fusion_sonuclari.csv"
SENSOR_CSV = "radar_sensor_tracks.csv"


def load_data():
    gt_path = next((p for p in GT_CSV_CANDIDATES if os.path.exists(p)), None)
    if gt_path is None:
        raise FileNotFoundError(
            "Ground truth CSV dosyasi bulunamadi. Lutfen ground_truth_adsb.csv veya ground_truth_adsb4.csv adli dosyayi klasore koyun."
        )
    gt_df = pd.read_csv(gt_path)
    sensor_df = pd.read_csv(SENSOR_CSV)
    
    # Füzyon sonuçlarını yükle (Hata yönetimi ile)
    try:
        fused_df = pd.read_csv(OUTPUT_FUSED_CSV)
    except FileNotFoundError:
        print(f"UYARI: {OUTPUT_FUSED_CSV} bulunamadı. Boş veri çerçevesi oluşturuluyor.")
        fused_df = pd.DataFrame()

    try:
        basic_df = pd.read_csv(OUTPUT_BASIC_CSV)
    except FileNotFoundError:
        print(f"UYARI: {OUTPUT_BASIC_CSV} bulunamadı. Boş veri çerçevesi oluşturuluyor.")
        basic_df = pd.DataFrame()

    return gt_df, sensor_df, fused_df, basic_df


def interpolate_gt_state(sub, t):
    if len(sub) < 2 or t < sub["time"].iloc[0] or t > sub["time"].iloc[-1]:
        return None
    x = np.interp(t, sub["time"], sub["x"])
    y = np.interp(t, sub["time"], sub["y"])
    vx = np.interp(t, sub["time"], sub["vx"])
    vy = np.interp(t, sub["time"], sub["vy"])
    return {"track_id": sub["callsign"].iloc[0] if "callsign" in sub.columns else sub["target"].iloc[0], "x": x, "y": y, "vx": vx, "vy": vy}


def frame_matches(fused_frame, gt_states, max_distance=1000.0):
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


def compute_tracking_metrics(gt_df, fused_df, max_match_distance=1000.0):
    if fused_df.empty:
        return {k: 0.0 for k in ["precision", "recall", "f1_score", "mota", "id_switches", "total_tp", "total_fp", "total_fn", "total_gt", "rmse_pos_m", "rmse_vel_mps", "nees"]} | {"nees_ideal": 4.0}

    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    gt_tracks = {cs: sub.sort_values("time") for cs, sub in gt_df.groupby(gt_key)}
    total_tp = total_fp = total_fn = total_gt = 0
    id_switches = 0
    prev_assign = {}
    matched_rows = []

    for t in sorted(fused_df["time"].unique()):
        fused_frame = fused_df[fused_df["time"] == t].reset_index(drop=True)
        gt_states = []
        for cs, sub in gt_tracks.items():
            state = interpolate_gt_state(sub, t)
            if state is not None:
                gt_states.append(state)
        if not gt_states:
            total_fp += len(fused_frame)
            for fid in fused_frame["global_track_id"].unique():
                prev_assign[fid] = None
            continue

        matches, unmatched_fused, unmatched_gt = frame_matches(fused_frame, gt_states, max_distance=max_match_distance)
        total_tp += len(matches)
        total_fp += len(unmatched_fused)
        total_fn += len(unmatched_gt)
        total_gt += len(gt_states)

        for ri, ci in matches:
            fused_row = fused_frame.iloc[ri]
            gt_state = gt_states[ci]
            fused_id = fused_row["global_track_id"]
            gt_id = gt_state["track_id"]
            if fused_id in prev_assign and prev_assign[fused_id] is not None and prev_assign[fused_id] != gt_id:
                id_switches += 1
            prev_assign[fused_id] = gt_id
            matched_rows.append((fused_row, gt_state))

        for ri in unmatched_fused:
            fused_id = fused_frame.iloc[ri]["global_track_id"]
            prev_assign[fused_id] = None

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(total_gt, 1)

    rmse_pos = rmse_vel = nees_mean = np.nan
    if matched_rows:
        errors_pos, errors_vel, nees_vals = [], [], []
        for fused_row, gt_state in matched_rows:
            dx = fused_row["x"] - gt_state["x"]
            dy = fused_row["y"] - gt_state["y"]
            dvx = fused_row["vx"] - gt_state["vx"]
            dvy = fused_row["vy"] - gt_state["vy"]
            errors_pos.append(dx**2 + dy**2)
            errors_vel.append(dvx**2 + dvy**2)
            
            # NEES Hesabı
            cov = np.diag([
                fused_row.get("sigma_x_m", fused_row.get("pos_sigma_m", 1.0))**2,
                fused_row.get("sigma_vx_mps", fused_row.get("vel_sigma_mps", 1.0))**2,
                fused_row.get("sigma_y_m", fused_row.get("pos_sigma_m", 1.0))**2,
                fused_row.get("sigma_vy_mps", fused_row.get("vel_sigma_mps", 1.0))**2,
            ])
            try:
                invP = np.linalg.inv(cov)
                e = np.array([dx, dvx, dy, dvy])
                nees_vals.append(float(e.T @ invP @ e))
            except np.linalg.LinAlgError:
                pass # Singular matrix durumu

        rmse_pos = float(np.sqrt(np.mean(errors_pos))) if errors_pos else np.nan
        rmse_vel = float(np.sqrt(np.mean(errors_vel))) if errors_vel else np.nan
        nees_mean = float(np.mean(nees_vals)) if nees_vals else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mota": mota,
        "id_switches": id_switches,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "rmse_pos_m": rmse_pos,
        "rmse_vel_mps": rmse_vel,
        "nees": nees_mean,
        "nees_ideal": 4.0,
    }


def print_metric_comparison(metrics_adv, metrics_basic):
    print("\n" + "=" * 85)
    print(f"{'METRİK KARŞILAŞTIRMASI':^85}")
    print("=" * 85)
    
    # Konsola düzgün basmak için bir DataFrame oluşturuyoruz
    comparison_df = pd.DataFrame({
        "Metrik": ["Precision", "Recall", "F1 Score", "MOTA", "ID Switch", "RMSE Pos (m)", "RMSE Vel (m/s)", "NEES (Ideal=4.0)"],
        "Gelişmiş Füzyon": [
            f"{metrics_adv['precision']:.3f}",
            f"{metrics_adv['recall']:.3f}",
            f"{metrics_adv['f1_score']:.3f}",
            f"{metrics_adv['mota']:.3f}",
            f"{metrics_adv['id_switches']}",
            f"{metrics_adv['rmse_pos_m']:.2f}",
            f"{metrics_adv['rmse_vel_mps']:.2f}",
            f"{metrics_adv['nees']:.2f}"
        ],
        "Temel Füzyon": [
            f"{metrics_basic['precision']:.3f}",
            f"{metrics_basic['recall']:.3f}",
            f"{metrics_basic['f1_score']:.3f}",
            f"{metrics_basic['mota']:.3f}",
            f"{metrics_basic['id_switches']}",
            f"{metrics_basic['rmse_pos_m']:.2f}",
            f"{metrics_basic['rmse_vel_mps']:.2f}",
            f"{metrics_basic['nees']:.2f}"
        ]
    })
    
    print(comparison_df.to_string(index=False))
    print("=" * 85 + "\n")


def evaluate_and_plot_comparison(gt_df, sensor_df, fused_df, basic_df, metrics_adv, metrics_basic):
    gt_key = "callsign" if "callsign" in gt_df.columns else "target"
    
    # Çizim Alanını Oluştur (2 Satır, 3 Sütun)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Füzyon Algoritmaları Karşılaştırmalı Performans Analizi", fontsize=18, fontweight="bold")
    
    def plot_algorithm_row(ax_xy, ax_err, result_df, title_prefix):
        if result_df.empty:
            ax_xy.set_title(f"{title_prefix} - Veri Yok")
            return

        track_ids = result_df["global_track_id"].unique()
        cmap = plt.get_cmap("tab10", max(len(track_ids), 1))

        # Best Match Fonksiyonu (Sadece ilgili DF için)
        def best_match(tid):
            tdf = result_df[result_df["global_track_id"] == tid]
            best, best_err = None, np.inf
            for cs in gt_df[gt_key].unique():
                sub = gt_df[gt_df[gt_key] == cs].sort_values("time")
                err = np.sqrt((tdf["x"].values - np.interp(tdf["time"], sub["time"], sub["x"]))**2 +
                              (tdf["y"].values - np.interp(tdf["time"], sub["time"], sub["y"]))**2).mean()
                if err < best_err:
                    best_err, best = err, cs
            return best, best_err

        match_map = {tid: best_match(tid) for tid in track_ids}

        # 1. X-Y Grafiği
        raw_real = sensor_df[sensor_df["is_clutter"] != True] if "is_clutter" in sensor_df.columns else sensor_df
        ax_xy.scatter(raw_real["x"], raw_real["y"], s=6, c="lightgray", alpha=0.35, zorder=1, label="Radar Ölçümleri")
        
        for cs in gt_df[gt_key].unique():
            sub = gt_df[gt_df[gt_key] == cs].sort_values("time")
            ax_xy.plot(sub["x"], sub["y"], linewidth=2.2, linestyle="--", color="black", zorder=3, label=f"GT {cs}")
            
        for i, tid in enumerate(sorted(track_ids)):
            tdf = result_df[result_df["global_track_id"] == tid].sort_values("time")
            ax_xy.plot(tdf["x"], tdf["y"], linewidth=2.0, color=cmap(i), alpha=0.95, zorder=5, label=f"Track {tid}")
            ax_xy.scatter(tdf["x"], tdf["y"], s=18, color=cmap(i), edgecolor="white", linewidth=0.8, zorder=6)
            
        ax_xy.set_title(f"{title_prefix} - XY Konum")
        ax_xy.set_xlabel("x (m)")
        ax_xy.set_ylabel("y (m)")
        ax_xy.grid(alpha=0.35)
        ax_xy.set_aspect("equal", adjustable="datalim")
        
        handles, labels = ax_xy.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_xy.legend(list(by_label.values())[:10], list(by_label.keys())[:10], fontsize=8, loc="best", ncol=2)

        # 2. Hata (Zaman) Grafiği
        for i, tid in enumerate(sorted(track_ids)):
            tdf = result_df[result_df["global_track_id"] == tid].sort_values("time")
            cs, _ = match_map[tid]
            if cs:
                sub = gt_df[gt_df[gt_key] == cs].sort_values("time")
                errs = np.sqrt((tdf["x"].values - np.interp(tdf["time"], sub["time"], sub["x"]))**2 +
                               (tdf["y"].values - np.interp(tdf["time"], sub["time"], sub["y"]))**2)
                ax_err.plot(tdf["time"], errs, color=cmap(i), linewidth=1.8, marker="o", markersize=3, label=tid)
                
        ax_err.set_title(f"{title_prefix} - Konum Hatası (m)")
        ax_err.set_xlabel("Zaman (s)")
        ax_err.set_ylabel("Hata (m)")
        ax_err.grid(alpha=0.35)
        ax_err.legend(fontsize=8, loc="upper right")

    # Satır 1: Gelişmiş Füzyon
    plot_algorithm_row(axes[0, 0], axes[0, 1], fused_df, "Gelişmiş Füzyon")
    
    # Satır 2: Temel Füzyon
    plot_algorithm_row(axes[1, 0], axes[1, 1], basic_df, "Temel Füzyon")

    # Sağ Üst (Satır 1, Sütun 3): Sensör TQ Dağılımı
    ax_tq = axes[0, 2]
    sensors = sensor_df["sensor"].unique()
    tq_values = [sensor_df[sensor_df["sensor"] == s]["track_quality"].dropna().values for s in sensors]
    bp = ax_tq.boxplot(tq_values, patch_artist=True)
    ax_tq.set_xticklabels(sensors)
    for patch, color in zip(bp["boxes"], ["#4e79a7", "#f28e2b", "#e15759"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax_tq.set_title("Sensör Bazında TQ Dağılımı")
    ax_tq.grid(alpha=0.3)

    # Sağ Alt (Satır 2, Sütun 3): Metriklerin Bar Chart ile Karşılaştırması
    ax_bar = axes[1, 2]
    labels = ['Precision', 'Recall', 'F1 Score', 'MOTA']
    adv_vals = [metrics_adv['precision'], metrics_adv['recall'], metrics_adv['f1_score'], metrics_adv['mota']]
    bas_vals = [metrics_basic['precision'], metrics_basic['recall'], metrics_basic['f1_score'], metrics_basic['mota']]
    
    x = np.arange(len(labels))
    width = 0.35

    ax_bar.bar(x - width/2, adv_vals, width, label='Gelişmiş', color='#2ca02c', alpha=0.8)
    ax_bar.bar(x + width/2, bas_vals, width, label='Temel', color='#d62728', alpha=0.8)

    ax_bar.set_ylabel('Skor (0-1)')
    ax_bar.set_title('Genel Başarı Metrikleri Karşılaştırması')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.legend()
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.set_ylim(0, 1.1)

    # Düzenleme ve Kayıt
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("pipeline_comparison_results.png", dpi=150, bbox_inches="tight")
    print("Görsel kaydedildi: pipeline_comparison_results.png")
    plt.show()


if __name__ == "__main__":
    print("Değerlendirme başlatılıyor... Dosyalar okunuyor.")
    gt_df, sensor_df, fused_df, basic_df = load_data()
    
    print("Gelişmiş Füzyon metrikleri hesaplanıyor...")
    metrics_adv = compute_tracking_metrics(gt_df, fused_df)
    
    print("Temel Füzyon metrikleri hesaplanıyor...")
    metrics_basic = compute_tracking_metrics(gt_df, basic_df)
    
    print_metric_comparison(metrics_adv, metrics_basic)
    evaluate_and_plot_comparison(gt_df, sensor_df, fused_df, basic_df, metrics_adv, metrics_basic)