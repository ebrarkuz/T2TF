"""Snapshot üzerinden çalışan bağımsız Streamlit gerçek zamanlı ENU haritası."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from realtime.runtime_config import RuntimeConfig
from visualization.realtime_state import load_snapshot


def _radar_hover(message: dict) -> str:
    position = message.get("position", {})
    velocity = message.get("velocity", {})
    polar = message.get("measurement") or {}
    return "<br>".join([
        "Tür: Radar ölçümü",
        f"Measurement ID: {message.get('measurement_id', 'N/A')}",
        f"Sensor ID: {message.get('sensor_id', 'N/A')}",
        f"Sequence: {message.get('sequence_number', 'N/A')}",
        f"Radar timestamp: {message.get('timestamp', 'N/A')}",
        f"Alınma zamanı: {message.get('received_at', 'N/A')}",
        f"X/Y/Z: {position.get('x_m', 'N/A')} / {position.get('y_m', 'N/A')} / {position.get('z_m', 'N/A')} m",
        f"VX/VY/VZ: {velocity.get('vx_mps', 'N/A')} / {velocity.get('vy_mps', 'N/A')} / {velocity.get('vz_mps', 'N/A')} m/s",
        f"Range: {polar.get('range_m', 'N/A')}",
        f"Azimuth: {polar.get('azimuth_rad', 'N/A')}",
        f"Elevation: {polar.get('elevation_rad', 'N/A')}",
        f"Radial velocity: {polar.get('radial_velocity_mps', 'N/A')}",
        f"Fusion'da kullanıldı: {message.get('used_in_fusion', False)}",
        f"Track ID: {message.get('assigned_track_id', 'N/A')}",
        f"Ret nedeni: {message.get('rejection_reason') or 'N/A'}",
    ])


def _fused_hover(message: dict, max_used: int = 5) -> str:
    position = message.get("position", {})
    velocity = message.get("velocity", {})
    metadata = message.get("fusion_metadata", {})
    used = metadata.get("used_measurements", [])
    lines = [
        "Tür: Fused track",
        f"Track ID: {message.get('track_id')}",
        f"Status: {message.get('track_status')}",
        f"State timestamp: {message.get('state_timestamp')}",
        f"Publish timestamp: {message.get('publish_timestamp')}",
        f"X/Y/Z: {position.get('x_m')} / {position.get('y_m')} / {position.get('z_m')} m",
        f"VX/VY/VZ: {velocity.get('vx_mps')} / {velocity.get('vy_mps')} / {velocity.get('vz_mps')} m/s",
        f"Update: {metadata.get('update_type')}",
        f"Filter: {metadata.get('filter_name')}",
        f"IMM: {metadata.get('model_probabilities')}",
        f"Kullanılan radar sayısı: {len(used)}",
    ]
    for index, measurement in enumerate(used[:max_used], start=1):
        delta = float(message.get("state_timestamp", 0.0)) - float(
            measurement.get("measurement_timestamp", 0.0)
        )
        lines.append(
            f"{index}) {measurement.get('measurement_id')} | {measurement.get('sensor_id')} | "
            f"seq={measurement.get('sequence_number')} | t={measurement.get('measurement_timestamp')} | Δt={delta:.3f}s"
        )
    if len(used) > max_used:
        lines.append(f"... ve {len(used) - max_used} ölçüm daha")
    return "<br>".join(lines)


def _load_ground_truth(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.ParserError):
        return pd.DataFrame()


def main() -> None:
    config = RuntimeConfig.load()
    st.set_page_config(page_title="Gerçek Zamanlı Radar Füzyonu", layout="wide")
    st.title("Gerçek Zamanlı Radar Füzyon Haritası")

    st.sidebar.subheader("Bağlantı ve görünüm")
    st.sidebar.text_input("Radar UDP bind", f"{config.radar_udp_host}:{config.radar_udp_port}", disabled=True)
    st.sidebar.text_input("Fused UDP hedef", f"{config.fused_udp_host}:{config.fused_udp_port}", disabled=True)
    ground_truth_path = st.sidebar.text_input("Ground truth CSV", config.ground_truth_path)
    show_gt = st.sidebar.checkbox("Ground truth göster", True)
    show_radar = st.sidebar.checkbox("Radar ölçümleri göster", True)
    show_fused = st.sidebar.checkbox("Fused track çizgilerini göster", True)
    radar_limit = st.sidebar.selectbox("Güncel radar ölçüm sayısı", [10, 20, 50, 100], index=1)
    fused_limit = st.sidebar.number_input("Track başına fused nokta", 10, 2000, config.max_fused_points_per_track, 10)
    refresh_ms = st.sidebar.number_input("Yenileme aralığı (ms)", 100, 5000, config.visualization_refresh_ms, 50)
    max_line_gap = st.sidebar.number_input("Maksimum çizgi boşluğu (s)", 0.1, 30.0, config.max_line_gap_s, 0.1)
    pause = st.sidebar.toggle("Görselleştirmeyi duraklat", False)
    if st.sidebar.button("Görsel geçmişi temizle"):
        st.session_state["clear_after"] = time.time()
        st.session_state.pop("last_snapshot", None)
        st.session_state["radar_visual_history"] = {}

    snapshot = load_snapshot(config.snapshot_path)
    if pause and "last_snapshot" in st.session_state:
        snapshot = st.session_state["last_snapshot"]
    elif not pause:
        st.session_state["last_snapshot"] = snapshot
    clear_after = float(st.session_state.get("clear_after", 0.0))

    diagnostics = snapshot.get("diagnostics", {})
    status_columns = st.columns(6)
    status_items = [
        ("Radar UDP", "Çalışıyor" if diagnostics.get("service_running") else "Veri bekleniyor"),
        ("Toplam alınan paket", diagnostics.get("total_received_packets", diagnostics.get("total_packets", 0))),
        ("Geçerli / ret", f"{diagnostics.get('valid_packets', 0)} / {diagnostics.get('rejected_packets', 0)}"),
        ("Paket/s", f"{diagnostics.get('packets_per_second', 0):.1f}"),
        ("Aktif / confirmed", f"{diagnostics.get('active_track_count', 0)} / {diagnostics.get('confirmed_track_count', 0)}"),
        ("Fused publish", diagnostics.get("published_fused_messages", 0)),
        ("Son radar paketi", diagnostics.get("last_radar_packet_time") or "N/A"),
        ("Son fused publish", diagnostics.get("last_fused_publish_time") or "N/A"),
        ("Ort./maks. gecikme", f"{diagnostics.get('average_fusion_latency_ms', 0):.2f}/{diagnostics.get('latency_max_ms', 0):.2f} ms"),
        ("Queue", f"{diagnostics.get('radar_queue_depth', 0)}/{diagnostics.get('radar_queue_capacity', 0)}"),
        ("Radar tamponu", len(snapshot.get("radar_measurements", []))),
        ("Fused tamponu", sum(len(points) for points in snapshot.get("fused_tracks", {}).values())),
    ]
    for index, (label, value) in enumerate(status_items):
        status_columns[index % 6].metric(label, value)

    incoming_radar = [
        item for item in snapshot.get("radar_measurements", [])
        if float(item.get("received_at", 0.0)) >= clear_after
    ]
    radar_history = st.session_state.setdefault("radar_visual_history", {})
    for item in incoming_radar:
        measurement_id = str(item.get("measurement_id"))
        radar_history.pop(measurement_id, None)
        radar_history[measurement_id] = item
    while len(radar_history) > int(radar_limit):
        radar_history.pop(next(iter(radar_history)))
    radar = list(radar_history.values())
    fused_tracks = {
        track_id: [
            point for point in points
            if float(point.get("publish_timestamp", 0.0)) >= clear_after
        ][-int(fused_limit):]
        for track_id, points in snapshot.get("fused_tracks", {}).items()
    }
    sensor_options = sorted({str(item.get("sensor_id")) for item in radar})
    selected_sensors = st.sidebar.multiselect("Sensor ID filtresi", sensor_options, default=sensor_options)
    track_options = sorted(fused_tracks)
    selected_tracks = st.sidebar.multiselect("Track ID filtresi", track_options, default=track_options)

    figure = go.Figure()
    ground_truth = _load_ground_truth(ground_truth_path)
    if show_gt and not ground_truth.empty:
        key = "callsign" if "callsign" in ground_truth else "target"
        for target, group in ground_truth.groupby(key, sort=False):
            group = group.sort_values("time")
            if len(group) > 5000:
                group = group.iloc[::max(1, len(group) // 5000)]
            custom = group[[column for column in ("time", "z", "vx", "vy", "vz") if column in group]]
            figure.add_trace(go.Scatter(
                x=group["x"], y=group["y"], mode="lines", name=f"GT {target}",
                customdata=custom,
                hovertemplate=f"GT {target}<br>X=%{{x:.1f}}<br>Y=%{{y:.1f}}<br>time/z/vx/vy/vz=%{{customdata}}<extra></extra>",
            ))
            figure.add_trace(go.Scatter(
                x=[group.iloc[0]["x"], group.iloc[-1]["x"]],
                y=[group.iloc[0]["y"], group.iloc[-1]["y"]],
                mode="markers", marker=dict(size=7), name=f"{target} başlangıç/bitiş",
                showlegend=False,
            ))

    if show_radar:
        selected = [item for item in radar if str(item.get("sensor_id")) in selected_sensors]
        if selected:
            figure.add_trace(go.Scatter(
                x=[item["position"]["x_m"] for item in selected],
                y=[item["position"]["y_m"] for item in selected],
                mode="markers", marker=dict(size=8, symbol="x", color="#ff7f0e"),
                name="Son radar ölçümleri",
                hovertext=[_radar_hover(item) for item in selected], hoverinfo="text",
            ))

    if show_fused:
        for track_id in selected_tracks:
            points = sorted(fused_tracks.get(track_id, []), key=lambda point: point["state_timestamp"])
            if not points:
                continue
            frame = pd.DataFrame({
                "time": [point["state_timestamp"] for point in points],
                "x": [point["position"]["x_m"] for point in points],
                "y": [point["position"]["y_m"] for point in points],
                "hover": [_fused_hover(point) for point in points],
            })
            frame["segment"] = frame["time"].diff().gt(float(max_line_gap)).cumsum()
            for segment_id, segment in frame.groupby("segment", sort=False):
                figure.add_trace(go.Scatter(
                    x=segment["x"], y=segment["y"], mode="lines+markers",
                    marker=dict(size=4), line=dict(width=3),
                    name=f"Track {track_id} / {segment_id}",
                    hovertext=segment["hover"], hoverinfo="text",
                ))
            latest = frame.iloc[-1]
            figure.add_trace(go.Scatter(
                x=[latest["x"]], y=[latest["y"]], mode="markers+text",
                text=[track_id], textposition="top center", marker=dict(size=12),
                name=f"Güncel {track_id}", hovertext=[latest["hover"]], hoverinfo="text",
                showlegend=False,
            ))

    figure.update_layout(
        height=760, xaxis_title="X / East (m)", yaxis_title="Y / North (m)",
        title="Lokal ENU X-Y görünümü", legend=dict(orientation="h"),
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(figure, width="stretch")
    if not diagnostics.get("service_running"):
        st.info("Radar verisi bekleniyor… Füzyon servisini ayrı terminalde başlatın.")
    st.caption(
        f"Snapshot: {Path(config.state_snapshot_path).resolve()} · Arayüz yenileme hedefi: {refresh_ms} ms. "
        "Servis adreslerini environment değişkenleriyle değiştirip servisi yeniden başlatın."
    )
    if not pause:
        time.sleep(float(refresh_ms) / 1000.0)
        st.rerun()


if __name__ == "__main__":
    main()
