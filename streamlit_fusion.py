import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_ground_truth(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data
def load_sensor_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


@st.cache_data
def load_fused_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def enu_to_latlon(x: float, y: float, ref_lat: float, ref_lon: float) -> (float, float):
    R = 6371000.0
    lat = ref_lat + (y / R) * 180.0 / np.pi
    lon = ref_lon + (x / (R * np.cos(ref_lat * np.pi / 180.0))) * 180.0 / np.pi
    return lat, lon


def build_map_data(gt_df: pd.DataFrame, sensor_df: pd.DataFrame, fused_df: pd.DataFrame):
    ref_lat = gt_df["lat"].median()
    ref_lon = gt_df["lon"].median()

    sensor_df = sensor_df.copy()
    fused_df = fused_df.copy()
    gt_df = gt_df.copy()

    sensor_df = sensor_df[sensor_df["is_clutter"] != True]
    

    lat_list: List[float] = []
    lon_list: List[float] = []
    type_list: List[str] = []
    time_list: List[float] = []
    id_list: List[str] = []
    hover_list: List[str] = []
    sensor_color_list: List[str] = []

    for _, row in sensor_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        lat_list.append(lat)
        lon_list.append(lon)
        type_list.append(f"Radar: {row['sensor']}")
        time_list.append(row["time"])
        id_list.append(row["local_track_id"])
        hover_list.append(
            f"Sensor: {row['sensor']}<br>Track: {row['local_track_id']}<br>Time: {row['time']:.2f}<br>TQ: {row['track_quality']}"
        )
        sensor_color_list.append(row["sensor"])

    fused_lat = []
    fused_lon = []
    fused_time = []
    fused_id = []
    fused_hover = []

    for _, row in fused_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        fused_lat.append(lat)
        fused_lon.append(lon)
        fused_time.append(row["time"])
        fused_id.append(row["global_track_id"])
        
        # Güvenli Erişim ve Kontrol (Doğrudan indeksleme)
        source_radars = str(row["source_radars"]) if "source_radars" in row else "Unknown"
        if source_radars == "nan" or source_radars.strip() == "":
            source_radars = "Unknown"
            
        source_details = str(row["source_measurement_details"]) if "source_measurement_details" in row else "Unknown"
        if source_details == "nan" or source_details.strip() == "":
            source_details = source_radars
            
        # DEVASA VERİ YÜKÜNÜ ENGELLEYEN YENİ KOD:
        # Metin çok uzunsa (örneğin 100 karakteri geçiyorsa) sadece son kısımlarını göster
        if len(source_details) > 100:
            # Noktalı virgüllerden bölüp sadece son 3 güncellemeyi alıyoruz
            meas_list = source_details.split(";")
            source_details = "... " + ";".join(meas_list[-3:]) if len(meas_list) > 3 else source_details[:100] + "..."
            
        # Prob (Olasılık) değeri için de güvenli erişim
        prob_val = float(row["prob"]) if "prob" in row else 0.0

        fused_hover.append(
            f"Fused ID: {row['global_track_id']}<br>Time: {row['time']:.2f}<br>Sources: {source_radars}<br>Measurements: {source_details}<br>Prob: {prob_val:.3f}"
        )

    gt_lat = []
    gt_lon = []
    gt_time = []
    gt_id = []
    gt_hover = []

    for _, row in gt_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        gt_lat.append(lat)
        gt_lon.append(lon)
        gt_time.append(row["time"])
        gt_id.append(row.get("callsign", str(row.name)))
        gt_hover.append(
            f"GT: {row.get('callsign', 'GT')}<br>Time: {row['time']:.2f}<br>Lat/Lon: {row['lat']:.6f},{row['lon']:.6f}"
        )

    data = pd.DataFrame({
        "lat": lat_list + fused_lat + gt_lat,
        "lon": lon_list + fused_lon + gt_lon,
        "type": type_list + ["Fused" for _ in fused_lat] + ["Ground Truth" for _ in gt_lat],
        "time": time_list + fused_time + gt_time,
        "id": id_list + fused_id + gt_id,
        "hover": hover_list + fused_hover + gt_hover,
        "sensor_group": sensor_color_list + ["Fused" for _ in fused_lat] + ["Ground Truth" for _ in gt_lat],
    })

    return data, ref_lat, ref_lon


def main():
    st.set_page_config(page_title="Radar Fusion Streamlit", layout="wide")
    st.title("Radar Füzyon Canlı Harita Görselleştirmesi")

    gt_file = st.sidebar.text_input("Ground truth CSV", "ground_truth_adsb_multi.csv")
    sensor_file = st.sidebar.text_input("Radar sensor CSV", "radar_sensor_tracks_gercekci.csv")
    fused_file = st.sidebar.text_input("Fused CSV", "res_real_adv.csv")
    refresh_sec = st.sidebar.number_input("Yenileme süresi (s)", min_value=1, max_value=30, value=3)

    if not os.path.exists(gt_file) or not os.path.exists(sensor_file) or not os.path.exists(fused_file):
        st.error("Lütfen tüm CSV dosyalarının bulunduğu konumu doğru girin.")
        return

    gt_df = load_ground_truth(gt_file)
    sensor_df = load_sensor_data(sensor_file)
    fused_df = load_fused_data(fused_file)

    if "source_radars" not in fused_df.columns:
        st.warning(
            "Seçili fused CSV'de `source_radars` sütunu yok. Bu durumda tüm füzyon noktalarında kaynak bilgisi `Unknown` görünebilir."
        )

    if "source_measurement_details" not in fused_df.columns:
        st.info(
            "Eğer `source_measurement_details` sütunu da yoksa, sadece radar isimleri gösterilir."
        )

    if "source_radars" in fused_df.columns:
        missing_sources = fused_df["source_radars"].isna() | (fused_df["source_radars"].astype(str).str.strip() == "")
        if missing_sources.any():
            count_missing = missing_sources.sum()
            st.warning(f"Fused CSV'de {count_missing} kayıt için source_radars boş. Bu noktalar Unknown olarak görünebilir.")

    data, ref_lat, ref_lon = build_map_data(gt_df, sensor_df, fused_df)

    st.markdown(
        "Bu uygulama, her radarın farklı renkte ölçümlerini, ground truth rotasını açık gri çizgi olarak ve füzyon sonuçlarını tıklanabilir noktalar olarak gösterir."
    )

    # Debug: Veri yapısını göster
    gt_data_in_frame = data[data["sensor_group"] == "Ground Truth"]
    st.sidebar.write(f"**Debug - Tüm veride GT noktaları:** {len(gt_data_in_frame)}")
    if len(gt_data_in_frame) > 0:
        st.sidebar.write(f"  - Zaman aralığı: {gt_data_in_frame['time'].min():.1f}s - {gt_data_in_frame['time'].max():.1f}s")
        st.sidebar.write(f"  - Hedefler: {sorted(gt_data_in_frame['id'].unique())}")

    st.sidebar.markdown("### Filtreler")
    sensors = sorted(sensor_df["sensor"].unique())
    selected_sensors = st.sidebar.multiselect("Radar seç", sensors, default=sensors)
    
    # Zaman aralığını belirle (GT verilerine göre)
    gt_times = data[data["sensor_group"] == "Ground Truth"]["time"]
    if len(gt_times) > 0:
        gt_min_time = float(gt_times.min())
        gt_max_time = float(gt_times.max())
    else:
        gt_min_time = float(data["time"].min())
        gt_max_time = float(data["time"].max())
    
    min_time = float(data["time"].min())
    max_time = float(data["time"].max())
    
    # Slider: GT verilerini tamamen kapsayacak şekilde default set et
    selected_time = st.sidebar.slider(
        "Zaman aralığı", 
        min_value=min_time, 
        max_value=max_time, 
        value=(gt_min_time, gt_max_time),  # Default olarak GT aralığını göster
        step=1.0
    )

    filtered = data[(data["time"] >= selected_time[0]) & (data["time"] <= selected_time[1])]
    filtered = filtered[filtered["sensor_group"].isin(selected_sensors + ["Fused", "Ground Truth"])]

    # Debug: Filtrelenmiş GT noktaları
    filtered_gt = filtered[filtered["sensor_group"] == "Ground Truth"]
    st.sidebar.write(f"**Filtrelenen GT noktaları:** {len(filtered_gt)}")
    if len(filtered_gt) > 0:
        st.sidebar.write(f"  - Hedefler: {sorted(filtered_gt['id'].unique())}")
        st.sidebar.write(f"  - Zaman aralığı: {filtered_gt['time'].min():.1f}s - {filtered_gt['time'].max():.1f}s")

    fig = go.Figure()

    for sensor in sensors:
        sensor_points = filtered[filtered["sensor_group"] == sensor]
        if not sensor_points.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=sensor_points["lat"],
                    lon=sensor_points["lon"],
                    mode="markers",
                    marker=dict(size=7),
                    name=f"Radar: {sensor}",
                    hovertext=sensor_points["hover"],
                    hoverinfo="text",
                )
            )

    fused_points = filtered[filtered["sensor_group"] == "Fused"]
    if not fused_points.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=fused_points["lat"],
                lon=fused_points["lon"],
                mode="markers",
                marker=dict(size=10, color="orange", symbol="circle"),
                name="Fused",
                hovertext=fused_points["hover"],
                hoverinfo="text",
            )
        )

    gt_points = filtered[filtered["sensor_group"] == "Ground Truth"]
    
    # Debug: Kaç GT noktası var ve hangi hedefler?
    if not gt_points.empty:
        unique_targets = sorted(gt_points["id"].unique())
        st.sidebar.info(f"**Ground Truth hedefleri:** {', '.join(unique_targets)}\n**Toplam GT noktası (filtered):** {len(gt_points)}")
        st.sidebar.write(f"**Trace ekleme detayları:**")
    else:
        st.sidebar.warning("Seçilen zaman aralığında Ground Truth verisi yok!")
    
    # Her hedef için ayrı renk (sıralı callsign'lere göre)
    gt_colors = ["#808080", "#0066CC", "#FF6600"]  # Gri, mavi, turuncu
    
    trace_count = 0
    if not gt_points.empty:
        sorted_callsigns = sorted(gt_points["id"].unique())
        st.sidebar.write(f"  Toplam hedef: {len(sorted_callsigns)}")
        
        for idx, callsign in enumerate(sorted_callsigns):
            grp = gt_points[gt_points["id"] == callsign].sort_values("time")
            st.sidebar.write(f"    - {callsign}: {len(grp)} points")
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=grp["lat"],
                    lon=grp["lon"],
                    mode="lines+markers",
                    line=dict(width=3, color=gt_colors[idx % len(gt_colors)]),
                    marker=dict(size=6, color=gt_colors[idx % len(gt_colors)]),
                    name=f"GT: {callsign}",
                    hovertext=grp["hover"],
                    hoverinfo="text",
                )
            )
            trace_count += 1
        st.sidebar.write(f"  ✓ {trace_count} trace eklendi")

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": ref_lat, "lon": ref_lon},
            zoom=7,
        ),
        margin={"r":0, "t":40, "l":0, "b":0},
        title="Radar Ölçümleri, Füzyon Sonuçları ve Ground Truth",
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    if not fused_df.empty:
        st.subheader("Seçili Zaman Aralığındaki Fusion Noktaları")
        # Sadece CSV dosyasında GERÇEKTEN var olan sütunları filtrele (Çökmeyi engeller)
        istenen_sutunlar = ["time", "global_track_id", "x", "y", "fused_tq", "prob", "n_sources", "source_radars", "source_measurement_details"]
        gosterilecek_sutunlar = [col for col in istenen_sutunlar if col in fused_df.columns]
        st.dataframe(
            fused_df[(fused_df["time"] >= selected_time[0]) & (fused_df["time"] <= selected_time[1])][
                gosterilecek_sutunlar
            ].sort_values(["time", "global_track_id"]).reset_index(drop=True),
            use_container_width=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Bu uygulamayı çalıştırmak için:\n`streamlit run streamlit_fusion.py`")
    st.sidebar.markdown(f"**Referans merkez:** {ref_lat:.6f}, {ref_lon:.6f}")


if __name__ == "__main__":
    main()
