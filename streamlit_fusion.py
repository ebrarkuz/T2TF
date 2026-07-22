import os
import json
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components


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


def build_color_map(values, palette):
    return {value: palette[idx % len(palette)] for idx, value in enumerate(values)}


def enu_to_latlon(x: float, y: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    R = 6371000.0
    lat = ref_lat + (y / R) * 180.0 / np.pi
    lon = ref_lon + (x / (R * np.cos(ref_lat * np.pi / 180.0))) * 180.0 / np.pi
    return lat, lon


def _get_altitude_column(df: pd.DataFrame) -> pd.Series:
    if "alt_m" in df.columns:
        return pd.to_numeric(df["alt_m"], errors="coerce").fillna(0.0)
    if "z" in df.columns:
        return pd.to_numeric(df["z"], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def _prepare_3d_route_frame(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    route_id_col: str,
    fallback_route_id: str,
) -> pd.DataFrame:
    frame = df.copy()

    if "lat" not in frame.columns or "lon" not in frame.columns:
        if "x" in frame.columns and "y" in frame.columns:
            lat_lon = frame.apply(lambda row: enu_to_latlon(float(row["x"]), float(row["y"]), ref_lat, ref_lon), axis=1)
            frame["lat"] = lat_lon.map(lambda item: item[0])
            frame["lon"] = lat_lon.map(lambda item: item[1])
        else:
            frame["lat"] = ref_lat
            frame["lon"] = ref_lon

    frame["alt_m"] = _get_altitude_column(frame)
    frame["time"] = pd.to_numeric(frame.get("time", 0.0), errors="coerce").fillna(0.0)
    if route_id_col in frame.columns:
        frame["route_id"] = frame[route_id_col].astype(str)
    else:
        frame["route_id"] = fallback_route_id

    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame = frame.dropna(subset=["lat", "lon"]).copy()
    return frame


def build_3d_deck(
    gt_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    pitch: float = 55.0,
    bearing: float = 0.0,
    zoom: float = 7.0,
):
    gt_frame = _prepare_3d_route_frame(
        gt_df,
        ref_lat,
        ref_lon,
        route_id_col="callsign" if "callsign" in gt_df.columns else ("target" if "target" in gt_df.columns else "route_id"),
        fallback_route_id="GT",
    )
    fused_frame = _prepare_3d_route_frame(
        fused_df,
        ref_lat,
        ref_lon,
        route_id_col="global_track_id" if "global_track_id" in fused_df.columns else "route_id",
        fallback_route_id="FUSED",
    )

    gt_paths = []
    for route_id, grp in gt_frame.groupby("route_id"):
        grp = grp.sort_values("time")
        path = grp[["lon", "lat", "alt_m"]].to_numpy().tolist()
        if len(path) >= 2:
            gt_paths.append({"route_id": route_id, "path": path})

    fused_paths = []
    for route_id, grp in fused_frame.groupby("route_id"):
        grp = grp.sort_values("time")
        path = grp[["lon", "lat", "alt_m"]].to_numpy().tolist()
        if len(path) >= 2:
            fused_paths.append({"route_id": route_id, "path": path})

    gt_points = gt_frame[["route_id", "lon", "lat", "alt_m", "time"]].copy()
    gt_points["route_type"] = "Ground Truth"
    fused_points = fused_frame[["route_id", "lon", "lat", "alt_m", "time"]].copy()
    fused_points["route_type"] = "Fused"

    layers = [
        pdk.Layer(
            "PathLayer",
            data=gt_paths,
            get_path="path",
            get_color=[0, 0, 255, 200],
            get_width=7,
            width_scale=1,
            width_min_pixels=3,
            pickable=True,
        ),
        pdk.Layer(
            "PathLayer",
            data=fused_paths,
            get_path="path",
            get_color=[255, 0, 0, 200],
            get_width=9,
            width_scale=1,
            width_min_pixels=4,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=gt_points,
            get_position="[lon, lat, alt_m]",
            get_fill_color=[0, 0, 255, 180],
            get_radius=50,
            radius_scale=1,
            radius_min_pixels=3,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=fused_points,
            get_position="[lon, lat, alt_m]",
            get_fill_color=[255, 120, 0, 180],
            get_radius=60,
            radius_scale=1,
            radius_min_pixels=3,
            pickable=True,
        ),
    ]

    combined = pd.concat([gt_points[["lat", "lon"]], fused_points[["lat", "lon"]]], ignore_index=True)
    center_lat = float(combined["lat"].mean()) if not combined.empty else ref_lat
    center_lon = float(combined["lon"].mean()) if not combined.empty else ref_lon
    if not np.isfinite(center_lat):
        center_lat = ref_lat
    if not np.isfinite(center_lon):
        center_lon = ref_lon

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=pitch,
            bearing=bearing,
        ),
        layers=layers,
        tooltip={
            "html": (
                "<b>{route_type}</b><br/>"
                "ID: {route_id}<br/>"
                "Time: {time}<br/>"
                "Altitude: {alt_m} m"
            ),
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )

    return deck, len(gt_paths), len(fused_paths), center_lat, center_lon


def _clean_route_points(grp: pd.DataFrame) -> list[dict]:
    """NaN/inf koordinat barındıran satırları 0.0'a düşürmek yerine tamamen atar.

    Bu, rotanın "Null Island" (0,0) üzerinden kıtalar arası sahte zikzaklar
    çizmesini engeller. Eksik nokta rotadan sadece düşer, rota o noktada
    kesintisiz şekilde komşu geçerli noktalar arasında devam eder.
    """
    grp = grp.copy()
    grp["lon"] = pd.to_numeric(grp["lon"], errors="coerce")
    grp["lat"] = pd.to_numeric(grp["lat"], errors="coerce")
    grp["alt_m"] = pd.to_numeric(grp["alt_m"], errors="coerce")
    grp["time"] = pd.to_numeric(grp["time"], errors="coerce")

    finite_mask = (
        np.isfinite(grp["lon"])
        & np.isfinite(grp["lat"])
        & np.isfinite(grp["alt_m"])
        & np.isfinite(grp["time"])
    )
    grp = grp.loc[finite_mask].sort_values("time")

    return [
        {"lon": float(r.lon), "lat": float(r.lat), "alt": float(r.alt_m), "time": float(r.time)}
        for r in grp.itertuples(index=False)
    ]


def build_cesium_html(
        gt_df: pd.DataFrame,
        fused_df: pd.DataFrame,
        ref_lat: float,
        ref_lon: float,
        pitch_deg: float = 55.0,
        heading_deg: float = 20.0,
):
        gt_frame = _prepare_3d_route_frame(
                gt_df,
                ref_lat,
                ref_lon,
                route_id_col="callsign" if "callsign" in gt_df.columns else ("target" if "target" in gt_df.columns else "route_id"),
                fallback_route_id="GT",
        )
        fused_frame = _prepare_3d_route_frame(
                fused_df,
                ref_lat,
                ref_lon,
                route_id_col="global_track_id" if "global_track_id" in fused_df.columns else "route_id",
                fallback_route_id="FUSED",
        )

        gt_routes = []
        for route_id, grp in gt_frame.groupby("route_id"):
                pts = _clean_route_points(grp)
                if len(pts) >= 2:
                        gt_routes.append({"route_id": str(route_id), "points": pts})

        fused_routes = []
        for route_id, grp in fused_frame.groupby("route_id"):
                pts = _clean_route_points(grp)
                if len(pts) >= 2:
                        fused_routes.append({"route_id": str(route_id), "points": pts})

        # Merkez/kamera hesabı da sadece finite (NaN olmayan) değerler üzerinden yapılıyor
        combined = pd.concat([gt_frame[["lat", "lon"]], fused_frame[["lat", "lon"]]], ignore_index=True)
        combined = combined[np.isfinite(combined["lat"]) & np.isfinite(combined["lon"])]
        center_lat = float(combined["lat"].mean()) if not combined.empty else float(ref_lat)
        center_lon = float(combined["lon"].mean()) if not combined.empty else float(ref_lon)
        if not np.isfinite(center_lat):
                center_lat = float(ref_lat)
        if not np.isfinite(center_lon):
                center_lon = float(ref_lon)

        all_alt = pd.concat([gt_frame["alt_m"], fused_frame["alt_m"]], ignore_index=True)
        all_alt = all_alt[np.isfinite(all_alt)]
        max_alt = float(all_alt.max()) if len(all_alt) > 0 else 0.0
        camera_height = max(60000.0, max_alt + 30000.0)

        gt_json = json.dumps(gt_routes, ensure_ascii=True)
        fused_json = json.dumps(fused_routes, ensure_ascii=True)

        html = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset=\"utf-8\" />
                <link href=\"https://unpkg.com/cesium@1.120/Build/Cesium/Widgets/widgets.css\" rel=\"stylesheet\" />
                <script src=\"https://unpkg.com/cesium@1.120/Build/Cesium/Cesium.js\"></script>
                <style>
                    html, body, #cesiumContainer {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; background: #0a0a0a; }}
                    #tip {{
                        position: absolute;
                        z-index: 1000;
                        pointer-events: none;
                        display: none;
                        padding: 6px 8px;
                        border-radius: 6px;
                        background: rgba(17, 24, 39, 0.9);
                        color: #fff;
                        font-family: sans-serif;
                        font-size: 12px;
                    }}
                </style>
            </head>
            <body>
                <div id=\"cesiumContainer\"></div>
                <div id=\"tip\"></div>
                <script>
                    const gtRoutes = {gt_json};
                    const fusedRoutes = {fused_json};

                    // KRİTİK: baseLayer:false vermezsen Cesium Viewer varsayılan
                    // olarak Ion üzerinden Bing katmanı yüklemeye çalışır. Ion
                    // token yoksa bu istek sessizce/console'da 401 ile
                    // başarısız olur ve globe tamamen siyah kalır.
                    const viewer = new Cesium.Viewer('cesiumContainer', {{
                        baseLayer: false,
                        timeline: false,
                        animation: false,
                        sceneModePicker: true,
                        baseLayerPicker: false,
                        geocoder: false,
                        homeButton: true,
                        fullscreenButton: true,
                        navigationHelpButton: true,
                        infoBox: false,
                        selectionIndicator: false,
                    }});

                    // Ion token gerektirmeyen, CORS destekli, subdomain'li
                    // (a/b/c/d) CartoDB "Positron" (light) katmanı.
                    // tile.openstreetmap.org doğrudan tarayıcıdan yoğun
                    // istekte sık sık 403/429 dönüp aynı siyah ekrana yol
                    // açabiliyor; CartoDB bunun için daha stabil.
                    const baseLayer = new Cesium.UrlTemplateImageryProvider({{
                        url: 'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                        subdomains: ['a', 'b', 'c', 'd'],
                        credit: '© OpenStreetMap contributors © CARTO',
                        maximumLevel: 19,
                    }});

                    let osmFallbackAdded = false;
                    let cartoErrorCount = 0;
                    baseLayer.errorEvent.addEventListener(() => {{
                        cartoErrorCount++;
                        // CartoDB de erişilemezse tek seferlik OSM fallback ekle
                        if (!osmFallbackAdded && cartoErrorCount > 5) {{
                            osmFallbackAdded = true;
                            viewer.imageryLayers.addImageryProvider(
                                new Cesium.UrlTemplateImageryProvider({{
                                    url: 'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                                    subdomains: ['a', 'b', 'c'],
                                    credit: '© OpenStreetMap contributors',
                                    maximumLevel: 19,
                                }})
                            );
                        }}
                    }});

                    viewer.imageryLayers.addImageryProvider(baseLayer);
                    viewer.scene.globe.depthTestAgainstTerrain = false;
                    viewer.scene.skyBox.show = false;
                    viewer.scene.skyAtmosphere.show = true;
                    viewer.scene.backgroundColor = Cesium.Color.BLACK;

                    function addRoute(route, color, width, routeType) {{
                        const flat = [];
                        for (const p of route.points) {{
                            // Python tarafı NaN'leri zaten attı; burada ikinci
                            // güvenlik katmanı olarak duruyor.
                            if (Number.isFinite(p.lon) && Number.isFinite(p.lat) && Number.isFinite(p.alt)) {{
                                flat.push(p.lon, p.lat, p.alt);
                            }}
                        }}

                        // Çizgi (polyline) artık kesintisiz çiziliyor çünkü
                        // flat dizisinde 0,0 sıçramaları yok
                        if (flat.length >= 6) {{
                            viewer.entities.add({{
                                polyline: {{
                                    positions: Cesium.Cartesian3.fromDegreesArrayHeights(flat),
                                    width: width,
                                    material: color,
                                    arcType: Cesium.ArcType.NONE
                                }}
                            }});
                        }}

                        for (const p of route.points) {{
                            if (!Number.isFinite(p.lon) || !Number.isFinite(p.lat) || !Number.isFinite(p.alt)) continue;
                            viewer.entities.add({{
                                position: Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt),
                                point: {{
                                    pixelSize: routeType === 'Ground Truth' ? 3 : 5,
                                    color: color,
                                    outlineColor: Cesium.Color.BLACK,
                                    outlineWidth: 1,
                                }},
                                meta: {{
                                    routeType: routeType,
                                    routeId: route.route_id,
                                    time: p.time,
                                    alt: p.alt,
                                }}
                            }});
                        }}
                    }}

                    // Ground Truth Rotaları (MAVİ)
                    for (const route of gtRoutes) {{
                        addRoute(route, Cesium.Color.DODGERBLUE, 4, 'Ground Truth');
                    }}

                    // Füzyon Rotaları (KIRMIZI)
                    for (const route of fusedRoutes) {{
                        addRoute(route, Cesium.Color.RED, 5, 'Fused');
                    }}

                    viewer.camera.setView({{
                        destination: Cesium.Cartesian3.fromDegrees({center_lon}, {center_lat}, {camera_height}),
                        orientation: {{
                            heading: Cesium.Math.toRadians({heading_deg}),
                            pitch: Cesium.Math.toRadians(-{pitch_deg}),
                            roll: 0.0,
                        }}
                    }});

                    const tip = document.getElementById('tip');
                    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
                    handler.setInputAction((movement) => {{
                        const picked = viewer.scene.pick(movement.endPosition);
                        if (picked && picked.id && picked.id.meta) {{
                            const m = picked.id.meta;
                            tip.style.display = 'block';
                            tip.style.left = (movement.endPosition.x + 12) + 'px';
                            tip.style.top = (movement.endPosition.y + 12) + 'px';
                            tip.innerHTML = `<b>${{m.routeType}}</b><br>ID: ${{m.routeId}}<br>Time: ${{Number(m.time).toFixed(2)}}<br>Altitude: ${{Number(m.alt).toFixed(1)}} m`;
                        }} else {{
                            tip.style.display = 'none';
                        }}
                    }}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
                </script>
            </body>
        </html>
        """

        return html, len(gt_routes), len(fused_routes), center_lat, center_lon


def build_map_data(
    gt_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    show_measurement_details: bool = False,
):
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
            
        # Prob (Olasılık) değeri için de güvenli erişim
        prob_val = float(row["prob"]) if "prob" in row else 0.0

        if show_measurement_details:
            source_details = str(row["source_measurement_details"]) if "source_measurement_details" in row else "Unknown"
            if source_details == "nan" or source_details.strip() == "":
                source_details = source_radars

            # Metin çok uzunsa sadece son kısımlarını göster
            if len(source_details) > 100:
                meas_list = source_details.split(";")
                source_details = "... " + ";".join(meas_list[-3:]) if len(meas_list) > 3 else source_details[:100] + "..."

            fused_hover.append(
                f"Fused ID: {row['global_track_id']}<br>Time: {row['time']:.2f}<br>Sources: {source_radars}<br>Measurements: {source_details}<br>Prob: {prob_val:.3f}"
            )
        else:
            fused_hover.append(
                f"Fused ID: {row['global_track_id']}<br>Time: {row['time']:.2f}<br>Sources: {source_radars}<br>Prob: {prob_val:.3f}"
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

    show_measurement_details = st.sidebar.checkbox(
        "Measurement details göster",
        value=False,
        help="Kapalıyken bu alan hover ve tabloda gizlenir. Büyük CSV'lerde performansı artırır.",
    )

    if "source_radars" in fused_df.columns:
        missing_sources = fused_df["source_radars"].isna() | (fused_df["source_radars"].astype(str).str.strip() == "")
        if missing_sources.any():
            count_missing = missing_sources.sum()
            st.warning(f"Fused CSV'de {count_missing} kayıt için source_radars boş. Bu noktalar Unknown olarak görünebilir.")

    data, ref_lat, ref_lon = build_map_data(
        gt_df,
        sensor_df,
        fused_df,
        show_measurement_details=show_measurement_details,
    )

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
        fused_palette = px.colors.qualitative.Dark24
        fused_ids = sorted(fused_points["id"].dropna().astype(str).unique())
        fused_color_map = build_color_map(fused_ids, fused_palette)

        st.sidebar.write(f"  Fused global ID sayısı: {len(fused_ids)}")
        for global_id in fused_ids:
            grp = fused_points[fused_points["id"].astype(str) == global_id].sort_values("time")
            fig.add_trace(
                go.Scattermapbox(
                    lat=grp["lat"],
                    lon=grp["lon"],
                    mode="markers",
                    marker=dict(size=10, color=fused_color_map[global_id], symbol="circle"),
                    name=f"Fused: {global_id}",
                    hovertext=grp["hover"],
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

    st.subheader("3B Görünüm (Cesium)")
    st.caption("2B harita korunur; bu panelde z ekseniyle kamerayı serbestçe döndürüp eğebilirsin.")

    cesium_pitch = st.slider("Cesium pitch", min_value=10.0, max_value=85.0, value=55.0, step=1.0)
    cesium_heading = st.slider("Cesium heading", min_value=-180.0, max_value=180.0, value=20.0, step=1.0)

    gt_3d = gt_df[(gt_df["time"] >= selected_time[0]) & (gt_df["time"] <= selected_time[1])].copy()
    fused_3d = fused_df[(fused_df["time"] >= selected_time[0]) & (fused_df["time"] <= selected_time[1])].copy()
    cesium_html, gt_route_count, fused_route_count, center_lat_3d, center_lon_3d = build_cesium_html(
        gt_3d,
        fused_3d,
        ref_lat,
        ref_lon,
        pitch_deg=cesium_pitch,
        heading_deg=cesium_heading,
    )

    st.sidebar.write(f"**3B rotalar (Cesium):** GT={gt_route_count}, Fusion={fused_route_count}")
    st.sidebar.write(f"**3B merkez:** {center_lat_3d:.6f}, {center_lon_3d:.6f}")
    components.html(cesium_html, height=700)

    if not fused_df.empty:
        st.subheader("Seçili Zaman Aralığındaki Fusion Noktaları")
        # Sadece CSV dosyasında GERÇEKTEN var olan sütunları filtrele (Çökmeyi engeller)
        istenen_sutunlar = ["time", "global_track_id", "x", "y", "z", "fused_tq", "prob", "n_sources", "source_radars"]
        if show_measurement_details:
            istenen_sutunlar.append("source_measurement_details")
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