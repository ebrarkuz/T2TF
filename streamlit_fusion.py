import os
import time
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk

# ---------------------------------------------------------------------------
# Füzyon algoritması (senin yazdığın, adım adım / step-by-step çalışan modül)
# ---------------------------------------------------------------------------
try:
    from fusion_imm import FusionCenterIMM3, measurement_cov_from_row, run_imm_fusion, TQ_MAX
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

from fusion_evaluation import compute_target_specific_metrics
from target5_akinci import (
    TARGET5_CALLSIGN,
    Target5Parameters,
    associate_fused_tracks_to_target5,
    match_fused_to_target5,
    target_separations,
    write_target5_scenario,
)


# ===========================================================================
# VERİ YÜKLEME YARDIMCILARI (OFFLINE MOD İÇİN - DEĞİŞMEDİ)
# ===========================================================================
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

# ===========================================================================
# THREE.JS 3D INSPECTOR (DEĞİŞMEDİ - AYNEN KORUNDU)
# ===========================================================================
def build_threejs_html(gt_df, fused_df):
    gt_cols = ["time", "callsign", "x", "y", "z"]
    fused_cols = ["time", "global_track_id", "x", "y", "z"]

    def cap_browser_rows(frame: pd.DataFrame, limit: int = 60000) -> pd.DataFrame:
        """Keep the embedded component comfortably below Streamlit's 200 MB limit."""
        if len(frame) <= limit:
            return frame
        step = int(np.ceil(len(frame) / limit))
        return frame.iloc[::step].copy()

    gt_df = cap_browser_rows(gt_df)
    fused_df = cap_browser_rows(fused_df)

    gt_csv_string = gt_df[
        [c for c in gt_cols if c in gt_df.columns]
    ].to_csv(index=False)

    fused_csv_string = fused_df[
        [c for c in fused_cols if c in fused_df.columns]
    ].to_csv(index=False)

    # HTML oluşturma işlemi...

    html_template = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fusion 3D Track Inspector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
      :root{ --bg:#030508; --panel:#0d131b; --panel-2:#101923; --border:#1c2733; --text:#d7e2ea; --text-dim:#6b7d8c; --accent-gt:#29d9c2; --accent-err:#ff5c5c; --axis-x:#ff6b6b; --axis-y:#5ce1a0; --axis-z:#ffd166; }
      *{box-sizing:border-box;}
      html,body{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:var(--bg); color:var(--text); font-family:'JetBrains Mono', monospace; }
      h1,h2,h3,.disp{ font-family:'Space Grotesk', sans-serif; }
      #app{ display:grid; grid-template-columns: 340px 1fr; grid-template-rows: 100vh; width:100vw; height:100vh; }
      #panel{ background:var(--panel); border-right:1px solid var(--border); overflow-y:auto; height:100vh; padding:18px 16px 40px 16px; }
      #panel::-webkit-scrollbar{ width:8px; } #panel::-webkit-scrollbar-thumb{ background:var(--border); border-radius:4px; }
      .brand{ display:flex; align-items:baseline; gap:8px; margin-bottom:4px; }
      .brand .dot{ width:9px; height:9px; border-radius:50%; background:var(--accent-gt); box-shadow:0 0 8px var(--accent-gt); }
      .brand h1{ font-size:16px; font-weight:700; margin:0; letter-spacing:.02em; }
      .sub{ color:var(--text-dim); font-size:11px; margin:0 0 20px 0; line-height:1.5; }
      .section{ margin-bottom:20px; padding-top:16px; border-top:1px solid var(--border); } .section:first-of-type{ border-top:none; padding-top:0; }
      .section h2{ font-size:10px; text-transform:uppercase; letter-spacing:.12em; color:var(--text-dim); margin:0 0 10px 0; font-weight:500; }
      .filebtn{ display:block; width:100%; text-align:left; padding:9px 10px; background:var(--panel-2); border:1px solid var(--accent-gt); border-radius:6px; color:var(--text); font-family:inherit; font-size:11.5px; margin-bottom:8px; }
      .filebtn .name{ color:var(--accent-gt); display:block; margin-top:2px; font-size:10.5px; }
      .rowbtns{ display:flex; gap:6px; }
      .smallbtn{ flex:1; background:var(--panel-2); border:1px solid var(--border); color:var(--text); font-family:inherit; font-size:11px; padding:7px 4px; border-radius:6px; cursor:pointer; }
      .smallbtn:hover{ border-color:var(--accent-gt); } .smallbtn.active{ border-color:var(--accent-gt); color:var(--accent-gt); }
      #timeRow{ display:flex; align-items:center; gap:8px; margin-top:10px; } #timeSlider{ flex:1; accent-color:var(--accent-gt); }
      #frameLabel{ font-size:10.5px; color:var(--text-dim); min-width:64px; text-align:right; }
      .legend-item{ display:flex; align-items:center; gap:8px; padding:5px 0; font-size:11px; } .legend-item input[type=checkbox]{ accent-color:var(--accent-gt); }
      .swatch{ width:10px; height:10px; border-radius:2px; flex-shrink:0; } .swatch.line{ border-radius:0; height:2px; }
      .legend-item .lbl{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .stat{ display:flex; justify-content:space-between; font-size:11px; padding:3px 0; } .stat .k{ color:var(--text-dim); } .stat .v{ font-weight:700; }
      .stat .v.axis-x{ color:var(--axis-x); } .stat .v.axis-y{ color:var(--axis-y); } .stat .v.axis-z{ color:var(--axis-z); }
      .match-row{ font-size:10.5px; padding:6px 8px; background:var(--panel-2); border-radius:6px; margin-bottom:6px; border-left:3px solid var(--accent-err); }
      .match-row .hdr{ color:var(--text); font-weight:700; margin-bottom:3px; } .match-row .axes span{ margin-right:10px; }
      #matchList{ max-height:220px; overflow-y:auto; } #matchList::-webkit-scrollbar{ width:6px; } #matchList::-webkit-scrollbar-thumb{ background:var(--border); border-radius:3px; }
      #viewport{ position:relative; height:100vh; overflow:hidden; } #canvasHost{ width:100%; height:100%; display:block; }
      #topbar{ position:absolute; top:14px; left:14px; right:14px; display:flex; justify-content:space-between; align-items:flex-start; pointer-events:none; }
      #viewbtns{ display:flex; gap:6px; pointer-events:all; }
      #viewbtns button{ background:rgba(13,19,27,.85); border:1px solid var(--border); color:var(--text); font-family:'Space Grotesk',sans-serif; font-size:11px; padding:7px 12px; border-radius:6px; cursor:pointer; backdrop-filter: blur(4px); }
      #viewbtns button:hover, #viewbtns button.active{ border-color:var(--accent-gt); color:var(--accent-gt); }
      #placeholder{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:10px; color:var(--text-dim); text-align:center; padding:40px; }
      #placeholder .big{ font-family:'Space Grotesk',sans-serif; font-size:20px; color:var(--text); }
    </style>
    </head>
    <body>
    <script id="gt_csv_data" type="text/csv">###GT_CSV_PLACEHOLDER###</script>
    <script id="fused_csv_data" type="text/csv">###FUSED_CSV_PLACEHOLDER###</script>

    <div id="app">
      <div id="panel">
        <div class="brand"><div class="dot"></div><h1>FUSION 3D INSPECTOR</h1></div>
        <p class="sub">Ground truth rota + fused track noktalarını 3B'de karşılaştır, eksen bazlı (X/Y/Z) hatayı incele.</p>
        <div class="section"><h2>1 · Veri Durumu</h2><div class="filebtn">Ground Truth (Streamlit'ten Alındı)<span class="name" id="gtName">Hazırlanıyor...</span></div><div class="filebtn">Fused Tracks (Streamlit'ten Alındı)<span class="name" id="fusedName">Hazırlanıyor...</span></div></div>
        <div class="section"><h2>2 · Oynatım</h2><div class="rowbtns"><button class="smallbtn" id="playBtn" disabled>▶ Oynat</button><button class="smallbtn" id="speedBtn" disabled>1x</button></div><div id="timeRow"><input type="range" id="timeSlider" min="0" max="0" value="0" step="1" disabled><span id="frameLabel">0 / 0</span></div></div>
        <div class="section"><h2>3 · Açı / Görünüm</h2><div class="rowbtns"><button class="smallbtn" data-view="iso">İzometrik</button><button class="smallbtn" data-view="top">Üstten (X-Y)</button></div><div class="rowbtns" style="margin-top:6px;"><button class="smallbtn" data-view="front">Önden (X-Z)</button><button class="smallbtn" data-view="side">Yandan (Y-Z)</button></div></div>
        <div class="section"><h2>4 · Katmanlar</h2><div class="legend-item"><input type="checkbox" id="toggleGtLine" checked><span class="swatch line" style="background:var(--accent-gt)"></span><span class="lbl">GT rotaları (Statik)</span></div><div class="legend-item"><input type="checkbox" id="toggleFusedTrail" checked><span class="swatch line" style="background:#8ecae6"></span><span class="lbl">Zamanla Uzayan Fused İzleri</span></div><div class="legend-item"><input type="checkbox" id="toggleErrLines" checked><span class="swatch line" style="background:#ffff00"></span><span class="lbl">Bağlantı ve Hata çizgileri</span></div><div class="legend-item"><input type="checkbox" id="toggleAxisBars" checked><span class="swatch line" style="background:linear-gradient(90deg,var(--axis-x),var(--axis-y),var(--axis-z))"></span><span class="lbl">Eksen hata çubukları</span></div><div class="legend-item"><input type="checkbox" id="toggleGrid" checked><span class="swatch" style="background:#172533"></span><span class="lbl">Zemin ızgarası</span></div></div>
        <div class="section"><h2>Dikey ölçek</h2><select id="verticalScale" class="smallbtn"><option value="1" selected>1x (gerçek)</option><option value="2">2x</option><option value="5">5x</option><option value="10">10x</option></select></div>
        <div class="section" id="tracksSection" style="display:none;"><h2>5 · Track'ler</h2><div id="trackLegend"></div></div>
        <div class="section" id="statsSection" style="display:none;"><h2>Kümülatif RMSE (0 → mevcut kare)</h2><div class="stat"><span class="k">RMSE X (doğu)</span><span class="v axis-x" id="rmseX">–</span></div><div class="stat"><span class="k">RMSE Y (kuzey)</span><span class="v axis-y" id="rmseY">–</span></div><div class="stat"><span class="k">RMSE Z (irtifa)</span><span class="v axis-z" id="rmseZ">–</span></div><div class="stat"><span class="k">RMSE toplam</span><span class="v" id="rmseTotal">–</span></div><div class="stat"><span class="k">Eşleşen / GT / FP</span><span class="v" id="matchCounts">–</span></div></div>
        <div class="section" id="frameMatchSection" style="display:none;"><h2>Bu Karedeki Eşleşmeler</h2><div id="matchList"></div></div>
      </div>
      <div id="viewport">
        <div id="canvasHost"></div>
        <div id="topbar"><div></div><div id="viewbtns" style="display:none;"><button data-view="iso" class="active">İzometrik</button><button data-view="top">Üstten</button><button data-view="front">Önden</button><button data-view="side">Yandan</button></div></div>
        <div id="placeholder"><div class="big">Yükleniyor...</div><div>Veriler Streamlit'ten alınıyor.</div></div>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
    <script>
    (function(){
    "use strict";

    const PALETTE = ['#ffff00', '#ff00ff', '#00ffff', '#00ff00', '#ff8800', '#aa00ff', '#ff0055'];
    function colorForIndex(i){ return PALETTE[i % PALETTE.length]; }

    let gtTracks = {};       
    let fusedTracks = {};    
    let frameTimes = [];     
    let currentFrame = 0;
    let playing = false;
    let playTimer = null;
    let playSpeed = 1;
    const SPEEDS = [1,2,4,0.5];
    let speedIdx = 0;

    function pickCol(headers, candidates){
      const lower = headers.map(h=>h.toLowerCase().trim());
      for(const c of candidates){
        const idx = lower.indexOf(c);
        if(idx>=0) return headers[idx];
      }
      return null;
    }

    function safeNum(val, fallback = 0) {
      if (val == null || val === '') return fallback;
      if (typeof val === 'string') val = val.replace(',', '.');
      const num = parseFloat(val);
      return isNaN(num) ? fallback : num;
    }

    function parseCsvString(csvString, cb){
      Papa.parse(csvString, { header:true, dynamicTyping:false, skipEmptyLines:true, complete: (res)=>cb(res.data, res.meta.fields) });
    }

    function buildGtTracks(rows, fields){
      const idCol = pickCol(fields, ['callsign','target','track_id','id']);
      const timeCol = pickCol(fields, ['time','t','timestamp']);
      const xCol = pickCol(fields, ['x']), yCol = pickCol(fields, ['y']);
      const zCol = pickCol(fields, ['z','alt','altitude','alt_m']);
      
      if(!idCol || !timeCol || !xCol || !yCol) return {};
      const tracks = {};
      rows.forEach(r=>{
        const id = String(r[idCol]); const t = safeNum(r[timeCol], null); const x = safeNum(r[xCol], null); const y = safeNum(r[yCol], null);
        if(t === null || x === null || y === null) return; 
        if(!tracks[id]) tracks[id] = [];
        tracks[id].push({ time: t, x: x, y: y, z: zCol ? safeNum(r[zCol]) : 0 });
      });
      Object.values(tracks).forEach(arr=>arr.sort((a,b)=>a.time-b.time));
      return tracks;
    }

    function buildFusedTracks(rows, fields){
      const idCol = pickCol(fields, ['global_track_id','track_id','id']);
      const timeCol = pickCol(fields, ['time','t','timestamp']);
      const xCol = pickCol(fields, ['x']), yCol = pickCol(fields, ['y']), zCol = pickCol(fields, ['z','alt','altitude']);
      
      if(!idCol || !timeCol || !xCol || !yCol) return {};
      const tracks = {};
      rows.forEach(r=>{
        const id = String(r[idCol]); const t = safeNum(r[timeCol], null); const x = safeNum(r[xCol], null); const y = safeNum(r[yCol], null);
        if(t === null || x === null || y === null) return;
        if(!tracks[id]) tracks[id] = [];
        tracks[id].push({ time: t, x: x, y: y, z: zCol ? safeNum(r[zCol]) : 0 });
      });
      Object.values(tracks).forEach(arr=>arr.sort((a,b)=>a.time-b.time));
      return tracks;
    }

    function interpGtAt(arr, t){
      if(arr.length<2 || t<arr[0].time || t>arr[arr.length-1].time) return null;
      let lo=0, hi=arr.length-1;
      if(t<=arr[0].time) return arr[0]; if(t>=arr[hi].time) return arr[hi];
      while(hi-lo>1){ const mid=(lo+hi)>>1; if(arr[mid].time<=t) lo=mid; else hi=mid; }
      const a=arr[lo], b=arr[hi], f = (t-a.time)/(b.time-a.time || 1);
      return { time: t, x: a.x + f*(b.x-a.x), y: a.y + f*(b.y-a.y), z: a.z + f*(b.z-a.z) };
    }

    let scene, camera, renderer, controls, canvasHost;
    let gtGroup, fusedGroup, errGroup, gridGroup, labelGroup;
    let sceneCenter = new THREE.Vector3(), sceneRadius = 100;

    let verticalScale = 1;
    let plotOrigin = {x:0, y:0, z:0};
    // ENU -> Three.js: east=X, up=Y, north=-Z. GT and fused data share
    // the same local origin and vertical exaggeration.
    function toThree(p){
      return new THREE.Vector3(
        p.x - plotOrigin.x,
        (p.z - plotOrigin.z) * verticalScale,
        -(p.y - plotOrigin.y)
      );
    }

    function initScene(){
      canvasHost = document.getElementById('canvasHost');
      scene = new THREE.Scene(); scene.background = new THREE.Color(0x030508); 
      camera = new THREE.PerspectiveCamera(45, canvasHost.clientWidth/canvasHost.clientHeight, 0.1, 1e8); 
      renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(window.devicePixelRatio); renderer.setSize(canvasHost.clientWidth, canvasHost.clientHeight);
      canvasHost.appendChild(renderer.domElement);
      controls = new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.08;
      scene.add(new THREE.AmbientLight(0xffffff, 0.8)); const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1,2,1); scene.add(dl);
      gtGroup = new THREE.Group(); scene.add(gtGroup); fusedGroup = new THREE.Group(); scene.add(fusedGroup); errGroup = new THREE.Group(); scene.add(errGroup); gridGroup = new THREE.Group(); scene.add(gridGroup); labelGroup = new THREE.Group(); scene.add(labelGroup);
      window.addEventListener('resize', onResize); animate();
    }
    function onResize(){ camera.aspect = canvasHost.clientWidth/canvasHost.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(canvasHost.clientWidth, canvasHost.clientHeight); }
    function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

    function makeTextSprite(text, colorHex){
      const cnv = document.createElement('canvas'); cnv.width=256; cnv.height=64; const ctx = cnv.getContext('2d');
      ctx.font = 'bold 44px Space Grotesk, sans-serif'; ctx.fillStyle = colorHex; ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.shadowColor = "black"; ctx.shadowBlur = 4; ctx.lineWidth = 4; ctx.strokeText(text, 10, 32); ctx.fillText(text, 10, 32);
      const tex = new THREE.CanvasTexture(cnv); const mat = new THREE.SpriteMaterial({map:tex, depthTest:false, transparent:true}); const spr = new THREE.Sprite(mat);
      spr.scale.set(sceneRadius*0.18, sceneRadius*0.045, 1); return spr;
    }

    function buildStaticScene(){
      [gtGroup, fusedGroup, gridGroup, labelGroup, errGroup].forEach(g=>{ while(g.children.length) g.remove(g.children[0]); });
      const rawPoints = Object.values(gtTracks).flat().concat(Object.values(fusedTracks).flat());
      if(rawPoints.length){
        plotOrigin = {
          x: rawPoints.reduce((v,p)=>Math.min(v,p.x), Infinity),
          y: rawPoints.reduce((v,p)=>Math.min(v,p.y), Infinity),
          z: rawPoints.reduce((v,p)=>Math.min(v,p.z), Infinity)
        };
      }
      const box = new THREE.Box3();
      Object.values(gtTracks).forEach(arr=>arr.forEach(p=>box.expandByPoint(toThree(p)))); Object.values(fusedTracks).forEach(arr=>arr.forEach(p=>box.expandByPoint(toThree(p))));
      if(box.isEmpty()){ box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(100,100,100)); }
      box.getCenter(sceneCenter); sceneRadius = Math.max(box.getSize(new THREE.Vector3()).length()/2, 10);
      const gridSize = Math.max(sceneRadius*2.4, 100); const grid = new THREE.GridHelper(gridSize, 30, 0x1c3140, 0x0a1622); grid.position.set(sceneCenter.x, box.min.y, sceneCenter.z); gridGroup.add(grid);
      const axisLen = sceneRadius*0.5, origin = new THREE.Vector3(box.min.x, box.min.y, box.min.z);
      addAxisArrow(origin, new THREE.Vector3(1,0,0), axisLen, 0xff6b6b, 'X'); addAxisArrow(origin, new THREE.Vector3(0,0,-1), axisLen, 0x5ce1a0, 'Y'); addAxisArrow(origin, new THREE.Vector3(0,1,0), axisLen, 0xffd166, 'Z');
      Object.entries(gtTracks).forEach(([id,arr])=>{
        const pts = arr.map(toThree); if(pts.length < 2) return; const path = new THREE.CurvePath();
        for(let i=0; i<pts.length-1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i+1]));
        const geo = new THREE.TubeGeometry(path, Math.min(pts.length * 2, 300), sceneRadius * 0.004, 6, false); const mat = new THREE.MeshBasicMaterial({color:0x29d9c2, transparent:true, opacity:0.4}); gtGroup.add(new THREE.Mesh(geo, mat));
      });
      let idx=0; Object.keys(fusedTracks).forEach(id=>{ trackColors[id] = colorForIndex(idx++); }); fitCameraToScene('iso');
    }

    function addAxisArrow(origin, dir, len, colorHex, label){
      gridGroup.add(new THREE.ArrowHelper(dir, origin, len, colorHex, len*0.12, len*0.06));
      const spr = makeTextSprite(label, '#'+colorHex.toString(16).padStart(6,'0')); spr.position.copy(origin).add(dir.clone().multiplyScalar(len*1.15)); gridGroup.add(spr); 
    }

    let trackColors = {}, movingMeshes = {}, gtMovingMeshes = {}, gtDynamicLabels = {}; 

    function ensureMovingMesh(store, id, colorHex, radius, isFused){
      if(store[id]) return store[id];
      let mat; if (isFused) mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.9, roughness: 0.2 }); else mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.3 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 16), mat); scene.add(mesh); store[id] = mesh; return mesh;
    }
    function ensureGtLabel(id) { if(!gtDynamicLabels[id]) { const spr = makeTextSprite(id, '#29d9c2'); scene.add(spr); gtDynamicLabels[id] = spr; } return gtDynamicLabels[id]; }
    function clearGroupSafely(group) { while(group.children.length > 0) { const child = group.children[0]; if(child.geometry) child.geometry.dispose(); if(child.material) child.material.dispose(); group.remove(child); } }

    function updateFrame(){
      const t = frameTimes[currentFrame]; if(t===undefined) return;
      clearGroupSafely(errGroup); clearGroupSafely(fusedGroup);
      const gtNowState = {}, isGtVis = document.getElementById('toggleGtLine').checked; gtGroup.visible = isGtVis;

      Object.entries(gtTracks).forEach(([id,arr])=>{
        const st = interpGtAt(arr, t), mesh = ensureMovingMesh(gtMovingMeshes, id, 0x29d9c2, sceneRadius*0.015, false), label = ensureGtLabel(id);
        mesh.visible = isGtVis && !!st; label.visible = isGtVis && !!st;
        if(st){ const p3 = toThree(st); mesh.position.copy(p3); label.position.copy(p3).add(new THREE.Vector3(sceneRadius*0.05, sceneRadius*0.05, 0)); gtNowState[id] = {data:st, three:p3}; }
      });

      const fusedNow = [], showTrail = document.getElementById('toggleFusedTrail').checked;
      Object.entries(fusedTracks).forEach(([id,arr])=>{
        const pastPts = arr.filter(p => p.time <= t), color = trackColors[id] || '#8ecae6', wantVisible = document.getElementById('trackVis_'+id) ? document.getElementById('trackVis_'+id).checked : true;
        const mesh = ensureMovingMesh(movingMeshes, id, color, sceneRadius*0.006, true);
        if(pastPts.length > 0 && wantVisible) {
          mesh.visible = true; const currentData = pastPts[pastPts.length - 1], p3 = toThree(currentData); mesh.position.copy(p3); fusedNow.push({id, data:currentData, three:p3, color});
          if (showTrail && pastPts.length > 1) {
            const path = new THREE.CurvePath(); let validSegments = 0;
            for(let i=0; i<pastPts.length-1; i++) { const pA = toThree(pastPts[i]), pB = toThree(pastPts[i+1]); if(pA.distanceTo(pB) > 0.0001) { path.add(new THREE.LineCurve3(pA, pB)); validSegments++; } }
            if(validSegments > 0) fusedGroup.add(new THREE.Mesh(new THREE.TubeGeometry(path, Math.min(validSegments * 2, 200), sceneRadius * 0.0025, 6, false), new THREE.MeshBasicMaterial({color: color, transparent: true, opacity: 0.6})));
          }
        } else mesh.visible = false;
      });

      const matches = [];
      fusedNow.forEach(f=>{
        let best=null, bestD=Infinity, bestId=null;
        Object.entries(gtNowState).forEach(([gid,g])=>{ const d = Math.hypot(f.data.x-g.data.x, f.data.y-g.data.y, f.data.z-g.data.z); if(d<bestD){ bestD=d; best=g; bestId=gid; } });
        if(best) matches.push({fused:f, gt:best, gtId:bestId, dist:bestD, dx:f.data.x-best.data.x, dy:f.data.y-best.data.y, dz:f.data.z-best.data.z});
      });

      const showErr = document.getElementById('toggleErrLines').checked, showBars = document.getElementById('toggleAxisBars').checked;
      matches.forEach(m=>{
        if(showErr) errGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([m.fused.three, m.gt.three]), new THREE.LineBasicMaterial({ color: m.fused.color, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending })));
        if(showBars){ const g = m.gt.three; addErrBar(g, new THREE.Vector3(1,0,0), m.dx, 0xff6b6b); addErrBar(g, new THREE.Vector3(0,0,-1), m.dy, 0x5ce1a0); addErrBar(g, new THREE.Vector3(0,1,0), m.dz*verticalScale, 0xffd166); }
      });
      document.getElementById('frameLabel').textContent = (currentFrame+1)+' / '+frameTimes.length+'  (t='+(+t.toFixed(2))+')'; updateMatchPanel(matches); updateCumulativeStats();
    }

    function addErrBar(origin, dirUnit, signedLen, colorHex){
      errGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, origin.clone().add(dirUnit.clone().multiplyScalar(signedLen))]), new THREE.LineBasicMaterial({color:colorHex, transparent:true, opacity:0.8})));
    }

    function fitCameraToScene(view){
      const d = sceneRadius*2.6; let pos, up=new THREE.Vector3(0,1,0);
      if(view==='top'){ pos = sceneCenter.clone().add(new THREE.Vector3(0.0001, d, 0)); up.set(0,0,-1); }
      else if(view==='front'){ pos = sceneCenter.clone().add(new THREE.Vector3(0, 0, d)); }
      else if(view==='side'){ pos = sceneCenter.clone().add(new THREE.Vector3(d, 0, 0.0001)); }
      else { pos = sceneCenter.clone().add(new THREE.Vector3(d*0.6, d*0.55, d*0.6)); } 
      camera.position.copy(pos); camera.up.copy(up); camera.lookAt(sceneCenter); controls.target.copy(sceneCenter); controls.update();
    }

    let sceneReady=false;
    function tryBuildScene(){
      if(!Object.keys(gtTracks).length || !Object.keys(fusedTracks).length) return;
      Object.values(gtDynamicLabels).forEach(lbl => scene.remove(lbl)); for (let key in gtDynamicLabels) delete gtDynamicLabels[key];
      document.getElementById('placeholder').style.display='none'; document.getElementById('viewbtns').style.display='flex';
      if(!sceneReady){ initScene(); sceneReady=true; }
      buildStaticScene(); buildTrackLegend();
      frameTimes = Array.from(new Set(Object.values(fusedTracks).flat().map(r=>r.time))).sort((a,b)=>a-b);
      currentFrame = 0; const slider = document.getElementById('timeSlider'); slider.min=0; slider.max=Math.max(frameTimes.length-1,0); slider.value=0; slider.disabled=false;
      document.getElementById('playBtn').disabled=false; document.getElementById('speedBtn').disabled=false; document.getElementById('statsSection').style.display='block'; document.getElementById('frameMatchSection').style.display='block';
      updateFrame();
    }

    function buildTrackLegend(){
      document.getElementById('tracksSection').style.display='block'; const wrap = document.getElementById('trackLegend'); wrap.innerHTML='';
      Object.keys(fusedTracks).forEach((id)=>{
        const color = trackColors[id];
        const row = document.createElement('div'); row.className='legend-item';
        row.innerHTML = `<input type="checkbox" id="trackVis_${id}" checked> <span class="swatch" style="background:${color}"></span> <span class="lbl">Track ${id}</span>`;
        wrap.appendChild(row); row.querySelector('input').addEventListener('change', updateFrame);
      });
    }

    document.getElementById('timeSlider').addEventListener('input', e=>{ currentFrame = +e.target.value; updateFrame(); });
    document.getElementById('playBtn').addEventListener('click', ()=>{
      playing = !playing; document.getElementById('playBtn').textContent = playing ? '⏸ Duraklat' : '▶ Oynat';
      if(playing) startPlay(); else stopPlay();
    });
    document.getElementById('speedBtn').addEventListener('click', ()=>{ speedIdx = (speedIdx+1)%SPEEDS.length; playSpeed = SPEEDS[speedIdx]; document.getElementById('speedBtn').textContent = playSpeed+'x'; if(playing){ stopPlay(); startPlay(); } });
    function startPlay(){
      stopPlay();
      playTimer = setInterval(()=>{ currentFrame++; if(currentFrame>=frameTimes.length){ currentFrame=0; playing = false; document.getElementById('playBtn').textContent = '▶ Oynat'; stopPlay(); return;} document.getElementById('timeSlider').value = currentFrame; updateFrame(); }, 400/playSpeed);
    }
    function stopPlay(){ if(playTimer) clearInterval(playTimer); playTimer=null; }

    document.querySelectorAll('[data-view]').forEach(btn=>{ btn.addEventListener('click', ()=>{ fitCameraToScene(btn.dataset.view); document.querySelectorAll('[data-view]').forEach(b=>b.classList.remove('active')); document.querySelectorAll(`[data-view="${btn.dataset.view}"]`).forEach(b=>b.classList.add('active')); }); });
    ['toggleGtLine','toggleFusedTrail','toggleErrLines','toggleAxisBars','toggleGrid'].forEach(id=>{ document.getElementById(id).addEventListener('change', ()=>{ if(id==='toggleGrid'){ gridGroup.visible = document.getElementById(id).checked; } if(sceneReady) updateFrame(); }); });
    document.getElementById('verticalScale').addEventListener('change', e=>{
      verticalScale = Number(e.target.value) || 1;
      if(sceneReady){ buildStaticScene(); updateFrame(); }
    });

    function updateFrameLabel(t){ document.getElementById('frameLabel').textContent = (currentFrame+1)+' / '+frameTimes.length+'  (t='+(+t.toFixed(2))+')'; }

    function updateMatchPanel(matches){
      const wrap = document.getElementById('matchList'); wrap.innerHTML='';
      if(!matches.length){ wrap.innerHTML = '<div class="hint">Bu karede eşleşme yok.</div>'; return; }
      matches.sort((a,b)=>b.dist-a.dist).forEach(m=>{
        const color = trackColors[m.fused.id] || '#8ecae6';
        const div = document.createElement('div'); div.className='match-row'; div.style.borderLeftColor = color;
        div.innerHTML = `<div class="hdr" style="color:${color}">Track ${m.fused.id} → GT ${m.gtId}</div>
          <div class="axes"> <span style="color:var(--axis-x)">dX ${(m.dx>=0?'+':'')+m.dx.toFixed(1)}m</span> <span style="color:var(--axis-y)">dY ${(m.dy>=0?'+':'')+m.dy.toFixed(1)}m</span> <span style="color:var(--axis-z)">dZ ${(m.dz>=0?'+':'')+m.dz.toFixed(1)}m</span> </div>
          <div style="color:var(--text-dim); margin-top:2px;">3B mesafe: ${m.dist.toFixed(1)}m</div>`;
        wrap.appendChild(div);
      });
    }
    function signed(v){ return (v>=0?'+':'')+v.toFixed(1); }

    function updateCumulativeStats(){
      let sx=0, sy=0, sz=0, n=0, tp=0, gtCount=0, fp=0;
      for(let fi=0; fi<=currentFrame; fi++){
        const t = frameTimes[fi]; const gtNow = {}; let gtN=0;
        Object.entries(gtTracks).forEach(([id,arr])=>{ const st=interpGtAt(arr,t); if(st){ gtNow[id]=st; gtN++; } });
        gtCount += gtN;
        const fusedNow = []; Object.values(fusedTracks).forEach(arr=>{ const r = arr.find(p=>p.time===t); if(r) fusedNow.push(r); });
        const usedGt = new Set();
        fusedNow.forEach(f=>{
          let bestD=Infinity, bestId=null, best=null;
          Object.entries(gtNow).forEach(([gid,g])=>{ const d=Math.hypot(f.x-g.x, f.y-g.y, f.z-g.z); if(d<bestD){ bestD=d; bestId=gid; best=g; } });
          if(best && !usedGt.has(bestId)){ usedGt.add(bestId); tp++; sx += (f.x-best.x)**2; sy += (f.y-best.y)**2; sz += (f.z-best.z)**2; n++; } else { fp++; }
        });
      }
      document.getElementById('rmseX').textContent = n? Math.sqrt(sx/n).toFixed(2)+' m' : '–';
      document.getElementById('rmseY').textContent = n? Math.sqrt(sy/n).toFixed(2)+' m' : '–';
      document.getElementById('rmseZ').textContent = n? Math.sqrt(sz/n).toFixed(2)+' m' : '–';
      document.getElementById('rmseTotal').textContent = n? Math.sqrt((sx+sy+sz)/n).toFixed(2)+' m' : '–';
      document.getElementById('matchCounts').textContent = tp+' / '+gtCount+' / '+fp;
    }

    let loadedCount = 0;
    function checkAutoLoad() {
      loadedCount++;
      if(loadedCount === 2) {
        tryBuildScene();
      }
    }

    const gtCsvText = document.getElementById('gt_csv_data').textContent.trim();
    const fusedCsvText = document.getElementById('fused_csv_data').textContent.trim();

    if(gtCsvText && fusedCsvText) {
        parseCsvString(gtCsvText, (rows, fields)=>{
            gtTracks = buildGtTracks(rows, fields);
            document.getElementById('gtName').textContent = Object.keys(gtTracks).length+' hedef';
            checkAutoLoad();
        });
        parseCsvString(fusedCsvText, (rows, fields)=>{
            fusedTracks = buildFusedTracks(rows, fields);
            document.getElementById('fusedName').textContent = Object.keys(fusedTracks).length+' track';
            checkAutoLoad();
        });
    }
    })();
    </script>
    </body>
    </html>
    """

    html_content = html_template.replace("###GT_CSV_PLACEHOLDER###", gt_csv_string)
    html_content = html_content.replace("###FUSED_CSV_PLACEHOLDER###", fused_csv_string)
    return html_content


# ===========================================================================
# 2D PLOTLY HARİTA VERİ HAZIRLAMA (OFFLINE MOD İÇİN)
# ===========================================================================
def build_map_data(gt_df: pd.DataFrame, sensor_df: pd.DataFrame, fused_df: pd.DataFrame, show_measurement_details: bool = False):
    ref_lat, ref_lon = gt_df["lat"].median(), gt_df["lon"].median()
    sensor_df, fused_df, gt_df = sensor_df.copy(), fused_df.copy(), gt_df.copy()
    if "is_clutter" in sensor_df.columns: sensor_df = sensor_df[sensor_df["is_clutter"] != True]
    
    lat_list, lon_list, type_list, time_list, id_list, hover_list, sensor_color_list = [], [], [], [], [], [], []
    for _, row in sensor_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        lat_list.append(lat); lon_list.append(lon); type_list.append(f"Radar: {row['sensor']}")
        time_list.append(row["time"]); id_list.append(row["local_track_id"])
        hover_list.append(f"Sensor: {row['sensor']}<br>Track: {row['local_track_id']}<br>Time: {row['time']:.2f}<br>TQ: {row.get('track_quality', '')}")
        sensor_color_list.append(row["sensor"])

    fused_lat, fused_lon, fused_time, fused_id, fused_hover = [], [], [], [], []
    for _, row in fused_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        fused_lat.append(lat); fused_lon.append(lon); fused_time.append(row["time"]); fused_id.append(row["global_track_id"])
        prob_val = float(row["prob"]) if "prob" in row else 0.0
        source_radars = str(row["source_radars"]) if "source_radars" in row else "Unknown"
        fused_hover.append(f"Fused ID: {row['global_track_id']}<br>Time: {row['time']:.2f}<br>Prob: {prob_val:.3f}")

    gt_lat, gt_lon, gt_time, gt_id, gt_hover = [], [], [], [], []
    for _, row in gt_df.iterrows():
        lat, lon = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
        gt_lat.append(lat); gt_lon.append(lon); gt_time.append(row["time"]); gt_id.append(row.get("callsign", str(row.name)))
        gt_hover.append(f"GT: {row.get('callsign', 'GT')}<br>Time: {row['time']:.2f}<br>Lat/Lon: {lat:.6f},{lon:.6f}")

    data = pd.DataFrame({
        "lat": lat_list + fused_lat + gt_lat,
        "lon": lon_list + fused_lon + gt_lon,
        "type": type_list + ["Fused"]*len(fused_lat) + ["Ground Truth"]*len(gt_lat),
        "time": time_list + fused_time + gt_time,
        "id": id_list + fused_id + gt_id,
        "hover": hover_list + fused_hover + gt_hover,
        "sensor_group": sensor_color_list + ["Fused"]*len(fused_lat) + ["Ground Truth"]*len(gt_lat),
    })
    return data, ref_lat, ref_lon


# ===========================================================================
# OTONOM DÖNGÜ (YOL 1) VE PLOTLY ANİMASYON YARDIMCILARI
# ===========================================================================
def _confirmed_tracks_to_rows(t_val: float, fc: "FusionCenterIMM3") -> List[Dict[str, Any]]:
    """fc.tracks içindeki CONFIRMED durumundaki track'leri düz satır listesine çevirir."""
    rows: List[Dict[str, Any]] = []
    for gt in fc.tracks:
        if gt.status == "CONFIRMED":
            # 9D State vektörü: [x, vx, ax, y, vy, ay, z, vz, az]
            rows.append({
                "time": t_val,
                "global_track_id": gt.id,
                "x": float(gt.state[0, 0]),
                "y": float(gt.state[3, 0]),
                "z": float(gt.state[6, 0]),
                "vx": float(gt.state[1, 0]),
                "vy": float(gt.state[4, 0]),
                "vz": float(gt.state[7, 0]),
                "prob": float(gt.existence_prob),
                "n_sources": len(gt.source_radar_names),
                "source_radars": ", ".join(sorted(gt.source_radar_names))
            })
    return rows

def build_animated_map_figure(
    gt_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    max_frames: int = 250
) -> go.Figure:
    import plotly.express as px
    import numpy as np

    all_times = sorted(sensor_df["time"].dropna().unique())
    if not all_times:
        return go.Figure()

    if len(all_times) > max_frames:
        indices = np.linspace(0, len(all_times) - 1, max_frames).astype(int)
        frame_times = [all_times[i] for i in indices]
    else:
        frame_times = all_times

    # ==========================================
    # RENK VE ID SABİTLEME 
    # ==========================================
    fused_ids_all = sorted(fused_df["global_track_id"].dropna().astype(str).unique()) if not fused_df.empty else []
    fused_palette = px.colors.qualitative.Dark24
    fused_color_map = {tid: fused_palette[i % len(fused_palette)] for i, tid in enumerate(fused_ids_all)}

    sensor_ids_all = sorted(sensor_df["sensor"].dropna().astype(str).unique()) if not sensor_df.empty else []
    fallback_palette = px.colors.qualitative.Set2
    sensor_color_map = {}
    fallback_idx = 0
    for s in sensor_ids_all:
        s_up = s.upper()
        if "A" in s_up: sensor_color_map[s] = "#00FFFF" # Radar A: Turkuaz
        elif "B" in s_up: sensor_color_map[s] = "#FF00FF" # Radar B: Pembe
        elif "C" in s_up: sensor_color_map[s] = "#FFFF00" # Radar C: Sarı
        else:
            sensor_color_map[s] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1

    callsigns = []
    if gt_df is not None and not gt_df.empty:
        if "callsign" in gt_df.columns: callsigns = sorted(gt_df["callsign"].dropna().unique())
        elif "target" in gt_df.columns: callsigns = sorted(gt_df["target"].dropna().unique())
        else: callsigns = ["GT"]

    # ==========================================
    # KATMAN ÜRETİCİ (BÜG KORUMALI)
    # ==========================================
    def get_traces(t_val, is_base_frame=False):
        traces = []

        # 1. GROUND TRUTH
        for cs in callsigns:
            if gt_df is not None and not gt_df.empty:
                if "callsign" in gt_df.columns: grp = gt_df[(gt_df["time"] <= t_val) & (gt_df["callsign"] == cs)]
                elif "target" in gt_df.columns: grp = gt_df[(gt_df["time"] <= t_val) & (gt_df["target"] == cs)]
                else: grp = gt_df[gt_df["time"] <= t_val]
                grp = grp.sort_values("time")
            else:
                grp = pd.DataFrame()

            lats, lons, texts = [], [], []
            for _, r in grp.iterrows():
                la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                lats.append(la); lons.append(lo)
                texts.append(f"GT: {cs}<br>t={r['time']:.2f}s")

            # Veri yoksa Plotly çökmesin diye np.nan (Tanımsız/Görünmez) kullanıyoruz
            lat_line = lats if len(lats) > 1 else [np.nan]
            lon_line = lons if len(lons) > 1 else [np.nan]
            lat_pt = [lats[-1]] if lats else [np.nan]
            lon_pt = [lons[-1]] if lons else [np.nan]
            txt_pt = [texts[-1]] if texts else [""]

            if is_base_frame:
                traces.append(go.Scattermapbox(lat=lat_line, lon=lon_line, mode="lines", line=dict(width=2, color="#00CC96"), name=f"GT Rota {cs}", hoverinfo="skip"))
                traces.append(go.Scattermapbox(lat=lat_pt, lon=lon_pt, mode="markers", marker=dict(size=12, color="#00CC96"), name=f"GT Anlık {cs}", hovertext=txt_pt, hoverinfo="text"))
            else:
                traces.append(go.Scattermapbox(lat=lat_line, lon=lon_line))
                traces.append(go.Scattermapbox(lat=lat_pt, lon=lon_pt, hovertext=txt_pt))

        # 2. FUSED
        for tid in fused_ids_all:
            grp = fused_df[(fused_df["time"] <= t_val) & (fused_df["global_track_id"].astype(str) == tid)]
            grp = grp.sort_values("time")
            color = fused_color_map[tid]

            lats, lons, texts = [], [], []
            for _, r in grp.iterrows():
                la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                lats.append(la); lons.append(lo)
                prob_val = float(r["prob"]) if "prob" in r and pd.notna(r["prob"]) else 0.0
                texts.append(f"Fused ID: {tid}<br>t={r['time']:.2f}s<br>Prob: {prob_val:.3f}")

            lat_line = lats if len(lats) > 1 else [np.nan]
            lon_line = lons if len(lons) > 1 else [np.nan]
            lat_pt = [lats[-1]] if lats else [np.nan]
            lon_pt = [lons[-1]] if lons else [np.nan]
            txt_pt = [texts[-1]] if texts else [""]

            if is_base_frame:
                traces.append(go.Scattermapbox(lat=lat_line, lon=lon_line, mode="lines", line=dict(width=3, color=color), name=f"Fused İz {tid}", hoverinfo="skip", showlegend=False))
                traces.append(go.Scattermapbox(lat=lat_pt, lon=lon_pt, mode="markers", marker=dict(size=14, color=color), name=f"Fused {tid}", hovertext=txt_pt, hoverinfo="text", showlegend=False))
            else:
                traces.append(go.Scattermapbox(lat=lat_line, lon=lon_line))
                traces.append(go.Scattermapbox(lat=lat_pt, lon=lon_pt, hovertext=txt_pt))

        # 3. RADAR (KÜMÜLATİF, KAYBOLMAYAN NOKTALAR)
        radar_past = sensor_df[sensor_df["time"] <= t_val] if not sensor_df.empty else pd.DataFrame()
        if "is_clutter" in radar_past.columns:
            radar_past = radar_past[radar_past["is_clutter"] != True]

        for sensor_name in sensor_ids_all:
            s_grp = radar_past[radar_past["sensor"] == sensor_name]
            color = sensor_color_map[sensor_name]

            rad_lats, rad_lons, rad_texts = [], [], []
            for _, r in s_grp.iterrows():
                la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                rad_lats.append(la); rad_lons.append(lo)
                rad_texts.append(f"Sensor: {sensor_name}<br>Track: {r.get('local_track_id','?')}<br>t={r['time']:.2f}s")

            lat_rad = rad_lats if rad_lats else [np.nan]
            lon_rad = rad_lons if rad_lons else [np.nan]
            txt_rad = rad_texts if rad_texts else [""]

            if is_base_frame:
                traces.append(go.Scattermapbox(lat=lat_rad, lon=lon_rad, mode="markers", marker=dict(size=9, color=color), name=f"Radar: {sensor_name}", hovertext=txt_rad, hoverinfo="text"))
            else:
                # Sadece koordinat gönderiyoruz, marker komutunu frame içine koymuyoruz!
                traces.append(go.Scattermapbox(lat=lat_rad, lon=lon_rad, hovertext=txt_rad))

        return traces

    # ==========================================
    # FİGÜRÜ VE ANİMASYONU OLUŞTURMA
    # ==========================================
    fig = go.Figure(data=get_traces(frame_times[0], is_base_frame=True))

    frames = []
    for t in frame_times:
        frames.append(go.Frame(data=get_traces(t, is_base_frame=False), name=str(t)))
    fig.frames = frames

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center={"lat": ref_lat, "lon": ref_lon}, zoom=7.5),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left", x=0.0, xanchor="left", y=-0.05, yanchor="top",
            buttons=[
                dict(label="▶ Oynat", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                dict(label="⏸ Duraklat", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0, x=0.15, xanchor="left", y=-0.05, yanchor="top", len=0.85,
            currentvalue=dict(font=dict(size=14), prefix="⏱ Zaman: ", suffix=" s", visible=True, xanchor="right"),
            transition=dict(duration=0, easing="linear"), pad=dict(b=10, t=10),
            steps=[dict(args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))], label=f"{t:.1f}", method="animate") for t in frame_times]
        )]
    )
    return fig

    # =====================================================================
    # 3. FİGÜRÜ VE ANİMASYON KARELERİNİ OLUŞTUR
    # =====================================================================
    fig = go.Figure(data=get_traces(frame_times[0]))

    frames = []
    for t in frame_times:
        frames.append(go.Frame(data=get_traces(t), name=str(t)))
    fig.frames = frames

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center={"lat": ref_lat, "lon": ref_lon}, zoom=7.5),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left", x=0.0, xanchor="left", y=-0.05, yanchor="top",
            buttons=[
                dict(label="▶ Oynat", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                dict(label="⏸ Duraklat", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0, x=0.15, xanchor="left", y=-0.05, yanchor="top", len=0.85,
            currentvalue=dict(font=dict(size=14), prefix="⏱ Zaman: ", suffix=" s", visible=True, xanchor="right"),
            transition=dict(duration=0, easing="linear"), pad=dict(b=10, t=10),
            steps=[dict(args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
                        label=f"{t:.1f}", method="animate") for t in frame_times]
        )]
    )
    return fig

import pydeck as pdk  # Kütüphane listesinde yoksa en başa eklemeyi unutmayın!

def run_live_fusion_simulation(sensor_csv_path: str, gt_df_for_map: Optional[pd.DataFrame], map_update_every: int = 5, step_delay: float = 0.05):
    """Plotly yerine ultra-hafif PyDeck (WebGL) kullanarak haritayı canlı, izli ve çökmeden gösterir."""
    if not os.path.exists(sensor_csv_path):
        st.error(f"Radar CSV bulunamadı: `{sensor_csv_path}`")
        return

    sensor_df = pd.read_csv(sensor_csv_path)
    if "time" in sensor_df.columns:
        sensor_df["time"] = pd.to_numeric(sensor_df["time"], errors="coerce")
    sensor_df = sensor_df.sort_values("time")

    grouped = sensor_df.groupby("time", sort=True)
    time_values = sorted(grouped.groups.keys())
    total_steps = len(time_values)

    if total_steps == 0:
        st.warning("Radar verisinde işlenecek zaman adımı bulunamadı.")
        return

    if gt_df_for_map is not None and "lat" in gt_df_for_map.columns:
        ref_lat = gt_df_for_map["lat"].median()
        ref_lon = gt_df_for_map["lon"].median()
    else:
        ref_lat, ref_lon = 0.0, 0.0

    fc = FusionCenterIMM3(use_ci=True, verbose=False)

    st.markdown("#### 📡 Canlı Füzyon Simülasyonu Çalışıyor...")
    metrics_cols = st.columns(4)
    time_metric = metrics_cols[0].empty()
    meas_metric = metrics_cols[1].empty()
    track_metric = metrics_cols[2].empty()
    progress_metric = metrics_cols[3].empty()
    progress_bar = st.progress(0)
    
    map_placeholder = st.empty()
    fused_rows_accum: List[Dict[str, Any]] = []
    radar_rows_accum: List[Dict[str, Any]] = []

    def radar_bucket(sensor_name: str) -> str:
        """Radar adini A/B/C olarak kararlı şekilde siniflandirir."""
        s = str(sensor_name).strip().upper()
        if not s:
            return "OTHER"

        # "RADAR_A", "SENSOR-B", "C" gibi adlari token bazli ele al.
        tokens = s.replace("-", "_").split("_")
        for tok in reversed(tokens):
            if tok in {"A", "B", "C"}:
                return tok

        # Son karakter ile kodlanan adlar icin: "RDRA", "RADARB", "..._C" vb.
        if s.endswith("A"):
            return "A"
        if s.endswith("B"):
            return "B"
        if s.endswith("C"):
            return "C"
        return "OTHER"

    # Harita Kamerası (Siz dokunmadıkça sabit durur, siz dokununca sizin açınızı ezmez)
    view_state = pdk.ViewState(latitude=ref_lat, longitude=ref_lon, zoom=7.5, pitch=0)

    for step_idx, t_val in enumerate(time_values):
        group = grouped.get_group(t_val)
        measurements = []
        for _, row in group.iterrows():
            if row.get("is_clutter", False):
                continue
            z_val, vz_val = row.get("z", np.nan), row.get("vz", np.nan)
            
            state = np.array([
                row["x"], row.get("vx", 0.0), row["y"], row.get("vy", 0.0),
                0.0 if pd.isna(z_val) else float(z_val), 0.0 if pd.isna(vz_val) else float(vz_val),
            ]).reshape(6, 1)
            
            cov = measurement_cov_from_row(row)
            tq = row.get("track_quality", TQ_MAX)
            measurements.append({"state": state, "cov": cov, "tq": tq, "src": (row["sensor"], row.get("local_track_id", "0"))})

            # Radar ölçümlerini kümülatif göstermek için geçmişi biriktiriyoruz.
            la, lo = enu_to_latlon(row["x"], row["y"], ref_lat, ref_lon)
            sensor_name = str(row.get("sensor", "?")).strip()
            radar_rows_accum.append({
                "lat": la,
                "lon": lo,
                "sensor": sensor_name,
                "id": f"Radar: {sensor_name}",
            })
            
        fc.process_batch(t_val, measurements)
        step_rows = _confirmed_tracks_to_rows(t_val, fc)
        fused_rows_accum.extend(step_rows)

        # -------------------------------------------------------------
        # PYDECK KATMANLARI (GELİŞMİŞ GÖRÜNÜM)
        # -------------------------------------------------------------
        if step_idx % map_update_every == 0 or step_idx == total_steps - 1:
            gt_path_data, gt_curr_data = [], []
            fused_hist_data, fused_curr_data = [], []
            rad_a_hist_data, rad_b_hist_data, rad_c_hist_data, rad_other_hist_data = [], [], [], []

            # 1. GROUND TRUTH (Geçmiş Çizgisi ve Anlık Nokta)
            if gt_df_for_map is not None:
                gt_past = gt_df_for_map[gt_df_for_map["time"] <= t_val]
                if not gt_past.empty:
                    callsign_col = "callsign" if "callsign" in gt_past.columns else "target" if "target" in gt_past.columns else None
                    groups = gt_past.groupby(callsign_col) if callsign_col else [(0, gt_past)]
                    
                    for cs, grp in groups:
                        grp = grp.sort_values("time")
                        path_coords = []
                        for _, r in grp.iterrows():
                            la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                            path_coords.append([lo, la]) # PyDeck formatı [lon, lat] ister
                        
                        if len(path_coords) > 1:
                            gt_path_data.append({"path": path_coords, "id": f"GT Rota: {cs}"})
                            
                        # GT'nin o an bulunduğu uç noktası
                        last_la, last_lo = path_coords[-1][1], path_coords[-1][0]
                        gt_curr_data.append({"lat": last_la, "lon": last_lo, "id": f"GT Anlık: {cs}"})

            # 2. FUSED (Geçmiş İzleri ve Anlık Nokta)
            past_fused = fused_rows_accum[:-len(step_rows)] if len(step_rows) > 0 else fused_rows_accum
            for r in past_fused:
                la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                fused_hist_data.append({"lat": la, "lon": lo, "id": f"Fused İz: {r['global_track_id']}"})

            for r in step_rows:
                la, lo = enu_to_latlon(r["x"], r["y"], ref_lat, ref_lon)
                fused_curr_data.append({"lat": la, "lon": lo, "id": f"Fused Anlık: {r['global_track_id']}"})

            # 3. RADAR (Kümülatif geçmiş ölçümler, sensöre göre renklendirilmiş)
            for r in radar_rows_accum:
                bucket = radar_bucket(r.get("sensor", ""))
                if bucket == "A":
                    rad_a_hist_data.append(r)
                elif bucket == "B":
                    rad_b_hist_data.append(r)
                elif bucket == "C":
                    rad_c_hist_data.append(r)
                else:
                    rad_other_hist_data.append(r)

            # ---- KATMANLARI OLUŞTUR (Layers) ----
            layers = []

            # GT Rotası (Çizgi / PathLayer)
            if gt_path_data:
                layers.append(pdk.Layer(
                    "PathLayer", data=gt_path_data, get_path="path",
                    get_color="[0, 204, 150, 160]", width_scale=20, width_min_pixels=3, pickable=False
                ))

            # Fused Geçmiş İz (Şeffaf Küçük Noktalar)
            if fused_hist_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=fused_hist_data, get_position="[lon, lat]",
                    get_color="[239, 85, 59, 80]", get_radius=60, pickable=False
                ))

            # Radar A (Turkuaz)
            if rad_a_hist_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=rad_a_hist_data, get_position="[lon, lat]",
                    get_color="[0, 255, 255, 190]", get_radius=90, pickable=True
                ))

            # Radar B (Pembe)
            if rad_b_hist_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=rad_b_hist_data, get_position="[lon, lat]",
                    get_color="[255, 0, 255, 190]", get_radius=90, pickable=True
                ))

            # Radar C (Sarı)
            if rad_c_hist_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=rad_c_hist_data, get_position="[lon, lat]",
                    get_color="[255, 255, 0, 190]", get_radius=90, pickable=True
                ))

            # A/B/C dışı sensörler (varsayılan turuncu)
            if rad_other_hist_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=rad_other_hist_data, get_position="[lon, lat]",
                    get_color="[255, 161, 90, 180]", get_radius=90, pickable=True
                ))

            # GT Anlık (Büyük Yeşil)
            if gt_curr_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=gt_curr_data, get_position="[lon, lat]",
                    get_color="[0, 204, 150, 255]", get_radius=200, pickable=True
                ))

            # Fused Anlık (Büyük Kırmızı)
            if fused_curr_data:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=fused_curr_data, get_position="[lon, lat]",
                    get_color="[239, 85, 59, 255]", get_radius=200, pickable=True
                ))

            deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{id}"})
            map_placeholder.pydeck_chart(deck)

        # Metrikleri güncelle
        if step_idx % 5 == 0 or step_idx == total_steps - 1:
            time_metric.metric("⏱ Zaman", f"{t_val:.2f} s")
            meas_metric.metric("📥 Gelen Ölçüm", len(measurements))
            track_metric.metric("🎯 Aktif Hedef", len(step_rows))
            progress_metric.metric("İlerleme", f"{step_idx + 1} / {total_steps}")
            progress_bar.progress((step_idx + 1) / total_steps)

        if step_delay > 0:
            time.sleep(step_delay)

    st.success(f"✅ Canlı simülasyon tamamlandı! Toplam {len(fused_rows_accum)} fused nokta üretildi.")
    st.session_state["live_fused_history_df"] = pd.DataFrame(fused_rows_accum)
    st.session_state["live_gt_df"] = gt_df_for_map

# ===========================================================================
# HEDEF 5 – AKINCI 3B ANALIZI
# ===========================================================================
def render_target5_analysis_page():
    st.header("Hedef 5 – Spiral Tırmanış Analizi")
    st.caption(
        "Platform-temsili sentetik profil; gerçek operasyonel uçuş kaydı değildir. "
        "Senaryo mevcut radar modellerini ve IMM füzyonunu kullanır."
    )

    with st.form("target5_scenario_form"):
        first, second, third = st.columns(3)
        with first:
            total_speed = st.number_input("Toplam hız (m/s)", 40.0, 120.0, 77.0, 1.0)
            radius_start = st.number_input("Başlangıç yarıçapı (m)", 100.0, 3000.0, 500.0, 50.0)
            radius_end = st.number_input("Bitiş yarıçapı (m)", 200.0, 5000.0, 1800.0, 50.0)
            start_altitude = st.number_input("Başlangıç lokal Z (m)", -5000.0, 10000.0, 3000.0, 100.0)
            climb_rate = st.number_input("Tırmanış oranı (m/s)", 0.1, 20.0, 5.0, 0.5)
        with second:
            turn_count = st.number_input("Tur sayısı", 1, 10, 4, 1)
            turn_direction = st.selectbox("Dönüş yönü", ["ccw", "cw"], index=0)
            st.caption("Süre, gerçek 3B yay uzunluğu / toplam hız ile otomatik hesaplanır.")
        with third:
            center_x = st.number_input("Spiral merkez X (m)", -400000.0, 400000.0, 100000.0, 1000.0)
            center_y = st.number_input("Spiral merkez Y (m)", -400000.0, 400000.0, 100000.0, 1000.0)
            sample_rate = st.number_input("GT örnekleme (Hz)", 1.0, 50.0, 20.0, 1.0)
            noise_scale = st.number_input("Sensör gürültü çarpanı", 0.25, 3.0, 1.0, 0.05)
            detection_probability = st.slider("Temel detection probability", 0.10, 1.0, 0.90, 0.01)
            seed = st.number_input("Random seed", 0, 1000000, 42, 1)
        submitted = st.form_submit_button("Hedef 5 senaryosunu üret ve IMM füzyonunu çalıştır", type="primary")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(project_dir, "ground_truth_adsb_multi_target5.csv")
    sensor_path = os.path.join(project_dir, "radar_sensor_tracks_gercekci_target5.csv")
    fused_path = os.path.join(project_dir, "res_target5_imm.csv")
    debug_path = os.path.join(project_dir, "target5_imm_association_debug.csv")

    if submitted:
        params = Target5Parameters(
            sample_rate_hz=float(sample_rate),
            total_speed_mps=float(total_speed),
            turn_count=int(turn_count),
            radius_start_m=float(radius_start),
            radius_end_m=float(radius_end),
            climb_rate_mps=float(climb_rate),
            center_x_m=float(center_x),
            center_y_m=float(center_y),
            start_altitude_m=float(start_altitude),
            turn_direction=turn_direction,
        )
        try:
            with st.spinner("Ground truth ve radar ölçümleri üretiliyor..."):
                paths = write_target5_scenario(
                    project_dir,
                    params,
                    seed=int(seed),
                    noise_scale=float(noise_scale),
                    detection_probability=float(detection_probability),
                )
            with st.spinner("IMM füzyonu çalıştırılıyor..."):
                run_imm_fusion(
                    sensor_csv=str(paths["sensor"]),
                    output_csv=fused_path,
                    verbose=False,
                    diagnostics_csv=debug_path,
                )
            load_ground_truth.clear()
            load_sensor_data.clear()
            load_fused_data.clear()
            st.success("Hedef 5 senaryosu ve IMM çıktısı oluşturuldu.")
        except Exception as exc:
            st.error(f"Hedef 5 senaryosu üretilemedi: {exc}")
            return

    if not all(os.path.exists(path) for path in (gt_path, sensor_path, fused_path)):
        st.info("Analizi görmek için formu göndererek Hedef 5 senaryosunu üretin.")
        return

    gt_all = load_ground_truth(gt_path)
    sensor_all = load_sensor_data(sensor_path)
    fused_all = load_fused_data(fused_path)
    target5_gt = gt_all[gt_all["callsign"].astype(str) == TARGET5_CALLSIGN].copy()
    target5_sensor = sensor_all[
        sensor_all["callsign_true"].astype(str) == TARGET5_CALLSIGN
    ].copy()
    other_sensor = sensor_all[
        (sensor_all["callsign_true"].astype(str) != TARGET5_CALLSIGN)
        & (sensor_all["time"] >= float(target5_gt["time"].min()))
        & (sensor_all["time"] <= float(target5_gt["time"].max()))
    ].copy()
    target5_tracks, track_summary = associate_fused_tracks_to_target5(
        target5_gt, fused_all, max_match_distance_m=400.0, max_allowed_gap_s=5.0
    )
    matched = match_fused_to_target5(target5_gt, fused_all, max_match_distance_m=400.0)
    main_track_id = (
        str(track_summary.iloc[0]["track_id"]) if not track_summary.empty else None
    )
    track_view = st.radio(
        "Fused track görünümü",
        ["Ana track", "Tüm Hedef 5 track parçaları"],
        horizontal=True,
    )
    vertical_exaggeration = st.select_slider(
        "3B dikey ölçek",
        options=[1.0, 2.0, 5.0, 10.0],
        value=5.0,
        format_func=lambda value: f"{value}x" + (" (gerçek ölçek)" if value == 1 else ""),
    )
    if track_view == "Ana track" and main_track_id is not None:
        displayed_tracks = target5_tracks[
            target5_tracks["global_track_id"].astype(str) == main_track_id
        ].copy()
    else:
        displayed_tracks = target5_tracks.copy()
    evaluation_times = sensor_all["time"].dropna().unique().tolist()
    metrics = compute_target_specific_metrics(
        gt_all,
        fused_all,
        TARGET5_CALLSIGN,
        max_match_distance=400.0,
        evaluation_times=evaluation_times,
    )
    if matched.empty:
        target5_id_switches = 0
        target5_id_precision = 0.0
    else:
        assigned_ids = matched["global_track_id"].astype(str)
        target5_id_switches = int((assigned_ids != assigned_ids.shift()).iloc[1:].sum())
        identity_consistent_tp = max(int(metrics.get("total_tp", 0)) - target5_id_switches, 0)
        denominator = identity_consistent_tp + int(metrics.get("total_fp", 0)) + target5_id_switches
        target5_id_precision = identity_consistent_tp / denominator if denominator else 0.0
    target5_id_f1 = (
        2.0 * target5_id_precision * metrics.get("recall", 0.0)
        / (target5_id_precision + metrics.get("recall", 0.0))
        if target5_id_precision + metrics.get("recall", 0.0) > 0.0
        else 0.0
    )
    metrics["id_switches"] = target5_id_switches
    metrics["id_precision"] = target5_id_precision
    metrics["id_f1"] = target5_id_f1
    separations = target_separations(
        target5_gt,
        gt_all[gt_all["callsign"].astype(str) != TARGET5_CALLSIGN],
    )

    tab_2d, tab_3d, tab_results = st.tabs(
        ["2B Rota", "3B Rota", "Füzyon Sonuçları"]
    )
    # Önceki füzyon teşhisi korunur, ancak ayrı dördüncü sekme yerine sonuç
    # panelinin altında gösterilir.
    tab_diagnostics = tab_results

    diagnostic_measurements = pd.DataFrame()
    diagnostic_deletions = pd.DataFrame()
    if os.path.exists(debug_path):
        diagnostic_all = pd.read_csv(debug_path)
        diagnostic_measurements = diagnostic_all[
            (diagnostic_all.get("label", "").astype(str) == TARGET5_CALLSIGN)
            & (diagnostic_all.get("event_type", "") == "measurement")
        ].copy()
        diagnostic_deletions = diagnostic_all[
            (diagnostic_all.get("label", "").astype(str) == TARGET5_CALLSIGN)
            & (diagnostic_all.get("event_type", "") == "track_deleted")
        ].copy()

    prediction_count = int(
        target5_tracks.get(
            "is_prediction_only", pd.Series(False, index=target5_tracks.index)
        ).fillna(False).astype(bool).sum()
    )
    xy_gate_rejections = int(
        diagnostic_measurements.get(
            "source_xy_rejected", pd.Series(False, index=diagnostic_measurements.index)
        ).fillna(False).astype(bool).sum()
    )
    z_gate_rejections = int(
        diagnostic_measurements.get(
            "source_z_rejected", pd.Series(False, index=diagnostic_measurements.index)
        ).fillna(False).astype(bool).sum()
    )
    new_track_count = int(
        diagnostic_measurements.get(
            "created", pd.Series(False, index=diagnostic_measurements.index)
        ).fillna(False).astype(bool).sum()
    )
    accepted_associations = int(
        diagnostic_measurements.get(
            "accepted_stage", pd.Series(dtype=str)
        ).isin(["source_map", "hungarian", "revive"]).sum()
    )
    association_rejections = max(len(diagnostic_measurements) - accepted_associations, 0)
    gating_rejections = int(
        (
            diagnostic_measurements.get(
                "source_xy_rejected", pd.Series(False, index=diagnostic_measurements.index)
            ).fillna(False).astype(bool)
            | diagnostic_measurements.get(
                "source_z_rejected", pd.Series(False, index=diagnostic_measurements.index)
            ).fillna(False).astype(bool)
        ).sum()
    )
    longest_gap = float(track_summary["max_time_gap_s"].max()) if not track_summary.empty else 0.0
    time_match_rejections = int(track_summary["time_match_rejected"].sum()) if not track_summary.empty else 0
    diag_columns = tab_diagnostics.columns(5)
    for index, (label, value) in enumerate((
        ("GT nokta", len(target5_gt)),
        ("Radar ölçümü", len(target5_sensor)),
        ("İlişkili fused state", len(target5_tracks)),
        ("Prediction-only", prediction_count),
        ("Global ID", len(track_summary)),
        ("Ana track", main_track_id or "N/A"),
        ("ID switch", target5_id_switches),
        ("Kabul edilen association", accepted_associations),
        ("Association reddi", association_rejections),
        ("Gate reddi (tekil)", gating_rejections),
        ("Source-map XY ret", xy_gate_rejections),
        ("Source-map Z ret", z_gate_rejections),
        ("Yeni track", new_track_count),
        ("Track silinmesi", len(diagnostic_deletions)),
        ("En uzun fused boşluk", f"{longest_gap:.2f} s"),
        ("Zaman toleransı reddi", time_match_rejections),
    )):
        diag_columns[index % 5].metric(label, value)

    tab_diagnostics.caption(
        "Gate sayaçları aynı radar CSV'sinin, mevcut IMM eşikleri değiştirilmeden yapılan "
        "tanısal tekrar çalıştırmasından gelir. XY ve Z sayaçları source-map continuity "
        "denemesindeki gerçek Mahalanobis retleridir."
    )
    if not track_summary.empty:
        display_summary = track_summary.rename(columns={
            "track_id": "Global ID",
            "start_time": "Başlangıç (s)",
            "end_time": "Bitiş (s)",
            "point_count": "Nokta",
            "duration": "Süre (s)",
            "mean_3d_error": "Ort. 3B hata (m)",
            "max_3d_error": "Maks. 3B hata (m)",
            "prediction_only_count": "Prediction-only",
            "time_match_rejected": "Zaman toleransı reddi",
            "max_time_gap_s": "Maks. boşluk (s)",
        })
        tab_diagnostics.subheader("Hedef 5 track parçalanması")
        tab_diagnostics.dataframe(display_summary, use_container_width=True, hide_index=True)

    range_frames = [frame for frame in (target5_gt, target5_sensor, target5_tracks) if not frame.empty]
    if range_frames:
        range_data = pd.concat([frame[["x", "y", "z"]] for frame in range_frames], ignore_index=True)
        tab_diagnostics.dataframe(pd.DataFrame({
            "Eksen": ["X", "Y", "Z"],
            "Minimum (m)": [range_data[axis].min() for axis in ("x", "y", "z")],
            "Maksimum (m)": [range_data[axis].max() for axis in ("x", "y", "z")],
            "Aralık (m)": [range_data[axis].max() - range_data[axis].min() for axis in ("x", "y", "z")],
        }), use_container_width=True, hide_index=True)

    if not matched.empty:
        switch_mask = matched["global_track_id"].astype(str).ne(
            matched["global_track_id"].astype(str).shift()
        )
        switch_table = matched.loc[switch_mask, ["time", "global_track_id"]].copy()
        switch_table["Önceki ID"] = switch_table["global_track_id"].shift()
        tab_diagnostics.subheader("ID geçiş zamanları")
        tab_diagnostics.dataframe(switch_table.iloc[1:], use_container_width=True, hide_index=True)

    if diagnostic_measurements.empty:
        tab_diagnostics.warning(
            "Gate tanı CSV'si bulunamadı. Hedef 5 senaryosunu yeniden üretince tanı dosyası da oluşur."
        )
    else:
        tab_diagnostics.subheader("Association aşaması")
        stage_counts = diagnostic_measurements["accepted_stage"].fillna("rejected").value_counts()
        tab_diagnostics.dataframe(
            stage_counts.rename_axis("Aşama").reset_index(name="Ölçüm sayısı"),
            use_container_width=True,
            hide_index=True,
        )
        gate_rows = diagnostic_measurements[
            diagnostic_measurements.get(
                "source_xy_rejected", pd.Series(False, index=diagnostic_measurements.index)
            ).fillna(False).astype(bool)
            | diagnostic_measurements.get(
                "source_z_rejected", pd.Series(False, index=diagnostic_measurements.index)
            ).fillna(False).astype(bool)
        ]
        if not gate_rows.empty:
            tab_diagnostics.subheader("Gate dışına çıkan source-map denemeleri")
            tab_diagnostics.dataframe(
                gate_rows[[
                    "time", "sensor", "local_track_id", "source_map_track_id",
                    "source_d2_xy", "source_gate_xy", "source_d2_z", "source_gate_z",
                ]],
                use_container_width=True,
                hide_index=True,
            )

    if not matched.empty and "dominant_model" in matched:
        tab_diagnostics.subheader("Hedef 5 üzerinde etkin IMM modelleri")
        tab_diagnostics.dataframe(
            matched["dominant_model"].value_counts().rename_axis("Dominant model")
            .reset_index(name="Fused state"),
            use_container_width=True,
            hide_index=True,
        )

    one_second_edges = np.arange(
        float(target5_gt["time"].min()), float(target5_gt["time"].max()) + 1.01, 1.0
    )
    no_sensor_intervals = int(
        (np.histogram(target5_sensor["time"], bins=one_second_edges)[0] == 0).sum()
    ) if len(one_second_edges) > 1 else 0
    tab_diagnostics.subheader("Eksik görünen fused bölümlerin ayrımı")
    tab_diagnostics.dataframe(pd.DataFrame({
        "Neden": [
            "Sensör ölçümü olmayan 1 s aralık",
            "Yalnızca prediction olan state",
            "Track silinmesi",
            "Association başarısız / yeni track açıldı",
            "Ana-track görünüm filtresinde gizlenen state",
        ],
        "Sayı": [
            no_sensor_intervals,
            prediction_count,
            len(diagnostic_deletions),
            association_rejections,
            max(len(target5_tracks) - len(displayed_tracks), 0),
        ],
    }), use_container_width=True, hide_index=True)
    fig_xy = go.Figure()
    fig_xy.add_trace(go.Scatter(
        x=target5_gt["x"], y=target5_gt["y"], mode="lines",
        name="Hedef 5 Ground Truth", line=dict(color="#00CC96", width=4),
    ))
    for sensor_name, group in target5_sensor.groupby("sensor"):
        fig_xy.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers",
            marker=dict(size=5, opacity=0.5), name=f"Radar: {sensor_name}",
        ))
    for (track_id, segment_id), segment in displayed_tracks.groupby(
        ["global_track_id", "segment_id"], sort=False
    ):
        segment = segment.sort_values("time")
        fig_xy.add_trace(go.Scatter(
            x=segment["x"], y=segment["y"], mode="lines+markers",
            name=f"IMM {track_id} / parça {segment_id}",
            line=dict(width=3), marker=dict(size=4),
        ))
    fig_xy.add_trace(go.Scatter(
        x=[target5_gt.iloc[0]["x"], target5_gt.iloc[-1]["x"]],
        y=[target5_gt.iloc[0]["y"], target5_gt.iloc[-1]["y"]],
        mode="markers+text", text=["Başlangıç", "Bitiş"],
        textposition="top center", marker=dict(size=11), name="Başlangıç/Bitiş",
    ))
    fig_xy.update_layout(title="X-Y üstten görünüm", xaxis_title="X (m)", yaxis_title="Y (m)", height=600)
    fig_xy.update_yaxes(scaleanchor="x", scaleratio=1)
    tab_2d.plotly_chart(fig_xy, use_container_width=True)

    fig_xz = go.Figure()
    fig_xz.add_trace(go.Scatter(x=target5_gt["x"], y=target5_gt["z"], mode="lines", name="Ground Truth"))
    for (track_id, segment_id), segment in displayed_tracks.groupby(
        ["global_track_id", "segment_id"], sort=False
    ):
        segment = segment.sort_values("time")
        fig_xz.add_trace(go.Scatter(
            x=segment["x"], y=segment["z"], mode="lines+markers",
            name=f"IMM {track_id} / parça {segment_id}",
        ))
    for sensor_name, group in target5_sensor.groupby("sensor"):
        fig_xz.add_trace(go.Scatter(x=group["x"], y=group["z"], mode="markers", name=f"Radar: {sensor_name}", marker=dict(size=4, opacity=0.45)))
    fig_xz.update_layout(title="X-Z yandan görünüm", xaxis_title="X (m)", yaxis_title="İrtifa / lokal Z (m)", height=500)
    tab_2d.plotly_chart(fig_xz, use_container_width=True)

    fig_time_altitude = go.Figure()
    fig_time_altitude.add_trace(go.Scatter(
        x=target5_gt["time"], y=target5_gt["z"], mode="lines",
        name="Ground Truth irtifa",
    ))
    for (track_id, segment_id), segment in displayed_tracks.groupby(
        ["global_track_id", "segment_id"], sort=False
    ):
        segment = segment.sort_values("time")
        fig_time_altitude.add_trace(go.Scatter(
            x=segment["time"], y=segment["z"], mode="lines",
            name=f"IMM {track_id} / parça {segment_id}",
        ))
    fig_time_altitude.update_layout(
        title="Zaman–irtifa görünümü", xaxis_title="Zaman (s)", yaxis_title="Lokal Z (m)", height=450
    )
    tab_2d.plotly_chart(fig_time_altitude, use_container_width=True)

    origin_z = float(target5_gt.iloc[0]["z"])

    def z_display(values):
        values = pd.to_numeric(values, errors="coerce")
        return origin_z + (values - origin_z) * vertical_exaggeration

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=target5_gt["x"], y=target5_gt["y"], z=z_display(target5_gt["z"]),
        mode="lines", name="Hedef 5 Ground Truth",
        line=dict(color="#00CC96", width=6),
        customdata=target5_gt[["time", "speed", "heading", "z"]],
        hovertemplate="GT<br>t=%{customdata[0]:.2f}s<br>x=%{x:.1f}<br>y=%{y:.1f}<br>gerçek z=%{customdata[3]:.1f}<br>hız=%{customdata[1]:.1f}<br>heading=%{customdata[2]:.1f}°<extra></extra>",
    ))
    for sensor_name, group in target5_sensor.groupby("sensor"):
        fig3d.add_trace(go.Scatter3d(
            x=group["x"], y=group["y"], z=z_display(group["z"]), mode="markers",
            marker=dict(size=3, opacity=0.55), name=f"Radar: {sensor_name}",
            customdata=group[["time", "local_track_id", "z"]],
            hovertemplate="%{customdata[1]}<br>t=%{customdata[0]:.2f}s<br>x=%{x:.1f}<br>y=%{y:.1f}<br>gerçek z=%{customdata[2]:.1f}<extra></extra>",
        ))
    for sensor_name, group in other_sensor.groupby("sensor"):
        fig3d.add_trace(go.Scatter3d(
            x=group["x"], y=group["y"], z=z_display(group["z"]), mode="markers",
            marker=dict(size=2, opacity=0.15),
            name=f"Diğer hedef ölçümleri: {sensor_name}",
            visible="legendonly",
        ))
    for (track_id, segment_id), segment in displayed_tracks.groupby(
        ["global_track_id", "segment_id"], sort=False
    ):
        segment = segment.sort_values("time")
        fig3d.add_trace(go.Scatter3d(
            x=segment["x"], y=segment["y"], z=z_display(segment["z"]), mode="lines",
            line=dict(width=5), name=f"IMM {track_id} / parça {segment_id}",
            customdata=segment[["time", "global_track_id", "estimated_speed", "z"]],
            hovertemplate="Fused %{customdata[1]}<br>t=%{customdata[0]:.2f}s<br>x=%{x:.1f}<br>y=%{y:.1f}<br>gerçek z=%{customdata[3]:.1f}<br>hız=%{customdata[2]:.1f}<extra></extra>",
        ))
        prediction_mask = segment.get(
            "is_prediction_only", pd.Series(False, index=segment.index)
        ).fillna(False).astype(bool)
        for mask, label, symbol in (
            (~prediction_mask, "ölçüm güncellemesi", "circle"),
            (prediction_mask, "prediction-only", "diamond"),
        ):
            points = segment[mask]
            if not points.empty:
                fig3d.add_trace(go.Scatter3d(
                    x=points["x"], y=points["y"], z=z_display(points["z"]), mode="markers",
                    marker=dict(size=3, symbol=symbol),
                    name=f"{track_id} {label}", showlegend=True,
                ))
    fig3d.add_trace(go.Scatter3d(
        x=[target5_gt.iloc[0]["x"], target5_gt.iloc[-1]["x"]],
        y=[target5_gt.iloc[0]["y"], target5_gt.iloc[-1]["y"]],
        z=z_display(pd.Series([target5_gt.iloc[0]["z"], target5_gt.iloc[-1]["z"]])),
        mode="markers+text", text=["Başlangıç", "Bitiş"], textposition="top center",
        marker=dict(size=7, color=["#AB63FA", "#FFA15A"]), name="Başlangıç/Bitiş",
    ))
    fig3d.update_layout(
        height=720,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="İrtifa / lokal Z (m)", aspectmode="data"),
        legend=dict(orientation="h"), margin=dict(l=0, r=0, b=0, t=40),
    )
    tab_3d.plotly_chart(fig3d, use_container_width=True)

    tab_results.subheader("Hedef 5 sonuçları")
    route_segments = np.diff(target5_gt[["x", "y", "z"]].to_numpy(float), axis=0)
    route_length_m = float(np.linalg.norm(route_segments, axis=1).sum())
    duration_actual_s = float(target5_gt["time"].iloc[-1] - target5_gt["time"].iloc[0])
    center_x_actual = float(target5_gt.get(
        "spiral_center_x_m", pd.Series([target5_gt["x"].mean()])
    ).iloc[0])
    center_y_actual = float(target5_gt.get(
        "spiral_center_y_m", pd.Series([target5_gt["y"].mean()])
    ).iloc[0])
    radii = np.hypot(
        target5_gt["x"].to_numpy(float) - center_x_actual,
        target5_gt["y"].to_numpy(float) - center_y_actual,
    )
    spiral_summary = pd.DataFrame({
        "Spiral özelliği": [
            "Tur sayısı", "Merkez X", "Merkez Y", "Başlangıç yarıçapı",
            "Bitiş yarıçapı", "Hesaplanan süre", "Toplam 3B rota uzunluğu",
            "Başlangıç irtifası", "Bitiş irtifası", "Toplam irtifa kazancı",
            "Ortalama hız", "Minimum hız", "Maksimum hız", "Dikey görsel ölçek",
            "Fused track sayısı", "Ana fused track ID",
        ],
        "Değer": [
            str(int(round(float(target5_gt.get("spiral_theta_rad", pd.Series([8.0 * np.pi])).iloc[-1]) / (2.0 * np.pi)))),
            f"{center_x_actual:.2f} m", f"{center_y_actual:.2f} m",
            f"{radii[0]:.2f} m", f"{radii[-1]:.2f} m",
            f"{duration_actual_s:.3f} s", f"{route_length_m:.2f} m",
            f"{target5_gt['z'].iloc[0]:.2f} m", f"{target5_gt['z'].iloc[-1]:.2f} m",
            f"{target5_gt['z'].iloc[-1] - target5_gt['z'].iloc[0]:.2f} m",
            f"{target5_gt['speed'].mean():.3f} m/s",
            f"{target5_gt['speed'].min():.3f} m/s",
            f"{target5_gt['speed'].max():.3f} m/s",
            f"{vertical_exaggeration:.1f}x", str(len(track_summary)), main_track_id or "N/A",
        ],
    })
    tab_results.dataframe(spiral_summary, use_container_width=True, hide_index=True)
    metric_columns = tab_results.columns(5)
    headline = [
        ("Position RMSE", metrics.get("rmse_pos_m"), "m"),
        ("Velocity RMSE", metrics.get("rmse_vel_mps"), "m/s"),
        ("Precision", metrics.get("precision"), ""),
        ("Recall", metrics.get("recall"), ""),
        ("F1", metrics.get("f1_score"), ""),
        ("ID Precision", metrics.get("id_precision"), ""),
        ("ID F1", metrics.get("id_f1"), ""),
        ("MOTA", metrics.get("mota"), ""),
        ("ID switches", metrics.get("id_switches"), ""),
        ("Ortalama NEES", metrics.get("nees"), ""),
    ]
    for index, (label, value, suffix) in enumerate(headline):
        display = "N/A" if value is None or pd.isna(value) else f"{value:.3f}{' ' + suffix if suffix else ''}"
        metric_columns[index % 5].metric(label, display)

    axis_table = pd.DataFrame({
        "Metrik": ["X RMSE", "Y RMSE", "Z RMSE", "VX RMSE", "VY RMSE", "VZ RMSE"],
        "Değer": [metrics.get(key) for key in ("rmse_x_m", "rmse_y_m", "rmse_z_m", "rmse_vx_mps", "rmse_vy_mps", "rmse_vz_mps")],
        "Birim": ["m", "m", "m", "m/s", "m/s", "m/s"],
    })
    summary_left, summary_right = tab_results.columns(2)
    summary_left.dataframe(axis_table, use_container_width=True, hide_index=True)
    if matched.empty:
        error_summary = pd.DataFrame({"Metrik": ["Eşleşmiş fused nokta"], "Değer": [0]})
    else:
        errors = matched["position_error_3d"]
        error_summary = pd.DataFrame({
            "Metrik": ["Minimum 3B hata", "Ortalama 3B hata", "Maksimum 3B hata", "Track ID'leri", "Diğer hedeflere minimum ayrım"],
            "Değer": [
                f"{errors.min():.2f} m", f"{errors.mean():.2f} m", f"{errors.max():.2f} m",
                ", ".join(sorted(matched["global_track_id"].unique())),
                f"{min(separations.values()):.2f} m" if separations else "N/A",
            ],
        })
    summary_right.dataframe(error_summary, use_container_width=True, hide_index=True)
    if separations:
        tab_results.dataframe(
            pd.DataFrame([{"Hedef": target, "Minimum 3B ayrım (m)": distance} for target, distance in separations.items()]),
            use_container_width=True, hide_index=True,
        )

    if matched.empty:
        tab_results.warning("400 m değerlendirme kapısı içinde Hedef 5 ile eşleşen fused track bulunamadı.")
        return

    tab_results.subheader("Zamana bağlı hata ve durum grafikleri")
    chart1, chart2 = tab_results.columns(2)
    error_3d = go.Figure(go.Scatter(x=matched["time"], y=matched["position_error_3d"], name="3B pozisyon hatası"))
    error_3d.update_layout(xaxis_title="Zaman (s)", yaxis_title="Hata (m)")
    chart1.plotly_chart(error_3d, use_container_width=True)
    axis_error = go.Figure()
    for axis in ("x", "y", "z"):
        axis_error.add_trace(go.Scatter(x=matched["time"], y=matched[f"error_{axis}"], name=axis.upper()))
    axis_error.update_layout(xaxis_title="Zaman (s)", yaxis_title="Eksen hatası (m)")
    chart2.plotly_chart(axis_error, use_container_width=True)

    chart3, chart4 = tab_results.columns(2)
    velocity_error = go.Figure(go.Scatter(x=matched["time"], y=matched["velocity_error_3d"], name="Hız hatası"))
    velocity_error.update_layout(xaxis_title="Zaman (s)", yaxis_title="Hız hatası (m/s)")
    chart3.plotly_chart(velocity_error, use_container_width=True)
    altitude = go.Figure()
    altitude.add_trace(go.Scatter(x=matched["time"], y=matched["gt_z"], name="GT Z"))
    altitude.add_trace(go.Scatter(x=matched["time"], y=matched["z"], name="Fused Z"))
    altitude.update_layout(xaxis_title="Zaman (s)", yaxis_title="Lokal Z (m)")
    chart4.plotly_chart(altitude, use_container_width=True)

    chart5, chart6 = tab_results.columns(2)
    speed_fig = go.Figure()
    speed_fig.add_trace(go.Scatter(x=matched["time"], y=matched["gt_speed"], name="GT hız"))
    speed_fig.add_trace(go.Scatter(x=matched["time"], y=matched["estimated_speed"], name="Fused hız"))
    speed_fig.update_layout(xaxis_title="Zaman (s)", yaxis_title="Hız (m/s)")
    chart5.plotly_chart(speed_fig, use_container_width=True)
    heading_fig = go.Figure()
    heading_fig.add_trace(go.Scatter(x=matched["time"], y=matched["gt_heading"], name="GT heading"))
    heading_fig.add_trace(go.Scatter(x=matched["time"], y=matched["estimated_heading"], name="Fused heading"))
    heading_fig.update_layout(xaxis_title="Zaman (s)", yaxis_title="Heading (°)")
    chart6.plotly_chart(heading_fig, use_container_width=True)

    probability_columns = [column for column in matched if column.startswith("mode_prob_")]
    if probability_columns:
        probability_fig = go.Figure()
        for column in probability_columns:
            probability_fig.add_trace(go.Scatter(x=matched["time"], y=matched[column], name=column.replace("mode_prob_", "")))
        probability_fig.update_layout(title="IMM model olasılıkları", xaxis_title="Zaman (s)", yaxis_title="Olasılık")
        tab_results.plotly_chart(probability_fig, use_container_width=True)

    sigma_columns = [column for column in ("sigma_x_m", "sigma_y_m", "sigma_z_m") if column in matched]
    if sigma_columns:
        covariance_fig = go.Figure()
        for column in sigma_columns:
            covariance_fig.add_trace(go.Scatter(x=matched["time"], y=matched[column], name=column))
        covariance_fig.update_layout(title="Filtre 1σ konum belirsizliği", xaxis_title="Zaman (s)", yaxis_title="Sigma (m)")
        tab_results.plotly_chart(covariance_fig, use_container_width=True)

    status_left, status_right = tab_results.columns(2)
    one_second_bins = np.arange(0.0, float(target5_gt["time"].max()) + 1.0, 1.0)
    detected_bins = np.histogram(target5_sensor["time"], bins=one_second_bins)[0] > 0
    detection_fig = go.Figure(go.Scatter(x=one_second_bins[:-1], y=detected_bins.astype(int), mode="lines", name="Ölçüm var"))
    detection_fig.update_layout(title="Ölçüm alınan/alınmayan aralıklar", xaxis_title="Zaman (s)", yaxis=dict(tickvals=[0, 1], ticktext=["Yok", "Var"]))
    status_left.plotly_chart(detection_fig, use_container_width=True)
    id_fig = go.Figure(go.Scatter(x=matched["time"], y=matched["global_track_id"], mode="markers", name="Track ID"))
    id_fig.update_layout(title="Track ID zaman çizelgesi", xaxis_title="Zaman (s)", yaxis_title="Global track ID")
    status_right.plotly_chart(id_fig, use_container_width=True)


# ===========================================================================
# ANA UYGULAMA
# ===========================================================================
def main():
    st.set_page_config(page_title="Radar Fusion Streamlit", layout="wide")
    st.title("🔴 Radar Füzyon Analizi & Otonom Simülasyon Ekranı")

    mode = st.sidebar.radio(
        "Çalışma Modu",
        [
            "Offline (Kayıtlı Fused Veri)",
            "Hedef 5 – Spiral Tırmanış Analizi",
            "Otonom Playback Animasyon",
        ],
        index=0,
    )

    if mode == "Offline (Kayıtlı Fused Veri)":
        gt_file = st.sidebar.text_input("Ground truth CSV", "ground_truth_adsb_multi.csv")
        sensor_file = st.sidebar.text_input("Radar sensor CSV", "radar_sensor_tracks_gercekci.csv")
        fused_file = st.sidebar.text_input("Fused CSV", "res_real_adv.csv")

        if not os.path.exists(gt_file) or not os.path.exists(sensor_file) or not os.path.exists(fused_file):
            st.error("Lütfen tüm CSV dosyalarının bulunduğu konumu doğru girin.")
            return

        gt_df = load_ground_truth(gt_file)
        sensor_df = load_sensor_data(sensor_file)
        fused_df = load_fused_data(fused_file)

        show_measurement_details = st.sidebar.checkbox("Measurement details göster", value=False)
        data, ref_lat, ref_lon = build_map_data(gt_df, sensor_df, fused_df, show_measurement_details)

        st.markdown("Bu uygulama, her radarın farklı renkte ölçümlerini, ground truth rotasını ve füzyon sonuçlarını analiz etmenizi sağlar.")
        st.sidebar.markdown("### Filtreler")
        sensors = sorted(sensor_df["sensor"].unique())
        selected_sensors = st.sidebar.multiselect("Radar seç", sensors, default=sensors)

        min_time, max_time = float(data["time"].min()), float(data["time"].max())
        selected_time = st.sidebar.slider("Zaman aralığı", min_value=min_time, max_value=max_time, value=(min_time, max_time), step=1.0)

        filtered = data[(data["time"] >= selected_time[0]) & (data["time"] <= selected_time[1])]
        filtered = filtered[filtered["sensor_group"].isin(selected_sensors + ["Fused", "Ground Truth"])]

        fig = go.Figure()
        for sensor in sensors:
            sensor_points = filtered[filtered["sensor_group"] == sensor]
            if not sensor_points.empty:
                fig.add_trace(go.Scattermapbox(lat=sensor_points["lat"], lon=sensor_points["lon"], mode="markers", marker=dict(size=7), name=f"Radar: {sensor}", hovertext=sensor_points["hover"], hoverinfo="text"))

        fused_points = filtered[filtered["sensor_group"] == "Fused"]
        if not fused_points.empty:
            fused_palette = px.colors.qualitative.Dark24
            fused_ids = sorted(fused_points["id"].dropna().astype(str).unique())
            fused_color_map = build_color_map(fused_ids, fused_palette)
            for global_id in fused_ids:
                grp = fused_points[fused_points["id"].astype(str) == global_id].sort_values("time")
                fig.add_trace(go.Scattermapbox(lat=grp["lat"], lon=grp["lon"], mode="markers", marker=dict(size=10, color=fused_color_map[global_id], symbol="circle"), name=f"Fused: {global_id}", hovertext=grp["hover"], hoverinfo="text"))

        gt_points = filtered[filtered["sensor_group"] == "Ground Truth"]
        gt_colors = ["#808080", "#0066CC", "#FF6600", "#00FF00"]
        if not gt_points.empty:
            sorted_callsigns = sorted(gt_points["id"].unique())
            for idx, callsign in enumerate(sorted_callsigns):
                grp = gt_points[gt_points["id"] == callsign].sort_values("time")
                fig.add_trace(go.Scattermapbox(lat=grp["lat"], lon=grp["lon"], mode="lines+markers", line=dict(width=3, color=gt_colors[idx % len(gt_colors)]), marker=dict(size=6, color=gt_colors[idx % len(gt_colors)]), name=f"GT: {callsign}", hovertext=grp["hover"], hoverinfo="text"))

        fig.update_layout(mapbox=dict(style="open-street-map", center={"lat": ref_lat, "lon": ref_lon}, zoom=7), margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("3B Görünüm (Three.js Fusion Inspector)")
        gt_3d = gt_df[(gt_df["time"] >= selected_time[0]) & (gt_df["time"] <= selected_time[1])].copy()
        fused_3d = fused_df[(fused_df["time"] >= selected_time[0]) & (fused_df["time"] <= selected_time[1])].copy()
        components.html(build_threejs_html(gt_3d, fused_3d), height=850, scrolling=False)

    elif mode == "Hedef 5 – Spiral Tırmanış Analizi":
        render_target5_analysis_page()

    else:
        st.markdown(
            "Bu modda füzyon **arkada çok yüksek hızda** hesaplanır. Bittiği an altta Play butonlu tek parça bir harita çıkar."
        )

        if not FUSION_AVAILABLE:
            st.error("`fusion_imm3.py` dosyası bulunamadı. Lütfen modülün aynı klasörde olduğundan emin olun.")
            return

        live_sensor_file = st.sidebar.text_input("Canlı radar CSV", "radar_sensor_tracks_gercekci.csv")
        live_gt_file = st.sidebar.text_input("Ground truth CSV (harita referansı için)", "ground_truth_adsb_multi.csv")

        start_clicked = st.sidebar.button("🚀 Otonom Füzyonu Başlat", type="primary")

        gt_df_for_map = None
        if os.path.exists(live_gt_file):
            try:
                gt_df_for_map = load_ground_truth(live_gt_file)
            except Exception as e:
                st.sidebar.warning(f"GT dosyası okunamadı: {e}")

        if start_clicked:
            run_live_fusion_simulation(sensor_csv_path=live_sensor_file, gt_df_for_map=gt_df_for_map)
        else:
            st.info("Simülasyonu başlatmak için sol menüdeki **🚀 Otonom Füzyonu Başlat** butonuna basın.")

        live_fused_history_df = st.session_state.get("live_fused_history_df")
        live_gt_df = st.session_state.get("live_gt_df")

        if live_fused_history_df is not None and not live_fused_history_df.empty:
            st.markdown("---")
            st.subheader("3B Görünüm (Three.js Fusion Inspector) — Simülasyon Sonu Analizi")
            gt_3d_live = live_gt_df.copy() if live_gt_df is not None else pd.DataFrame(columns=["time", "x", "y", "z", "callsign"])
            components.html(build_threejs_html(gt_3d_live, live_fused_history_df), height=850, scrolling=False)

if __name__ == "__main__":
    main()
