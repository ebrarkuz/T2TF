import os
import json
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def build_threejs_html(gt_df: pd.DataFrame, fused_df: pd.DataFrame) -> str:
    """
    Three.js 3D Track Inspector kodunu oluşturur ve Streamlit verilerini içine enjekte eder.
    """
    # DataFrame'leri CSV formatında metne dönüştür
    gt_csv_string = gt_df.to_csv(index=False)
    fused_csv_string = fused_df.to_csv(index=False)

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
      :root{
        --bg:#030508; 
        --panel:#0d131b;
        --panel-2:#101923;
        --border:#1c2733;
        --text:#d7e2ea;
        --text-dim:#6b7d8c;
        --accent-gt:#29d9c2;
        --accent-err:#ff5c5c;
        --axis-x:#ff6b6b;
        --axis-y:#5ce1a0;
        --axis-z:#ffd166;
      }
      *{box-sizing:border-box;}
      html,body{
        margin:0; padding:0; width:100%; height:100%; overflow:hidden;
        background:var(--bg); color:var(--text);
        font-family:'JetBrains Mono', monospace;
      }
      h1,h2,h3,.disp{ font-family:'Space Grotesk', sans-serif; }
      #app{ display:grid; grid-template-columns: 340px 1fr; grid-template-rows: 100vh; width:100vw; height:100vh; }

      #panel{ background:var(--panel); border-right:1px solid var(--border); overflow-y:auto; height:100vh; padding:18px 16px 40px 16px; }
      #panel::-webkit-scrollbar{ width:8px; }
      #panel::-webkit-scrollbar-thumb{ background:var(--border); border-radius:4px; }

      .brand{ display:flex; align-items:baseline; gap:8px; margin-bottom:4px; }
      .brand .dot{ width:9px; height:9px; border-radius:50%; background:var(--accent-gt); box-shadow:0 0 8px var(--accent-gt); }
      .brand h1{ font-size:16px; font-weight:700; margin:0; letter-spacing:.02em; }
      .sub{ color:var(--text-dim); font-size:11px; margin:0 0 20px 0; line-height:1.5; }

      .section{ margin-bottom:20px; padding-top:16px; border-top:1px solid var(--border); }
      .section:first-of-type{ border-top:none; padding-top:0; }
      .section h2{ font-size:10px; text-transform:uppercase; letter-spacing:.12em; color:var(--text-dim); margin:0 0 10px 0; font-weight:500; }

      .filebtn{ display:block; width:100%; text-align:left; padding:9px 10px; background:var(--panel-2); border:1px solid var(--accent-gt); border-radius:6px; color:var(--text); font-family:inherit; font-size:11.5px; margin-bottom:8px; }
      .filebtn .name{ color:var(--accent-gt); display:block; margin-top:2px; font-size:10.5px; }

      .hint{ font-size:10.5px; color:var(--text-dim); line-height:1.6; margin:6px 0 0 0; }
      code{ background:var(--panel-2); padding:1px 4px; border-radius:3px; color:var(--accent-gt); }

      .rowbtns{ display:flex; gap:6px; }
      .smallbtn{ flex:1; background:var(--panel-2); border:1px solid var(--border); color:var(--text); font-family:inherit; font-size:11px; padding:7px 4px; border-radius:6px; cursor:pointer; }
      .smallbtn:hover{ border-color:var(--accent-gt); }
      .smallbtn.active{ border-color:var(--accent-gt); color:var(--accent-gt); }

      #timeRow{ display:flex; align-items:center; gap:8px; margin-top:10px; }
      #timeSlider{ flex:1; accent-color:var(--accent-gt); }
      #frameLabel{ font-size:10.5px; color:var(--text-dim); min-width:64px; text-align:right; }

      .legend-item{ display:flex; align-items:center; gap:8px; padding:5px 0; font-size:11px; }
      .legend-item input[type=checkbox]{ accent-color:var(--accent-gt); }
      .swatch{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }
      .swatch.line{ border-radius:0; height:2px; }
      .legend-item .lbl{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

      .stat{ display:flex; justify-content:space-between; font-size:11px; padding:3px 0; }
      .stat .k{ color:var(--text-dim); }
      .stat .v{ font-weight:700; }
      .stat .v.axis-x{ color:var(--axis-x); }
      .stat .v.axis-y{ color:var(--axis-y); }
      .stat .v.axis-z{ color:var(--axis-z); }

      .match-row{ font-size:10.5px; padding:6px 8px; background:var(--panel-2); border-radius:6px; margin-bottom:6px; border-left:3px solid var(--accent-err); }
      .match-row .hdr{ color:var(--text); font-weight:700; margin-bottom:3px; }
      .match-row .axes span{ margin-right:10px; }
      #matchList{ max-height:220px; overflow-y:auto; }
      #matchList::-webkit-scrollbar{ width:6px; }
      #matchList::-webkit-scrollbar-thumb{ background:var(--border); border-radius:3px; }

      #viewport{ position:relative; height:100vh; overflow:hidden; }
      #canvasHost{ width:100%; height:100%; display:block; }
      #topbar{ position:absolute; top:14px; left:14px; right:14px; display:flex; justify-content:space-between; align-items:flex-start; pointer-events:none; }
      #viewbtns{ display:flex; gap:6px; pointer-events:all; }
      #viewbtns button{ background:rgba(13,19,27,.85); border:1px solid var(--border); color:var(--text); font-family:'Space Grotesk',sans-serif; font-size:11px; padding:7px 12px; border-radius:6px; cursor:pointer; backdrop-filter: blur(4px); }
      #viewbtns button:hover, #viewbtns button.active{ border-color:var(--accent-gt); color:var(--accent-gt); }
      
      #placeholder{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:10px; color:var(--text-dim); text-align:center; padding:40px; }
      #placeholder .big{ font-family:'Space Grotesk',sans-serif; font-size:20px; color:var(--text); }
    </style>
    </head>
    <body>
    
    <!-- Python verilerini buraya gizli element olarak yüklüyoruz -->
    <script id="gt_csv_data" type="text/csv">###GT_CSV_PLACEHOLDER###</script>
    <script id="fused_csv_data" type="text/csv">###FUSED_CSV_PLACEHOLDER###</script>

    <div id="app">
      <div id="panel">
        <div class="brand"><div class="dot"></div><h1>FUSION 3D INSPECTOR</h1></div>
        <p class="sub">Ground truth rota + fused track noktalarını 3B'de karşılaştır, eksen bazlı (X/Y/Z) hatayı incele.</p>

        <div class="section">
          <h2>1 · Veri Durumu</h2>
          <div class="filebtn">
            Ground Truth (Streamlit'ten Alındı)
            <span class="name" id="gtName">Hazırlanıyor...</span>
          </div>
          <div class="filebtn">
            Fused Tracks (Streamlit'ten Alındı)
            <span class="name" id="fusedName">Hazırlanıyor...</span>
          </div>
        </div>

        <div class="section">
          <h2>2 · Oynatım</h2>
          <div class="rowbtns">
            <button class="smallbtn" id="playBtn" disabled>▶ Oynat</button>
            <button class="smallbtn" id="speedBtn" disabled>1x</button>
          </div>
          <div id="timeRow">
            <input type="range" id="timeSlider" min="0" max="0" value="0" step="1" disabled>
            <span id="frameLabel">0 / 0</span>
          </div>
        </div>

        <div class="section">
          <h2>3 · Açı / Görünüm</h2>
          <div class="rowbtns">
            <button class="smallbtn" data-view="iso">İzometrik</button>
            <button class="smallbtn" data-view="top">Üstten (X-Y)</button>
          </div>
          <div class="rowbtns" style="margin-top:6px;">
            <button class="smallbtn" data-view="front">Önden (X-Z)</button>
            <button class="smallbtn" data-view="side">Yandan (Y-Z)</button>
          </div>
        </div>

        <div class="section">
          <h2>4 · Katmanlar</h2>
          <div class="legend-item"><input type="checkbox" id="toggleGtLine" checked><span class="swatch line" style="background:var(--accent-gt)"></span><span class="lbl">GT rotaları (Statik)</span></div>
          <div class="legend-item"><input type="checkbox" id="toggleFusedTrail" checked><span class="swatch line" style="background:#8ecae6"></span><span class="lbl">Zamanla Uzayan Fused İzleri</span></div>
          <div class="legend-item"><input type="checkbox" id="toggleErrLines" checked><span class="swatch line" style="background:#ffff00"></span><span class="lbl">Bağlantı ve Hata çizgileri</span></div>
          <div class="legend-item"><input type="checkbox" id="toggleAxisBars" checked><span class="swatch line" style="background:linear-gradient(90deg,var(--axis-x),var(--axis-y),var(--axis-z))"></span><span class="lbl">Eksen hata çubukları</span></div>
          <div class="legend-item"><input type="checkbox" id="toggleGrid" checked><span class="swatch" style="background:#172533"></span><span class="lbl">Zemin ızgarası</span></div>
        </div>

        <div class="section" id="tracksSection" style="display:none;">
          <h2>5 · Track'ler</h2>
          <div id="trackLegend"></div>
        </div>

        <div class="section" id="statsSection" style="display:none;">
          <h2>Kümülatif RMSE (0 → mevcut kare)</h2>
          <div class="stat"><span class="k">RMSE X (doğu)</span><span class="v axis-x" id="rmseX">–</span></div>
          <div class="stat"><span class="k">RMSE Y (kuzey)</span><span class="v axis-y" id="rmseY">–</span></div>
          <div class="stat"><span class="k">RMSE Z (irtifa)</span><span class="v axis-z" id="rmseZ">–</span></div>
          <div class="stat"><span class="k">RMSE toplam</span><span class="v" id="rmseTotal">–</span></div>
          <div class="stat"><span class="k">Eşleşen / GT / FP</span><span class="v" id="matchCounts">–</span></div>
        </div>

        <div class="section" id="frameMatchSection" style="display:none;">
          <h2>Bu Karedeki Eşleşmeler</h2>
          <div id="matchList"></div>
        </div>
      </div>

      <!-- ================= 3D VIEWPORT ================= -->
      <div id="viewport">
        <div id="canvasHost"></div>
        <div id="topbar">
          <div></div>
          <div id="viewbtns" style="display:none;">
            <button data-view="iso" class="active">İzometrik</button>
            <button data-view="top">Üstten</button>
            <button data-view="front">Önden</button>
            <button data-view="side">Yandan</button>
          </div>
        </div>
        <div id="placeholder">
          <div class="big">Yükleniyor...</div>
          <div>Veriler Streamlit'ten alınıyor.</div>
        </div>
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
      Papa.parse(csvString, {
        header:true, dynamicTyping:false, skipEmptyLines:true,
        complete: (res)=>cb(res.data, res.meta.fields)
      });
    }

    function buildGtTracks(rows, fields){
      const idCol = pickCol(fields, ['callsign','target','track_id','id']);
      const timeCol = pickCol(fields, ['time','t','timestamp']);
      const xCol = pickCol(fields, ['x']), yCol = pickCol(fields, ['y']);
      const zCol = pickCol(fields, ['z','alt','altitude','alt_m']);
      
      if(!idCol || !timeCol || !xCol || !yCol) return {};
      
      const tracks = {};
      rows.forEach(r=>{
        const id = String(r[idCol]);
        const t = safeNum(r[timeCol], null);
        const x = safeNum(r[xCol], null);
        const y = safeNum(r[yCol], null);
        
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
        const id = String(r[idCol]);
        const t = safeNum(r[timeCol], null);
        const x = safeNum(r[xCol], null);
        const y = safeNum(r[yCol], null);

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
      if(t<=arr[0].time) return arr[0];
      if(t>=arr[hi].time) return arr[hi];
      while(hi-lo>1){
        const mid=(lo+hi)>>1;
        if(arr[mid].time<=t) lo=mid; else hi=mid;
      }
      const a=arr[lo], b=arr[hi];
      const f = (t-a.time)/(b.time-a.time || 1);
      return { time: t, x: a.x + f*(b.x-a.x), y: a.y + f*(b.y-a.y), z: a.z + f*(b.z-a.z) };
    }

    let scene, camera, renderer, controls, canvasHost;
    let gtGroup, fusedGroup, errGroup, gridGroup, labelGroup;
    let sceneCenter = new THREE.Vector3(), sceneRadius = 100;

    const Z_SCALE = 50; 
    function toThree(p){ return new THREE.Vector3(p.x, p.z * Z_SCALE, p.y); } 

    function initScene(){
      canvasHost = document.getElementById('canvasHost');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x030508); 

      const w = canvasHost.clientWidth, h = canvasHost.clientHeight;
      camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 1e8); 
      renderer = new THREE.WebGLRenderer({antialias:true});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(w,h);
      canvasHost.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true; controls.dampingFactor = 0.08;

      scene.add(new THREE.AmbientLight(0xffffff, 0.8));
      const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1,2,1); scene.add(dl);

      gtGroup = new THREE.Group(); scene.add(gtGroup);
      fusedGroup = new THREE.Group(); scene.add(fusedGroup);
      errGroup = new THREE.Group(); scene.add(errGroup);
      gridGroup = new THREE.Group(); scene.add(gridGroup);
      labelGroup = new THREE.Group(); scene.add(labelGroup);

      window.addEventListener('resize', onResize);
      animate();
    }
    function onResize(){
      const w = canvasHost.clientWidth, h = canvasHost.clientHeight;
      camera.aspect = w/h; camera.updateProjectionMatrix();
      renderer.setSize(w,h);
    }
    function animate(){
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    function makeTextSprite(text, colorHex){
      const cnv = document.createElement('canvas'); cnv.width=256; cnv.height=64;
      const ctx = cnv.getContext('2d');
      ctx.font = 'bold 44px Space Grotesk, sans-serif';
      ctx.fillStyle = colorHex; ctx.textAlign='left'; ctx.textBaseline='middle';
      ctx.shadowColor = "black"; ctx.shadowBlur = 4; ctx.lineWidth = 4;
      ctx.strokeText(text, 10, 32); ctx.fillText(text, 10, 32);
      const tex = new THREE.CanvasTexture(cnv);
      const mat = new THREE.SpriteMaterial({map:tex, depthTest:false, transparent:true});
      const spr = new THREE.Sprite(mat);
      spr.scale.set(sceneRadius*0.18, sceneRadius*0.045, 1);
      return spr;
    }

    function buildStaticScene(){
      [gtGroup, fusedGroup, gridGroup, labelGroup, errGroup].forEach(g=>{ while(g.children.length) g.remove(g.children[0]); });

      const box = new THREE.Box3();
      Object.values(gtTracks).forEach(arr=>arr.forEach(p=>box.expandByPoint(toThree(p))));
      Object.values(fusedTracks).forEach(arr=>arr.forEach(p=>box.expandByPoint(toThree(p))));
      if(box.isEmpty()){ box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(100,100,100)); }
      box.getCenter(sceneCenter);
      sceneRadius = Math.max(box.getSize(new THREE.Vector3()).length()/2, 10);

      const gridSize = Math.max(sceneRadius*2.4, 100);
      const divisions = 30;
      const grid = new THREE.GridHelper(gridSize, divisions, 0x1c3140, 0x0a1622);
      grid.position.set(sceneCenter.x, box.min.y, sceneCenter.z);
      gridGroup.add(grid);

      const axisLen = sceneRadius*0.5;
      const origin = new THREE.Vector3(box.min.x, box.min.y, box.min.z);
      addAxisArrow(origin, new THREE.Vector3(1,0,0), axisLen, 0xff6b6b, 'X');
      addAxisArrow(origin, new THREE.Vector3(0,0,1), axisLen, 0x5ce1a0, 'Y');
      addAxisArrow(origin, new THREE.Vector3(0,1,0), axisLen, 0xffd166, 'Z');

      Object.entries(gtTracks).forEach(([id,arr])=>{
        const pts = arr.map(toThree);
        if(pts.length < 2) return;
        const path = new THREE.CurvePath();
        for(let i=0; i<pts.length-1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i+1]));
        const geo = new THREE.TubeGeometry(path, Math.min(pts.length * 2, 300), sceneRadius * 0.004, 6, false);
        const mat = new THREE.MeshBasicMaterial({color:0x29d9c2, transparent:true, opacity:0.4});
        const lineMesh = new THREE.Mesh(geo, mat);
        gtGroup.add(lineMesh);
      });

      let idx=0;
      Object.keys(fusedTracks).forEach(id=>{ trackColors[id] = colorForIndex(idx++); });
      fitCameraToScene('iso');
    }

    function addAxisArrow(origin, dir, len, colorHex, label){
      const arrow = new THREE.ArrowHelper(dir, origin, len, colorHex, len*0.12, len*0.06);
      gridGroup.add(arrow);
      const spr = makeTextSprite(label, '#'+colorHex.toString(16).padStart(6,'0'));
      spr.position.copy(origin).add(dir.clone().multiplyScalar(len*1.15));
      gridGroup.add(spr); 
    }

    let trackColors = {};
    const movingMeshes = {}; 
    const gtMovingMeshes = {}; 
    const gtDynamicLabels = {}; 

    function ensureMovingMesh(store, id, colorHex, radius, isFused){
      if(store[id]) return store[id];
      let geo = new THREE.SphereGeometry(radius, 16, 16);
      let mat;
      if (isFused) mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.9, roughness: 0.2 });
      else mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.3 });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      store[id] = mesh;
      return mesh;
    }

    function ensureGtLabel(id) {
      if(!gtDynamicLabels[id]) {
        const spr = makeTextSprite(id, '#29d9c2'); 
        scene.add(spr);
        gtDynamicLabels[id] = spr;
      }
      return gtDynamicLabels[id];
    }

    function clearGroupSafely(group) {
      while(group.children.length > 0) {
        const child = group.children[0];
        if(child.geometry) child.geometry.dispose();
        if(child.material) child.material.dispose();
        group.remove(child);
      }
    }

    function updateFrame(){
      const t = frameTimes[currentFrame];
      if(t===undefined) return;

      clearGroupSafely(errGroup);
      clearGroupSafely(fusedGroup);

      const gtNowState = {}; 
      const isGtVis = document.getElementById('toggleGtLine').checked;
      gtGroup.visible = isGtVis;

      Object.entries(gtTracks).forEach(([id,arr])=>{
        const st = interpGtAt(arr, t);
        const mesh = ensureMovingMesh(gtMovingMeshes, id, 0x29d9c2, sceneRadius*0.015, false);
        const label = ensureGtLabel(id);
        const visible = isGtVis && !!st;
        mesh.visible = visible; label.visible = visible;
        
        if(st){ 
          const p3 = toThree(st); 
          mesh.position.copy(p3);
          label.position.copy(p3).add(new THREE.Vector3(sceneRadius*0.05, sceneRadius*0.05, 0));
          gtNowState[id] = {data:st, three:p3}; 
        }
      });

      const fusedNow = [];
      const showTrail = document.getElementById('toggleFusedTrail').checked;

      Object.entries(fusedTracks).forEach(([id,arr])=>{
        const pastPts = arr.filter(p => p.time <= t);
        const color = trackColors[id] || '#8ecae6';
        const wantVisible = document.getElementById('trackVis_'+id) ? document.getElementById('trackVis_'+id).checked : true;
        const mesh = ensureMovingMesh(movingMeshes, id, color, sceneRadius*0.006, true);

        if(pastPts.length > 0 && wantVisible) {
          mesh.visible = true;
          const currentData = pastPts[pastPts.length - 1];
          const p3 = toThree(currentData);
          mesh.position.copy(p3);
          fusedNow.push({id, data:currentData, three:p3, color});

          if (showTrail && pastPts.length > 1) {
            const path = new THREE.CurvePath();
            let validSegments = 0;
            for(let i=0; i<pastPts.length-1; i++) {
              const pA = toThree(pastPts[i]);
              const pB = toThree(pastPts[i+1]);
              if(pA.distanceTo(pB) > 0.0001) { path.add(new THREE.LineCurve3(pA, pB)); validSegments++; }
            }
            if(validSegments > 0) {
              const geo = new THREE.TubeGeometry(path, Math.min(validSegments * 2, 200), sceneRadius * 0.0025, 6, false);
              const mat = new THREE.MeshBasicMaterial({color: color, transparent: true, opacity: 0.6});
              fusedGroup.add(new THREE.Mesh(geo, mat));
            }
          }
        } else {
          mesh.visible = false;
        }
      });

      const matches = [];
      fusedNow.forEach(f=>{
        let best=null, bestD=Infinity, bestId=null;
        Object.entries(gtNowState).forEach(([gid,g])=>{
          const d = Math.hypot(f.data.x-g.data.x, f.data.y-g.data.y, f.data.z-g.data.z);
          if(d<bestD){ bestD=d; best=g; bestId=gid; }
        });
        if(best) matches.push({fused:f, gt:best, gtId:bestId, dist:bestD, dx:f.data.x-best.data.x, dy:f.data.y-best.data.y, dz:f.data.z-best.data.z});
      });

      const showErr = document.getElementById('toggleErrLines').checked;
      const showBars = document.getElementById('toggleAxisBars').checked;
      
      matches.forEach(m=>{
        if(showErr){
          const geo = new THREE.BufferGeometry().setFromPoints([m.fused.three, m.gt.three]);
          const mat = new THREE.LineBasicMaterial({ color: m.fused.color, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending });
          errGroup.add(new THREE.Line(geo, mat));
        }
        if(showBars){
          const g = m.gt.three;
          addErrBar(g, new THREE.Vector3(1,0,0), m.dx, 0xff6b6b); 
          addErrBar(g, new THREE.Vector3(0,0,1), m.dy, 0x5ce1a0); 
          addErrBar(g, new THREE.Vector3(0,1,0), m.dz, 0xffd166); 
        }
      });

      updateFrameLabel(t);
      updateMatchPanel(matches);
      updateCumulativeStats();
    }

    function addErrBar(origin, dirUnit, signedLen, colorHex){
      const end = origin.clone().add(dirUnit.clone().multiplyScalar(signedLen));
      const geo = new THREE.BufferGeometry().setFromPoints([origin, end]);
      const mat = new THREE.LineBasicMaterial({color:colorHex, transparent:true, opacity:0.8});
      errGroup.add(new THREE.Line(geo, mat));
    }

    function fitCameraToScene(view){
      const d = sceneRadius*2.6;
      let pos, up=new THREE.Vector3(0,1,0);
      if(view==='top'){ pos = sceneCenter.clone().add(new THREE.Vector3(0.0001, d, 0)); up.set(0,0,-1); }
      else if(view==='front'){ pos = sceneCenter.clone().add(new THREE.Vector3(0, 0, d)); }
      else if(view==='side'){ pos = sceneCenter.clone().add(new THREE.Vector3(d, 0, 0.0001)); }
      else { pos = sceneCenter.clone().add(new THREE.Vector3(d*0.6, d*0.55, d*0.6)); } 
      camera.position.copy(pos);
      camera.up.copy(up);
      camera.lookAt(sceneCenter);
      controls.target.copy(sceneCenter);
      controls.update();
    }

    let sceneReady=false;
    function tryBuildScene(){
      if(!Object.keys(gtTracks).length || !Object.keys(fusedTracks).length) return;
      
      Object.values(gtDynamicLabels).forEach(lbl => scene.remove(lbl));
      for (let key in gtDynamicLabels) delete gtDynamicLabels[key];

      document.getElementById('placeholder').style.display='none';
      document.getElementById('viewbtns').style.display='flex';
      if(!sceneReady){ initScene(); sceneReady=true; }
      
      buildStaticScene();
      buildTrackLegend();

      frameTimes = Array.from(new Set(Object.values(fusedTracks).flat().map(r=>r.time))).sort((a,b)=>a-b);
      currentFrame = 0; 
      
      const slider = document.getElementById('timeSlider');
      slider.min=0; slider.max=Math.max(frameTimes.length-1,0); slider.value=0; slider.disabled=false;
      document.getElementById('playBtn').disabled=false;
      document.getElementById('speedBtn').disabled=false;
      document.getElementById('statsSection').style.display='block';
      document.getElementById('frameMatchSection').style.display='block';
      updateFrame();
    }

    function buildTrackLegend(){
      document.getElementById('tracksSection').style.display='block';
      const wrap = document.getElementById('trackLegend');
      wrap.innerHTML='';
      Object.keys(fusedTracks).forEach((id)=>{
        const color = trackColors[id];
        const row = document.createElement('div');
        row.className='legend-item';
        row.innerHTML = `<input type="checkbox" id="trackVis_${id}" checked> <span class="swatch" style="background:${color}"></span> <span class="lbl">Track ${id}</span>`;
        wrap.appendChild(row);
        row.querySelector('input').addEventListener('change', updateFrame);
      });
    }

    document.getElementById('timeSlider').addEventListener('input', e=>{ currentFrame = +e.target.value; updateFrame(); });
    document.getElementById('playBtn').addEventListener('click', ()=>{
      playing = !playing; document.getElementById('playBtn').textContent = playing ? '⏸ Duraklat' : '▶ Oynat';
      if(playing) startPlay(); else stopPlay();
    });
    document.getElementById('speedBtn').addEventListener('click', ()=>{
      speedIdx = (speedIdx+1)%SPEEDS.length; playSpeed = SPEEDS[speedIdx]; document.getElementById('speedBtn').textContent = playSpeed+'x';
      if(playing){ stopPlay(); startPlay(); }
    });
    function startPlay(){
      stopPlay();
      playTimer = setInterval(()=>{
        currentFrame++;
        if(currentFrame>=frameTimes.length){ currentFrame=0; playing = false; document.getElementById('playBtn').textContent = '▶ Oynat'; stopPlay(); return;}
        document.getElementById('timeSlider').value = currentFrame;
        updateFrame();
      }, 400/playSpeed);
    }
    function stopPlay(){ if(playTimer) clearInterval(playTimer); playTimer=null; }

    document.querySelectorAll('[data-view]').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        fitCameraToScene(btn.dataset.view);
        document.querySelectorAll('[data-view]').forEach(b=>b.classList.remove('active'));
        document.querySelectorAll(`[data-view="${btn.dataset.view}"]`).forEach(b=>b.classList.add('active'));
      });
    });
    ['toggleGtLine','toggleFusedTrail','toggleErrLines','toggleAxisBars','toggleGrid'].forEach(id=>{
      document.getElementById(id).addEventListener('change', ()=>{
        if(id==='toggleGrid'){ gridGroup.visible = document.getElementById(id).checked; }
        if(sceneReady) updateFrame();
      });
    });

    function updateFrameLabel(t){ document.getElementById('frameLabel').textContent = (currentFrame+1)+' / '+frameTimes.length+'  (t='+(+t.toFixed(2))+')'; }

    function updateMatchPanel(matches){
      const wrap = document.getElementById('matchList'); wrap.innerHTML='';
      if(!matches.length){ wrap.innerHTML = '<div class="hint">Bu karede eşleşme yok.</div>'; return; }
      matches.sort((a,b)=>b.dist-a.dist).forEach(m=>{
        const color = trackColors[m.fused.id] || '#8ecae6';
        const div = document.createElement('div'); div.className='match-row'; div.style.borderLeftColor = color;
        div.innerHTML = `<div class="hdr" style="color:${color}">Track ${m.fused.id} → GT ${m.gtId}</div>
          <div class="axes"> <span style="color:var(--axis-x)">dX ${signed(m.dx)}m</span> <span style="color:var(--axis-y)">dY ${signed(m.dy)}m</span> <span style="color:var(--axis-z)">dZ ${signed(m.dz)}m</span> </div>
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
        const fusedNow = [];
        Object.values(fusedTracks).forEach(arr=>{ const r = arr.find(p=>p.time===t); if(r) fusedNow.push(r); });
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

    // ==========================================
    // STREAMLIT ENTEGRASYONU - OTOMATİK BAŞLATMA
    // ==========================================
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

    # Verileri yer tutuculara enjekte ediyoruz (Python f-string kullanmıyoruz ki CSS curly bracket'leri sorun çıkarmasın)
    html_content = html_template.replace("###GT_CSV_PLACEHOLDER###", gt_csv_string)
    html_content = html_content.replace("###FUSED_CSV_PLACEHOLDER###", fused_csv_string)
    
    return html_content


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
        
        source_radars = str(row["source_radars"]) if "source_radars" in row else "Unknown"
        if source_radars == "nan" or source_radars.strip() == "":
            source_radars = "Unknown"
            
        prob_val = float(row["prob"]) if "prob" in row else 0.0

        if show_measurement_details:
            source_details = str(row["source_measurement_details"]) if "source_measurement_details" in row else "Unknown"
            if source_details == "nan" or source_details.strip() == "":
                source_details = source_radars
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
    
    if not os.path.exists(gt_file) or not os.path.exists(sensor_file) or not os.path.exists(fused_file):
        st.error("Lütfen tüm CSV dosyalarının bulunduğu konumu doğru girin.")
        return

    gt_df = load_ground_truth(gt_file)
    sensor_df = load_sensor_data(sensor_file)
    fused_df = load_fused_data(fused_file)

    if "source_radars" not in fused_df.columns:
        st.warning("Seçili fused CSV'de `source_radars` sütunu yok. Kaynak bilgisi `Unknown` görünebilir.")
    if "source_measurement_details" not in fused_df.columns:
        st.info("Eğer `source_measurement_details` sütunu da yoksa, sadece radar isimleri gösterilir.")

    show_measurement_details = st.sidebar.checkbox(
        "Measurement details göster", value=False,
        help="Kapalıyken bu alan hover ve tabloda gizlenir. Büyük CSV'lerde performansı artırır.",
    )

    data, ref_lat, ref_lon = build_map_data(gt_df, sensor_df, fused_df, show_measurement_details)

    st.markdown("Bu uygulama, her radarın farklı renkte ölçümlerini, ground truth rotasını ve füzyon sonuçlarını analiz etmenizi sağlar.")

    st.sidebar.markdown("### Filtreler")
    sensors = sorted(sensor_df["sensor"].unique())
    selected_sensors = st.sidebar.multiselect("Radar seç", sensors, default=sensors)
    
    gt_times = data[data["sensor_group"] == "Ground Truth"]["time"]
    if len(gt_times) > 0:
        gt_min_time, gt_max_time = float(gt_times.min()), float(gt_times.max())
    else:
        gt_min_time, gt_max_time = float(data["time"].min()), float(data["time"].max())
    
    min_time, max_time = float(data["time"].min()), float(data["time"].max())
    
    selected_time = st.sidebar.slider(
        "Zaman aralığı", min_value=min_time, max_value=max_time, value=(gt_min_time, gt_max_time), step=1.0
    )

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
    gt_colors = ["#808080", "#0066CC", "#FF6600"]
    if not gt_points.empty:
        sorted_callsigns = sorted(gt_points["id"].unique())
        for idx, callsign in enumerate(sorted_callsigns):
            grp = gt_points[gt_points["id"] == callsign].sort_values("time")
            fig.add_trace(go.Scattermapbox(lat=grp["lat"], lon=grp["lon"], mode="lines+markers", line=dict(width=3, color=gt_colors[idx % len(gt_colors)]), marker=dict(size=6, color=gt_colors[idx % len(gt_colors)]), name=f"GT: {callsign}", hovertext=grp["hover"], hoverinfo="text"))

    fig.update_layout(mapbox=dict(style="open-street-map", center={"lat": ref_lat, "lon": ref_lon}, zoom=7), margin={"r":0, "t":40, "l":0, "b":0}, height=700)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # YENİ 3D GÖRSELLEŞTİRME ENTEGRASYONU (Three.js Inspector)
    # -------------------------------------------------------------
    st.subheader("3B Görünüm (Three.js Fusion Inspector)")
    st.caption("Sol menüden 'Oynat' butonuna basarak hedefin zaman içindeki hareketini, füzyon noktalarının birikimini ve hata çubuklarını 3 boyutlu izleyebilirsiniz.")

    # Sürgüden seçilen zaman aralığındaki verileri filtrele
    gt_3d = gt_df[(gt_df["time"] >= selected_time[0]) & (gt_df["time"] <= selected_time[1])].copy()
    fused_3d = fused_df[(fused_df["time"] >= selected_time[0]) & (fused_df["time"] <= selected_time[1])].copy()
    
    # HTML Kodunu üret ve datayı içine yerleştir
    threejs_html = build_threejs_html(gt_3d, fused_3d)

    # Streamlit Components ile 800px yüksekliğinde ekrana bas
    components.html(threejs_html, height=850, scrolling=False)
    # -------------------------------------------------------------

    if not fused_df.empty:
        st.subheader("Seçili Zaman Aralığındaki Fusion Noktaları")
        istenen_sutunlar = ["time", "global_track_id", "x", "y", "z", "fused_tq", "prob", "n_sources", "source_radars"]
        if show_measurement_details: istenen_sutunlar.append("source_measurement_details")
        gosterilecek_sutunlar = [col for col in istenen_sutunlar if col in fused_df.columns]
        st.dataframe(fused_df[(fused_df["time"] >= selected_time[0]) & (fused_df["time"] <= selected_time[1])][gosterilecek_sutunlar].sort_values(["time", "global_track_id"]).reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()