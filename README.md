# Sensor Fusion Realtime

Gerçek radar track ölçümlerini JSON/UDP üzerinden alan, seçilebilir bir füzyon algoritmasıyla işleyen, fused track JSON mesajlarını yayımlayan ve basit bir 2B haritada gösteren ürün prototipi.

## Mimari

```text
Radar JSON -> UDP 7777 -> doğrulama -> seçilebilir füzyon
                                      |
                                      +-> UDP 8888 fused JSON
                                      +-> atomik state snapshot -> Streamlit 2B harita
```

Araştırma/benchmark ve veri üretimi bileşenleri bu pakette yoktur.

## Kurulum

```bash
git clone <repository-url>
cd sensor-fusion-realtime
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Test edilen ortam: Python 3.14, NumPy 2.5, pandas 3.0, SciPy 1.18, Streamlit 1.58, Plotly 6.8 ve PyYAML 6.0.

## Konfigürasyon

Tüm varsayılanlar [config/default.yaml](config/default.yaml) dosyasındadır. `fusion.algorithm` şu değerlerden biri olabilir:

- `basic_cv`: 6 durumlu sabit hız track-to-track füzyonu
- `advanced_ca`: 9 durumlu sabit ivme füzyonu; adaptive Q açılıp kapatılabilir
- `dual_imm`: XY için CV/CA/CT, Z için CV/Singer kullanan ayrık Dual-IMM

Her algoritmanın gerçek kod sabitleri `fusion.profiles.<algoritma>` altında bulunur. Algoritmayı değiştirmek için `fusion.algorithm` değerini düzenleyin; seçilen profile otomatik geçilir. Bilinmeyen parametreler sessizce yok sayılmaz, başlangıçta hata verir.

Desteklenen environment override örnekleri:

```powershell
$env:FUSION_ALGORITHM="advanced_ca"
$env:RADAR_UDP_PORT="7777"
$env:FUSED_UDP_PORT="8888"
python -m realtime.fusion_runtime
```

Farklı YAML için `python -m realtime.fusion_runtime --config config/my.yaml` veya `FUSION_CONFIG` kullanılabilir.

## Radar JSON şeması

```json
{
  "schema_version": 1,
  "message_type": "radar_measurement",
  "measurement_id": "radar_000001",
  "sensor_id": "RADAR_A",
  "source_track_id": "RADAR_A-T1",
  "timestamp": 1760000000.125,
  "sequence_number": 1,
  "coordinate_frame": "ENU",
  "position": {"x_m": 1000.0, "y_m": 500.0, "z_m": 3000.0},
  "velocity": {"vx_mps": 50.0, "vy_mps": 10.0, "vz_mps": 2.0},
  "covariance": null
}
```

`covariance`, verildiğinde `[x,vx,y,vy,z,vz]` sıralı 6x6 matristir. Geçersiz sürüm, tip, frame, JSON, eski/sırası bozuk veya duplicate paket servis durmadan reddedilir.

## Fused JSON şeması

```json
{
  "schema_version": 1,
  "message_type": "fused_track",
  "publish_timestamp": 1760000000.140,
  "state_timestamp": 1760000000.125,
  "track_id": "GT-0001",
  "track_status": "tentative",
  "coordinate_frame": "ENU",
  "position": {"x_m": 1000.0, "y_m": 500.0, "z_m": 3000.0},
  "velocity": {"vx_mps": 50.0, "vy_mps": 10.0, "vz_mps": 2.0},
  "fusion_metadata": {
    "update_type": "measurement_update",
    "filter_name": "Dual-IMM",
    "dominant_model": "XY:CV|Z:CV",
    "model_probabilities": {"CV": 0.7, "CA": 0.1, "CT": 0.2},
    "used_measurements": [{
      "measurement_id": "radar_000001",
      "sensor_id": "RADAR_A",
      "measurement_timestamp": 1760000000.125,
      "sequence_number": 1
    }]
  }
}
```

`used_measurements` association anında kaydedilir; sonradan konum yakınlığından tahmin edilmez.

## Çalıştırma

Terminal 1 — servis:

```bash
python -m realtime.fusion_runtime
```

Terminal 2 — 2B harita:

```bash
streamlit run visualization/realtime_map.py
```

Terminal 3 — CSV replay (`--speed 0` beklemeden, `--speed 2.0` iki kat hızlı, `--loop` sürekli):

```bash
python tools/radar_udp_replay.py --input data/radar_sensor_tracks_gercekci.csv --host 127.0.0.1 --port 7777 --speed 1.0
```

Terminal 4 — fused listener:

```bash
python tools/fused_udp_listener.py --host 0.0.0.0 --port 8888
```

Harita varsayılan olarak ground truth çizgilerini, son 20 radar noktasını, fused track çizgilerini ve güncel track marker'larını gösterir. Radar limiti 10/20/50/100 seçilebilir; track geçmişleri birbirine bağlanmaz.

## Paket verileri

`data/ground_truth_adsb_multi.csv` gerçek kolonları: `time,x,y,z,vx,vy,vz,callsign,lat,lon,alt_m,time_unix,target_id,target_name,speed,heading,turn_rate,climb_rate,spiral_center_x_m,spiral_center_y_m,spiral_radius_m,spiral_theta_rad`.

`data/radar_sensor_tracks_gercekci.csv` gerçek kolonları: `time,sensor,local_track_id,callsign_true,x,y,z,vx,vy,vz,track_quality,sigma_pos_m,sigma_vel_mps,is_clutter`. `callsign_true` ve `is_clutter` simülasyon değerlendirme etiketleridir; füzyon kararı bunları kullanmaz.

## Testler

```bash
python -m unittest discover -s tests -v
python -m realtime.fusion_runtime --help
python tools/radar_udp_replay.py --help
python tools/fused_udp_listener.py --help
```

Testler şema/duplicate kontrolü, üç algoritmanın factory üzerinden seçimi, YAML/env config, replay dönüşümü, UDP uçtan uca akış, measurement kimliği, kapanış ve görsel tampon ayrımını kapsar.

## Gerçek radar entegrasyonu

Radar gateway'i her track update için epoch saniye cinsinden `timestamp`, sensor bazında artan `sequence_number`, kalıcı `source_track_id` ve ENU Kartezyen durum üretmelidir. Ölçüm kovaryansı bilinmiyorsa `track_quality`, `sigma_pos_m` ve `sigma_vel_mps` kullanılabilir. UDP paket boyutu 65.507 byte'ı aşmamalıdır.

## Bilinen sınırlamalar

- Teslimat UDP olduğu için kayıp ve yeniden sıralama olabilir; uygulama duplicate/eski paketleri filtreler fakat yeniden iletim yapmaz.
- Yalnız ENU Kartezyen giriş desteklenir. Polar ölçüm için radar poz/oryantasyon kalibrasyonu bu pakete eklenmemiştir.
- Runtime tek worker ile sıralı işler; çok yüksek trafik için yatay ölçekleme veya broker gerekir.
- Snapshot dosyası tek makinedeki Streamlit arayüzü içindir; dağıtık kurulumda harici state katmanı gerekir.
- LICENSE varsayılan olarak kapalı/proprietary'dir; açık kaynak dağıtımı için proje sahibi uygun lisansla değiştirmelidir.
