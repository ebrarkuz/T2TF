# Sensor Fusion Realtime

Gerçek radar track ölçümlerini JSON/UDP üzerinden alan, seçilebilir füzyon algoritmasıyla işleyen ve tarayıcı gerektirmeyen yerel 2B masaüstü takip ekranında gösteren ürün prototipi.

## Mimari

```text
Radar JSON -> UDP 7777 -> doğrulama -> seçilebilir füzyon
                                      |-- UDP 8888: yalnız fused_track
                                      `-- UDP 8890: görsel telemetri
                                                     radar_measurement
                                                     fused_track
                                                     runtime_status

PySide6 masaüstü uygulaması <- UDP 8890
```

Masaüstü uygulaması füzyon servisinden ayrı süreçtir. Pencerenin kapanması UDP 7777 receiver'ını, filtre state'ini veya UDP 8888 yayınını durdurmaz. `8888` sözleşmesi değişmemiştir; harici tüketiciler yalnız `fused_track` almaya devam eder.

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

Doğrulanan ortam: Python 3.14, NumPy 2.5, pandas 3.0, SciPy 1.18, PyYAML 6.0, PySide6 6.11 ve pyqtgraph 0.14.

## Konfigürasyon

Varsayılanlar [config/default.yaml](config/default.yaml) dosyasındadır. Algoritma:

```yaml
fusion:
  algorithm: dual_imm  # basic_cv | advanced_ca | dual_imm
```

Gerçek algoritma parametreleri `fusion.profiles.<algoritma>` altındadır. Değişiklikler fusion runtime yeniden başlatıldığında uygulanır; çalışan filtre state'i sessizce değiştirilmez. Bilinmeyen parametreler başlangıçta hata verir.

Telemetri ve görselleştirme:

```yaml
visualization:
  telemetry_host: "127.0.0.1"
  telemetry_port: 8890
  max_visible_radar_measurements_per_sensor: 20
  max_fused_points_per_track: 2000
  track_archive_timeout_s: 5.0
  preserve_archived_tracks: true
  full_visual_history: false
  fused_display_interval_s: 1.0
  minimum_visible_tentative_points: 2
  refresh_interval_ms: 250
  max_line_gap_s: 1.0
```

Gercek zamanli association, ayni radar taramasini paket gelis sirasindan bagimsiz
islemek icin bounded micro-batch kullanir:

```yaml
runtime:
  micro_batch_enabled: true
  micro_batch_window_ms: 40
  timestamp_group_tolerance_ms: 5
  max_micro_batch_size: 100
  max_micro_batch_wait_ms: 75
  max_out_of_order_s: 0.1
```

Paketli CSV'de farkli taramalar arasindaki en kucuk pozitif zaman farki 10 ms
oldugu icin timestamp toleransi 5 ms secilmistir. Timeout hesabi monotonic saatle
yapilir; yeni timestamp grubu, boyut limiti, idle timeout ve shutdown tamponu
flush eder. Association cekirdegi gruptaki tum olcumleri tek `process_batch`
cagrisinda alir.

`RADAR_UDP_PORT`, `FUSED_UDP_PORT`, `TELEMETRY_UDP_PORT`, `FUSION_ALGORITHM` ve `FUSION_CONFIG` environment override olarak kullanılabilir.

## Radar JSON

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

`covariance`, verildiğinde `[x,vx,y,vy,z,vz]` sıralı 6x6 matristir. Geçersiz, duplicate, eski veya sırası bozuk mesajlar servisi çökertmeden reddedilir.

## Fused JSON

```json
{
  "schema_version": 1,
  "message_type": "fused_track",
  "publish_timestamp": 1760000000.140,
  "state_timestamp": 1760000000.125,
  "track_id": "GT-0001",
  "track_status": "tentative",
  "position": {"x_m": 1000.0, "y_m": 500.0, "z_m": 3000.0},
  "velocity": {"vx_mps": 50.0, "vy_mps": 10.0, "vz_mps": 2.0},
  "fusion_metadata": {
    "update_type": "measurement_update",
    "filter_name": "Dual-IMM",
    "dominant_model": "XY:CV|Z:CV",
    "used_measurements": [{
      "measurement_id": "radar_000001",
      "sensor_id": "RADAR_A",
      "measurement_timestamp": 1760000000.125,
      "sequence_number": 1
    }]
  }
}
```

`used_measurements` association sırasında kaydedilir; sonradan mesafeden tahmin edilmez.

## Çalıştırma

Terminal 1 — füzyon servisi:

```bash
python -m realtime.fusion_runtime
```

Terminal 2 — lokal masaüstü takip ekranı:

```bash
python -m visualization.desktop_app
```

Terminal 3 — CSV replay:

```bash
python tools/radar_udp_replay.py --input data/radar_sensor_tracks_gercekci.csv --host 127.0.0.1 --port 7777 --speed 1.0
```

`--speed 0` beklemeden, `--speed 2.0` iki kat hızlı, `--loop` sürekli replay yapar.

Terminal 4 — isteğe bağlı fused listener:

```bash
python tools/fused_udp_listener.py --host 0.0.0.0 --port 8888
```

## Masaüstü ekranı

PySide6 UI ana thread'de çalışır. UDP 8890 socket'i ayrı `QThread` içindeki worker tarafından okunur; Qt signal/slot ile doğrulanmış mesajlar bounded `DesktopState` deposuna aktarılır. `QTimer`, yapılandırılabilir aralıkta çizimi yeniler; her UDP mesajında repaint yapılmaz.

- Ground truth rotaları bir kez yüklenir ve çizgi olarak tutulur.
- Her `sensor_id`, kendi `deque(maxlen=20)` radar tamponunu kullanır; limit sensör başına 10/20/50/100 seçilebilir.
- Her sensör dinamik oluşturulan ayrı marker, renk ve legend serisiyle çizilir.
- Track rotaları füzyon state'inden bağımsız `active_tracks` ve `archived_tracks` görsel arşivlerinde tutulur.
- Beş saniye yeni fused mesajı almayan track yalnız görsel olarak arşivlenir; rotası ve son marker'ı ekranda kalır. Aynı ID yeniden mesaj alırsa geçmişiyle active duruma döner.
- Varsayılan `Normal gorunum`; active tentative ve active confirmed track'leri,
  archived confirmed geçmişlerini ve en az iki noktalı archived tentative
  hipotezleri gösterir. Tek noktalı archived tentative track'ler gizlenir.
- `Confirmed only`, `Active hypotheses` ve `Tum trackler / debug` filtreleri bulunur.
  Tentative track'ler soluk, içi boş ve kesikli kenarlı gösterilir.
- Her track kendi kalıcı `ScatterPlotItem` nokta serisinde tutulur ve varsayılan olarak en fazla 2000 state saklar.
- Limit aşılırsa başlangıç, bitiş, örneklenmiş eski noktalar ve en güncel noktalar korunur. `full_visual_history: true` downsampling'i kapatır.
- Fused geçmiş çizgi olarak birleştirilmez; her global track için saniyede bir ayrı nokta gösterilir.
- Güncel fused konum, geçmiş noktalarından daha büyük ayrı marker ile gösterilir.
- Radar tek bir `ScatterPlotItem` ile gösterilir.
- Track marker ve etiketleri yeniden kullanılır.
- Hover, mouse'a 14 piksel içindeki en yakın görünür noktayı bulup tek detay panelini günceller.
- Pause yalnız çizimi durdurur; UDP worker veri almaya devam eder.
- Geçmişi temizleme yalnız lokal çizim tamponunu temizler, füzyonu sıfırlamaz.

Ground truth bulunamazsa uygulama uyarı gösterir ve canlı telemetriyle çalışmayı sürdürür.

## Paket verileri

`data/ground_truth_adsb_multi.csv`: `time,x,y,z,vx,vy,vz,callsign,lat,lon,alt_m,time_unix,target_id,target_name,speed,heading,turn_rate,climb_rate,spiral_center_x_m,spiral_center_y_m,spiral_radius_m,spiral_theta_rad`.

`data/radar_sensor_tracks_gercekci.csv`: `time,sensor,local_track_id,callsign_true,x,y,z,vx,vy,vz,track_quality,sigma_pos_m,sigma_vel_mps,is_clutter`. Değerlendirme etiketleri füzyon kararında kullanılmaz.

## Windows EXE paketleme

PyInstaller çalışma zamanı için zorunlu değildir ve ana requirements dosyasına eklenmemiştir. İsteğe bağlı kurulumdan sonra:

```powershell
pip install pyinstaller
pyinstaller --name sensor-fusion-viewer --windowed visualization/desktop_app.py
```

Üretilen `dist/` ve `build/` çıktıları Git deposuna eklenmemelidir.

## Bilinen sınırlamalar

- UDP teslimatı kayıpsız değildir ve yeniden iletim yapmaz.
- Yalnız ENU Kartezyen giriş desteklenir; polar giriş için radar poz/oryantasyon kalibrasyonu gerekir.
- Tek telemetri UDP portunu aynı makinede aynı anda yalnız bir masaüstü uygulaması bind edebilir. Çoklu tüketici için multicast veya broker gerekir.
- Çok yüksek track/nokta sayısında nearest-point hover taraması ek uzamsal indeks gerektirebilir.
- Runtime tek worker ile sıralı işler.
- LICENSE varsayılan olarak kapalı/proprietary'dir.
