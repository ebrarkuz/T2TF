# Radar Fusion Project

## Stone Soup verisiyle karşılaştırma

Her iki radar simülatörü de aynı dört hedefli `ground_truth_adsb_multi.csv`
dosyasını kullanır. `radar_measurements_stonesoup.csv`, bu ground truth'un gerçek
Stone Soup radar ve ölçüm modellerinden geçirilmiş karşılığıdır. Üç algoritmayı
iki radar verisi üzerinde karşılaştırmak için:

```bash
python main.py
```

Stone Soup radar dosyasını yeniden üretmek için önce `python stone.py` çalıştırın.
Farklı dosyalar veya eşleştirme mesafesi de seçilebilir:

```bash
python main.py --source stonesoup --gt-csv my_truth.csv --radar-csv my_radar.csv --max-match-distance 200
```

Basic, Advanced ve IMM sonuçlarının iki veri kaynağını içeren ortak tablosu
`all_datasets_metrics.csv` olarak kaydedilir. Veri kaynağına özel raporlar
`adsb_metrics.csv` ve `stonesoup_metrics.csv`; hedef bazlı sonuçlar da ilgili
`*_target_metrics.csv` dosyalarıdır. İki veri kaynağının tüm hedef sonuçları
ayrıca `all_datasets_target_metrics.csv` dosyasında birleştirilir. Kendi
verinizdeki dört hedef için metrik tabloları terminalde ayrı ayrı gösterilir.
`is_clutter` sadece simülasyon etiketi kabul edilir ve
algoritmalara ön bilgi vermek için kullanılmaz. Recall ve MOTA tüm algoritmalar
için ortak radar tarama zamanlarında hesaplanır.

Yalnızca tek kaynağı çalıştırmak için `--source adsb` veya `--source stonesoup`
seçilebilir.

Proje yapısı:

- `fetch_ground_truth.py`: Ground truth verisini çeken ve oluşturan kod. Bu dosyaya dokunulmadı.
- `radar_sim.py`: Ground truth verisinden idealize ve gerçekçi radar sensör verileri üreten simülatör.
- `fusion_basic.py`: Basit füzyon metodu.
- `fusion_advanced.py`: Gelişmiş füzyon metodu (CI destekli).
- `fusion_evaluation.py`: Farklı kombinasyonları karşılaştırmak için metrik hesaplama.
- `main.py`: Grafik üretmeden, senaryoları çalıştırıp metrik tablosunu oluşturan sade ana dosya.
- `streamlit_fusion.py`: Fused sonuçları ve sensör verilerini interaktif olarak görselleştiren Streamlit uygulaması.

## Nasıl Kullanılır

1. Simülasyon verisini ve radar çıktısını üretmek için:

```bash
python main.py
```

2. Streamlit ile görselleştirmek için:

```bash
streamlit run streamlit_fusion.py
```

## Notlar

- `main.py` sadece metrik tablosu oluşturur; grafik üretmez.
- `streamlit_fusion.py` harita ve interaktif görselleştirme için ayrılmıştır.
- Aşağıdaki eski çıktı dosyaları ve görseller proje kökünden temizlendi.
