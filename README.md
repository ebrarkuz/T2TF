# Radar Fusion Project

Bu proje `ground_truth_adsb_multi.csv` verisinden radar ölçümleri üretir; Basic,
Advanced ve IMM füzyon algoritmalarını çalıştırır ve sonuçları karşılaştırır.

## Ana çalışma akışı

Mevcut `radar_sensor_tracks_gercekci.csv` dosyasını kullanmak için:

```bash
python main.py
```

Radar ölçümlerini önce güncel ground truth'tan yeniden üretmek için:

```bash
python main.py --regenerate-adsb
```

`main.py` yalnızca şu ADS-B tabanlı girdileri kullanır:

- `ground_truth_adsb_multi.csv`
- `radar_sensor_tracks_gercekci.csv`

Başlıca çıktılar:

- `res_real_basic.csv`
- `res_real_adv.csv`
- `res_real_imm.csv`
- `adsb_metrics.csv`
- `adsb_target_metrics.csv`

Eşleştirme mesafesi değiştirilebilir:

```bash
python main.py --max-match-distance 750
```

## Görselleştirme

```bash
streamlit run streamlit_fusion.py
```

## İsteğe bağlı Stone Soup akışı

Stone Soup ana çalışma akışından ayrılmıştır. Özellikle ihtiyaç duyulursa:

```bash
python stone.py
python stonesoup_benchmark.py
```

Bu komutlar `main.py` tarafından otomatik olarak çalıştırılmaz.

## Proje yapısı

- `radar_sim.py`: Ground truth'tan idealize ve gerçekçi radar verisi üretir.
- `fusion_basic.py`: Sabit hızlı Basic füzyon algoritmasıdır.
- `fusion_advanced.py`: Sabit ivmeli ve CI destekli Advanced algoritmadır.
- `fusion_imm.py`: IMM tabanlı füzyon algoritmasıdır.
- `fusion_evaluation.py`: Performans metriklerini hesaplar.
- `main.py`: Yalnızca ADS-B tabanlı ana benchmark akışıdır.
- `stonesoup_benchmark.py`: İsteğe bağlı, bağımsız Stone Soup benchmark akışıdır.
- `streamlit_fusion.py`: Sonuçları interaktif olarak görselleştirir.
