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
python main.py --max-match-distance 400
```

## Hedef 5 – AKINCI-benzeri sentetik profil

`ground_truth_adsb_multi.csv`, idealize radar ve gerçekçi radar dosyalarında
`HEDEF_5` bulunur. Profil 77 m/s sabit toplam hızla, yaklaşık 1800 m başlangıç
yarıçaplı 360° yükselen bir spiral üretir; rota boyunca dönüş ve tırmanış sürer.

Hedef 5'e özel hafif analiz senaryosu ve IMM çıktısı:

- `ground_truth_adsb_multi_target5.csv`
- `radar_sensor_tracks_gercekci_target5.csv`
- `res_target5_imm.csv`

Arayüzü açmak için:

```bash
streamlit run streamlit_fusion.py
```

Ardından çalışma modu olarak **Hedef 5 – AKINCI 3B Füzyon Analizi** seçilir.
Form gönderilmedikçe ağır füzyon yeniden çalıştırılmaz.

Hedef 5 testleri:

```bash
python -m unittest test_target5_akinci.py
```

## Görselleştirme

```bash
streamlit run streamlit_fusion.py
```

## Proje yapısı

- `radar_sim.py`: Ground truth'tan idealize ve gerçekçi radar verisi üretir.
- `fusion_basic.py`: Sabit hızlı Basic füzyon algoritmasıdır.
- `fusion_advanced.py`: Sabit ivmeli ve CI destekli Advanced algoritmadır.
- `fusion_imm.py`: IMM tabanlı füzyon algoritmasıdır.
- `fusion_evaluation.py`: Performans metriklerini hesaplar.
- `main.py`: Yalnızca ADS-B tabanlı ana benchmark akışıdır.
- `target5_akinci.py`: Hedef 5 yörüngesi, ayrım kontrolü ve radar entegrasyonudur.
- `streamlit_fusion.py`: Sonuçları interaktif olarak görselleştirir.
