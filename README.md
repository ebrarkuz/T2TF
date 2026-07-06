# Radar Fusion Project

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
