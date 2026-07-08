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

# Sonuçlar (Lineer Hız Modeli)

> **Not:** Aşağıdaki sonuçlar, global track'in gelecekteki zamana taşınması için **lineer hız modeli (linear velocity model)** kullanılan sürüme aittir. Bu yaklaşım doğrusal harekette yeterli performans gösterirken, **doğrusal olmayan (manevralı) hedef hareketlerinde** performansı belirgin şekilde düşmektedir. İlerleyen aşamalarda bu model yerine **Kalman Filter tabanlı bir hareket modeli** (gerekirse EKF/UKF) kullanılması planlanmaktadır.

## Genel Performans

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.986 | 0.542 | 0.699 | 0.534 | 0 | 209.314 | 3.437 | 43823.990 |
| İdealize + Gelişmiş (CI) | 0.954 | 0.678 | **0.793** | **0.645** | 2 | 212.356 | 8.787 | 13.087 |
| Gerçekçi + Temel | 0.961 | 0.389 | 0.554 | 0.373 | 0 | 234.396 | 6.474 | 54983.422 |
| Gerçekçi + Gelişmiş (CI) | 0.913 | 0.642 | **0.754** | **0.581** | 2 | 304.165 | 9.535 | **8.776** |
| Gerçekçi + Gelişmiş (No-CI) | 0.894 | 0.621 | 0.733 | 0.547 | 2 | 301.519 | 8.695 | 41.686 |

---

# Hedef Bazlı Sonuçlar

## HEDEF_2

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.473934 | 0.781250 | 0.589971 | -0.085938 | 0 | 151.855154 | 1.904388 | 23063.614547 |
| İdealize + Gelişmiş (CI) | 0.444436 | 0.948068 | **0.605177** | -0.237055 | 0 | **133.589326** | 4.134471 | **6.362326** |
| Gerçekçi + Temel | 0.441176 | 0.535714 | 0.483871 | -0.142857 | 0 | 253.906255 | 4.939203 | 64492.782129 |
| Gerçekçi + Gelişmiş (CI) | 0.427146 | 0.901866 | 0.579721 | -0.307646 | 0 | 282.510699 | 9.172975 | **6.815039** |
| Gerçekçi + Gelişmiş (No-CI) | 0.423834 | 0.883209 | 0.572795 | -0.317438 | 0 | 266.259632 | 9.092262 | 40.613454 |

---

## HEDEF_3_MANEVRA

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.118483 | 0.195312 | 0.147493 | -1.257812 | 0 | **426.818002** | **7.641353** | 182231.997104 |
| İdealize + Gelişmiş (CI) | 0.116712 | **0.248969** | **0.158924** | -1.635253 | 0 | 484.523851 | 16.541492 | **45.237958** |
| Gerçekçi + Temel | 0.068627 | 0.083333 | 0.075269 | -1.047619 | 0 | **302.865536** | 12.263045 | 91877.915126 |
| Gerçekçi + Gelişmiş (CI) | 0.096759 | 0.204295 | 0.131321 | -1.702789 | 0 | 544.591253 | 21.162968 | **37.678364** |
| Gerçekçi + Gelişmiş (No-CI) | 0.088432 | 0.184280 | 0.119513 | -1.715296 | 0 | 599.239531 | 17.842797 | 56.421231 |

> **Gözlem:** Manevra yapan hedefte tüm yöntemlerin performansı belirgin şekilde düşmektedir. Bunun temel nedeni, global track'in geleceğe taşınmasında kullanılan **lineer hız modelinin** doğrusal olmayan hareketleri doğru şekilde tahmin edememesidir.

---

## Hedef_1

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.393365 | 0.648438 | 0.489676 | -0.351562 | 0 | 164.741288 | **2.764105** | 27147.332325 |
| İdealize + Gelişmiş (CI) | 0.393241 | **0.838857** | **0.535465** | -0.455476 | 0 | **140.266627** | 12.914202 | **16.749592** |
| Gerçekçi + Temel | **0.450980** | 0.547619 | 0.494624 | -0.119048 | 0 | 171.896554 | 6.456980 | 29590.117944 |
| Gerçekçi + Gelişmiş (CI) | 0.389507 | **0.822396** | **0.528638** | -0.466586 | 0 | 236.051939 | 12.184748 | **14.287291** |
| Gerçekçi + Gelişmiş (No-CI) | 0.382527 | 0.797131 | 0.516970 | -0.489594 | 0 | 221.304881 | 12.420380 | 66.795015 |


# Sonuçlar (Sabit İvmeli Hareket Modeli)

> **Not:** Aşağıdaki sonuçlar, global track'in gelecekteki zamana taşınması için **sabit ivmeli hareket modeli (Constant Acceleration Motion Model)** kullanılan sürüme aittir. Bu model, lineer hız modeline kıyasla manevralı hedeflerin hareketini daha iyi tahmin etmeyi amaçlamaktadır.

## Genel Performans

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.986 | 0.542 | 0.699 | 0.534 | 0 | 209.314 | 3.437 | 43823.990 |
| İdealize + Gelişmiş (CI) | 0.918 | 0.809 | **0.860** | **0.736** | 4 | 250.208 | 11.921 | 5.943 |
| Gerçekçi + Temel | 0.961 | 0.389 | 0.554 | 0.373 | 0 | 234.396 | 6.474 | 54983.422 |
| Gerçekçi + Gelişmiş (CI) | 0.907 | 0.614 | **0.733** | **0.551** | 0 | 320.304 | 16.165 | **3.217** |
| Gerçekçi + Gelişmiş (No-CI) | 0.904 | 0.590 | 0.714 | 0.527 | 4 | 307.034 | 25.375 | 16.543 |

---

# Hedef Bazlı Sonuçlar

## HEDEF_2

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.473934 | 0.781250 | 0.589971 | -0.085938 | 0 | 151.855154 | 1.904388 | 23063.614547 |
| İdealize + Gelişmiş (CI) | 0.359630 | **0.950800** | 0.521869 | -0.742228 | 0 | **114.535142** | 6.624910 | 3.148911 |
| Gerçekçi + Temel | 0.441176 | 0.535714 | 0.483871 | -0.142857 | 0 | 253.906255 | 4.939203 | 64492.782129 |
| Gerçekçi + Gelişmiş (CI) | 0.364328 | **0.739605** | 0.488180 | -0.550840 | 0 | 250.946159 | 14.962737 | **2.248830** |
| Gerçekçi + Gelişmiş (No-CI) | 0.375775 | 0.736177 | 0.497570 | -0.486734 | 0 | **199.690345** | 25.379642 | 12.754767 |

---

## HEDEF_3_MANEVRA

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.118483 | 0.195312 | 0.147493 | -1.257812 | 0 | **426.818002** | **7.641353** | 182231.997104 |
| İdealize + Gelişmiş (CI) | 0.240267 | **0.635225** | **0.348658** | -1.373378 | 0 | 447.865979 | 20.273433 | 14.861470 |
| Gerçekçi + Temel | 0.068627 | 0.083333 | 0.075269 | -1.047619 | 0 | **302.865536** | 12.263045 | 91877.915126 |
| Gerçekçi + Gelişmiş (CI) | 0.214247 | **0.434932** | **0.287079** | -1.160186 | 0 | 514.256092 | 21.246833 | **6.110198** |
| Gerçekçi + Gelişmiş (No-CI) | 0.194153 | 0.380362 | 0.257081 | -1.198364 | 0 | 558.468564 | 30.134460 | 26.359635 |

> **Gözlem:** Sabit ivmeli hareket modeli, özellikle manevra yapan hedeflerde lineer hız modeline göre **Recall** ve **F1 Score** metriklerinde belirgin bir iyileşme sağlamaktadır. Buna karşın, daha karmaşık hareket modeli nedeniyle bazı senaryolarda **RMSE Konum** ve **RMSE Hız** değerlerinde artış gözlenmiştir.

---

## Hedef-1

| Füzyon Konfigürasyonu | Precision | Recall | F1 Score | MOTA | ID Switch | RMSE Konum (m) | RMSE Hız (m/s) | NEES |
|------------------------|---------:|------:|--------:|-----:|----------:|---------------:|---------------:|-----:|
| İdealize + Temel | 0.393365 | 0.648438 | 0.489676 | -0.351562 | 0 | 164.741288 | **2.764105** | 27147.332325 |
| İdealize + Gelişmiş (CI) | 0.318073 | **0.840930** | 0.461564 | -0.961968 | 0 | **117.300694** | 6.956559 | **2.352299** |
| Gerçekçi + Temel | **0.450980** | 0.547619 | **0.494624** | -0.119048 | 0 | 171.896554 | 6.456980 | 29590.117944 |
| Gerçekçi + Gelişmiş (CI) | 0.329406 | **0.668711** | 0.441386 | -0.692628 | 0 | 203.711025 | 19.608341 | **3.847976** |
| Gerçekçi + Gelişmiş (No-CI) | 0.334641 | 0.655591 | 0.443104 | -0.647905 | 0 | **171.698914** | 25.183260 | 21.094487 |
