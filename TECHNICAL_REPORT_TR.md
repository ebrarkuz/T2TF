# Radar Füzyon Projesi Teknik Raporu

## 1. Amaç ve Kapsam

Bu projede amaç, ADS-B tabanlı gerçek uçuş verisini referans alarak çoklu radar senaryoları üretmek, bu sensör verilerini iki farklı füzyon yaklaşımı ile birleştirmek ve sonuçları metriklerle karşılaştırmaktır. Çalışma zinciri dört ana aşamadan oluşur:

1. Ground truth verisinin hazırlanması
2. İdealize ve gerçekçi radar sensör verilerinin üretilmesi
3. Temel ve gelişmiş track-to-track füzyonun çalıştırılması
4. Performans metriklerinin hesaplanması ve senaryoların karşılaştırılması

Bu yapı kod tarafında özellikle [main.py](main.py#L1) dosyasında orkestre edilir.

## 2. Veri Kaynağı ve Ground Truth

Projedeki referans veri, OpenSky Network üzerinden alınan ADS-B iz verisidir. [fetch_ground_truth.py](fetch_ground_truth.py#L1) dosyası; API ile uçuş verisini çekmek, zaman ekseninde örneklemek ve CSV biçiminde saklamak için tasarlanmıştır. Simülasyon ve füzyon sürecinde bu veri, gerçek hedef hareketinin doğrulama zemini olarak kullanılır.

Ground truth dosyası, hedeflerin zamana bağlı konum ve hız bileşenlerini içerir. Füzyon algoritmalarında tahmin edilen global track’ler ile bu referans arasındaki farklar ölçülerek başarı değerlendirilir.

## 3. Aşama 1: Veri Üretimi ve Radar Simülasyonu

Veri üretim katmanı [radar_sim.py](radar_sim.py#L1) içinde yer alır. Bu modül, aynı ground truth üzerinden iki ayrı sensör çıktısı üretir:

- `radar_sensor_tracks_idealize.csv`
- `radar_sensor_tracks_gercekci.csv`

### 3.1 Radar Modeli

Projede üç sanal radar tanımlıdır: RADAR_A, RADAR_B ve RADAR_C. Her radar için ayrı yeniden ziyaret süresi, gecikme, algılama olasılığı, bias, bias drift ve clutter üretim parametreleri vardır. Böylece sensörler aynı hedefi farklı zamanlarda, farklı kalitede ve farklı sistematik hatalarla gözler.

Radar sensör kalitesi, `track_quality (TQ)` üzerinden modellenir. `TQ` değeri yükseldikçe konum ve hız gürültüsü düşer. Kodda bu dönüşüm `tq_to_sigma_pos()` ve `tq_to_sigma_vel()` fonksiyonlarıyla yapılır. Bu yaklaşım, sensör güvenilirliğini tek bir sayısal kalite skoru ile füzyon katmanına taşır.

### 3.2 İdealize Veri Üretimi

İdealize senaryoda amaç, karşılaştırma için daha basit bir referans üretmektir. Bu modelde:

- Algılama olasılığı sabittir
- Gürültü doğrudan Kartezyen koordinatlarda eklenir
- Sabit bias vardır
- Basitleştirilmiş uniform clutter kullanılır

Bu yaklaşım fiziksel olarak tam gerçekçi değildir, fakat temel algoritmanın davranışını görmek için yararlıdır. İdealize veri, daha kontrollü bir test ortamı sağlar.

### 3.3 Gerçekçi Veri Üretimi

Gerçekçi simülasyon, idealize modele göre daha fiziksel bir sensör davranışı taklit eder. Burada birkaç önemli iyileştirme vardır:

- Mesafeye bağlı algılama olasılığı kullanılır. Hedef uzaklaştıkça `Pd` düşer.
- Gürültü polar koordinatlarda eklenir. Önce menzil ve açı bozulur, sonra Kartezyen koordinata dönülür.
- Bias drift ile sabit hata zaman içinde kayar.
- Doppler tabanlı hız modeli uygulanır.
- Statik clutter ve geçici ghost track’ler ile sahte izler üretilir.

Bu bölümdeki temel fikir, gerçek radarda ölçüm hatalarının doğrudan $x/y$ üzerinde değil, çoğu zaman menzil-açı uzayında oluştuğunu simüle etmektir. Böylece uzakta hata eliptikleşir ve sensör karakteri daha gerçekçi görünür.

### 3.4 Clutter Üretimi

Projede clutter iki ana biçimde modellenmiştir:

- **Statik clutter**: binalar, tepeler, kalıcı metal yapılar gibi sabit yansıtıcılar
- **Dinamik clutter**: kuş sürüleri, hava olayları gibi geçici sahte hedefler

Gerçekçi senaryoda ayrıca radarlar arasında korelasyon oluşturmak için ortak global clutter noktaları kullanılır. Bu noktalara birden fazla radarın aynı anda veya farklı zamanlarda bakması, füzyon algoritması için daha zor bir problem yaratır.

## 4. Aşama 2: Temel Füzyon - Sabit Hız Modeli

Temel füzyon [fusion_basic.py](fusion_basic.py#L1) içinde uygulanır. Bu sürüm, track-to-track association ve LMMSE tabanlı birleştirme yapar.

### 4.1 Sabit Hızlı Track Modeli

Temel modelde her global track için durum vektörü şu şekildedir:

\[
x = [x, v_x, y, v_y]^T
\]

Track tahmini sabit hız varsayımıyla yapılır. Yani bir ölçüm gelmediğinde hedefin yeni konumu, önceki konum ve hızından ileri sarılır. Bu adım kodda `GlobalTrack.propagate()` fonksiyonunda uygulanır.

Bu modelin ana varsayımı şudur: kısa zaman aralıklarında hız yaklaşık sabittir. Bu nedenle durum geçiş matrisi lineer ve basittir. Süreç gürültüsü ile belirsizlik artırılır; böylece modelin gerçek hareketten sapması tamamen yok sayılmaz.

### 4.2 Kalman Mantığı ve LMMSE Güncellemesi

Temel füzyondaki güncelleme, klasik Kalman filtresinin ölçüm birleştirme mantığına benzeyen LMMSE formülasyonuna dayanır. Burada amaç, iki bağımsız gibi kabul edilen tahmini en küçük ortalama kare hata ile birleştirmektir.

Kullanılan fikir özetle şudur:

- Track tahmini bir öncül olarak alınır
- Sensörden gelen local track ikinci bilgi kaynağıdır
- İki kovaryans kullanılarak optimal ağırlık hesaplanır
- Daha güvenilir bilgi daha fazla ağırlık alır

Temel sürümde kovaryanslar arasında çapraz ilişki tam olarak bilinmediği için yaklaşık bir çapraz-kovaryans modeli kullanılır. Bu nedenle temel füzyon, pratikte güçlü ama sınırlı bir yaklaşım sunar.

### 4.3 Association: Eşleştirme ve Gating

Füzyonun başarısında yalnızca güncelleme değil, önce doğru ölçümün doğru traca eşlenmesi gerekir. Bu projede association aşaması şu şekilde çalışır:

1. Global track’ler ölçüm zamanına propagate edilir
2. Local ölçüm ile mevcut track’ler arasında Mahalanobis uzaklığı hesaplanır
3. Gating eşiği altında kalan adaylar tutulur
4. Hungarian algoritması ile en iyi eşleşme seçilir

Mahalanobis uzaklığı, ölçüm farkını kovaryansla normalize ettiği için sadece geometrik uzaklığa bakmaktan daha doğrudur. Böylece yüksek belirsizlikli bir track ile düşük belirsizlikli bir track aynı şekilde değerlendirilmez.

Hungarian algoritması, birden fazla aday olduğunda tekil ve toplam maliyeti en küçük yapan atamayı bulur. Bu sayede track-ölçüm eşleşmesi bir optimizasyon problemi olarak çözülür.

## 5. Aşama 3: Gelişmiş Füzyon - Sabit İvme Modeli ve CI

Gelişmiş füzyon [fusion_advanced.py](fusion_advanced.py#L1) içinde yer alır. Bu sürüm, temel modelin iki önemli zayıflığını hedefler:

- Hedef manevra yaparken sabit hız modeli yetersiz kalabilir
- Farklı sensörlerden gelen track’lerin hata korelasyonu tam bilinmeyebilir

### 5.1 Sabit İvme (CA) Modeli

Gelişmiş modelde durum uzayı 6 boyutludur:

\[
x = [x, v_x, a_x, y, v_y, a_y]^T
\]

Burada hızın da ötesine geçilerek ivme de durumun bir parçası haline getirilir. Bu sayede manevra yapan hedefler daha iyi takip edilir.

Sabit ivme modeli özellikle kısa ve orta vadeli takipte kullanışlıdır. Hedefin hızındaki değişim, doğrudan gürültü olarak değil, modelin doğal bir parçası olarak temsil edilir. Bu yaklaşım, sabit hız modeline göre manevralı uçuşlarda daha az gecikmeli ve daha tutarlı tahmin üretir.

### 5.2 Track Başlatma ve Propagation

İlk ölçüm geldiğinde 4 boyutlu sensör ölçümü 6 boyutlu state’e genişletilir. İvme başlangıçta çok büyük belirsizlikle bırakılır; çünkü sensör doğrudan ivme ölçmez. Bu, filtreye ivme konusunda serbestlik verir ama ilk anda aşırı güvenmez.

Zaman ilerledikçe CA geçiş matrisi ile state ileri taşınır ve süreç gürültüsü üzerinden ivmedeki değişim temsil edilir. Böylece model, sadece konumu değil, hareket trendini de takip eder.

### 5.3 CI: Covariance Intersection

Gelişmiş sürümün kritik farkı, füzyon aşamasında Covariance Intersection (CI) seçeneğidir. CI, iki tahmin arasındaki çapraz korelasyon bilinmediğinde kullanılır.

Bu projede sensörlerden gelen local track’ler ile global track’ler bağımsız değildir; çünkü aynı hedefin farklı sensörlerdeki izleri dolaylı olarak aynı fiziksel hedefe dayanır. Klasik Kalman/LMMSE yaklaşımı bu bağımlılığı tam bilmeden kullanılırsa kovaryansı olduğundan küçük gösterebilir. CI bu riski azaltır.

CI’nin temel mantığı şöyledir:

- İki bilgi kaynağının ters kovaryansları alınır
- $\omega$ ağırlığı ile bunlar birleştirilir
- $\omega$ değeri öyle seçilir ki çıkan kovaryans olabildiğince küçük, ama tutarlı olsun

Bu projede $\omega$, kovaryans izini minimize edecek şekilde sayısal optimizasyonla seçilir. Böylece sonuç daha muhafazakâr ama daha güvenilir bir birleşik tahmin olur.

### 5.4 CI ve No-CI Karşılaştırması

Gelişmiş füzyon iki biçimde çalıştırılabilir:

- `use_ci=True`: Covariance Intersection ile tutarlı füzyon
- `use_ci=False`: Standart LMMSE benzeri birleştirme

Bu karşılaştırma, ortak hata yapıları olduğunda CI’nin ne kadar korumacı davranıp daha sağlam sonuç verdiğini göstermek için önemlidir.

## 6. Association ve Track Yönetimi

Hem temel hem gelişmiş füzyonda association yalnızca eşleştirme değildir; aynı zamanda track yaşam döngüsünü de belirler.

### 6.1 Gate Mekanizması

Önce aday eşleşmeler için gating yapılır. Amaç, fiziksel olarak çok uzak ölçümleri gereksiz yere değerlendirmemektir. Bu, hem yanlış eşleşmeyi azaltır hem de hesap maliyetini düşürür.

### 6.2 Hungarian Ataması

Gating’den geçen adaylar arasında Hungarian algoritması ile optimum atama seçilir. Bu yöntem, bir ölçümün birden fazla track’e aynı anda bağlanmasını engeller.

### 6.3 Existence Probability

Track’lerin yaşamı sadece geometrik eşleşmeye değil, var olma olasılığına da bağlıdır. Ölçüm geldikçe bu olasılık artar, propagate oldukça azalır. Eşikler üzerinden `TENTATIVE`, `CONFIRMED` ve `DELETED` durumları arasında geçiş yapılır.

Bu yaklaşım, zayıf ya da geçici izlerin yanlışlıkla kalıcı track’e dönüşmesini zorlaştırır.

## 7. Aşama 4: Metrik Hesaplama ve Değerlendirme

Değerlendirme [fusion_evaluation.py](fusion_evaluation.py#L1) ile yapılır. Bu modül, fused sonuçları ground truth ile karşılaştırır ve hem genel hem hedef bazlı metrikler üretir.

### 7.1 Kullanılan Metrikler

Çalışmada aşağıdaki ölçütler hesaplanır:

- Precision
- Recall
- F1 Score
- ID Precision
- ID F1
- MOTA
- ID switch sayısı
- Konum RMSE
- Hız RMSE
- NEES

### 7.2 NEES Yorumu

NEES, tahmin hatasının kovaryans ile ne kadar tutarlı olduğunu ölçer. Sadece hata küçük mü diye bakmaz; aynı zamanda filtre bu hatayı ne kadar “gerçekçi” belirsizlikle açıkladı, ona da bakar. Bu yüzden yalnızca doğruluk değil, tutarlılık değerlendirmesi de sağlar.

### 7.3 Hedef Bazlı Analiz

Kod, tüm senaryolara ek olarak her hedef için ayrı metrik üretir. Bu, füzyon performansının hedef bazında değişip değişmediğini anlamak için önemlidir. Çünkü bazı hedefler daha manevralı, bazıları daha zayıf gözlenir ve algılama zorluğu farklı olabilir.

## 8. Ana Çalışma Akışı

[main.py](main.py#L1) dosyası tüm süreci sırayla çalıştırır:

1. Ground truth’tan idealize ve gerçekçi sensör CSV’leri üretilir
2. İdealize ve gerçekçi veriler için temel ve gelişmiş füzyon çalıştırılır
3. Senaryoların genel metrikleri hesaplanır
4. Hedef bazlı metrikler tablo halinde yazdırılır

Bu sayede tek komutla tüm deney seti yeniden üretilebilir.

## 9. Sonuç ve Yorum

Bu proje, çoklu radar verisinin doğrudan birleştirilmesinden önce simülasyon katmanının ne kadar önemli olduğunu gösterir. İdealize model kontrollü bir karşılaştırma zemini sunarken, gerçekçi model sensör fiziğine daha yakın davranır ve association ile füzyon algoritmalarını daha zorlar.

Temel sabit hızlı model kısa vadede yeterli olsa da, manevralı hedeflerde sabit ivme yaklaşımı daha esnektir. Bununla birlikte asıl kritik fark, CI kullanımıdır: ölçüm kaynakları arasındaki korelasyon bilinmediğinde CI, tutarlılığı koruyan daha güvenli bir füzyon stratejisi sağlar.

Özetle proje;

- veri üretimi,
- sensör hata modeli,
- association,
- track-to-track fusion,
- CI tabanlı tutarlı birleştirme,
- ve metrik tabanlı değerlendirme

adımlarını uçtan uca uygulayan bir radar füzyon deney altyapısıdır.