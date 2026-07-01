import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import math

# ---------------------------------------------------------
# 1. TQ'dan KOVARYANSA DÖNÜŞÜM (Rapordaki Adım 2)
# ---------------------------------------------------------
TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5

def tq_to_cov(tq):
    """TQ değerini 4x4 ölçüm kovaryans matrisine dönüştürür."""
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    sp = SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac
    sv = SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac
    return np.diag([sp**2, sv**2, sp**2, sv**2])

# ---------------------------------------------------------
# 2. GLOBAL TRACK SINIFI VE TRACK MANAGEMENT (Rapordaki Adım 6)
# ---------------------------------------------------------
class GlobalTrack:
    _id_counter = 0

    def __init__(self, time, state, cov, initial_tq):
        GlobalTrack._id_counter += 1
        self.id = f"GT-{GlobalTrack._id_counter:04d}"
        self.time = time
        self.state = state.reshape(4, 1) # [x, vx, y, vy]^T
        self.cov = cov
        
        # Existence Probability (Var olma olasılığı) Başlatma
        # TQ 1-15 aralığını 0.1 ile 0.8 arasında bir başlangıç olasılığına map ediyoruz
        self.existence_prob = 0.1 + 0.7 * ((initial_tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.state_status = "TENTATIVE" # TENTATIVE, CONFIRMED, DELETED

    def propagate(self, target_time):
        """Rapordaki Adım 1: Zaman Senkronizasyonu. Sabit hız modeli ile ileri sarım."""
        dt = target_time - self.time
        if dt <= 0:
            return

        # State Transition Matrix (F)
        F = np.array([
            [1, dt, 0,  0],
            [0, 1,  0,  0],
            [0, 0,  1, dt],
            [0, 0,  0,  1]
        ])

        # Process Noise (Q) - Küçük bir belirsizlik eklenir
        q_pos = (0.5 * 1.0 * dt**2)**2
        q_vel = (1.0 * dt)**2
        Q = np.diag([q_pos, q_vel, q_pos, q_vel])

        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        self.time = target_time

        # Eğer uzun süre güncelleme almazsa var olma olasılığı düşer (Misdetection penalty)
        self.existence_prob *= 0.95 
        self.update_status()

    def update_status(self):
        """Existence Probability'ye göre yaşam döngüsünü günceller."""
        if self.existence_prob > 0.85:
            self.state_status = "CONFIRMED"
        elif self.existence_prob < 0.2:
            self.state_status = "DELETED"
        else:
            self.state_status = "TENTATIVE"

# ---------------------------------------------------------
# 3. FÜZYON MERKEZİ (Rapordaki Adım 3, 4 ve 5)
# ---------------------------------------------------------
class FusionCenter:
    def __init__(self):
        self.global_tracks = []
        self.rho = 0.4 # Raporda belirtilen çapraz kovaryans katsayısı

    def process_measurement(self, meas_time, local_state, local_cov, local_tq, gate_threshold=25.0):
        # 1. Mevcut global trackleri ölçüm zamanına senkronize et
        for gt in self.global_tracks:
            gt.propagate(meas_time)

        # Silinmiş trackleri temizle
        self.global_tracks = [gt for gt in self.global_tracks if gt.state_status != "DELETED"]

        active_gts = self.global_tracks
        
        # 2. Track-to-Track Association (Adım 3)
        n_gt = len(active_gts)
        if n_gt > 0:
            cost_matrix = np.full((n_gt, 1), np.inf)
            
            for i, gt in enumerate(active_gts):
                Pi = gt.cov
                Pj = local_cov
                
                # Cross-Covariance Yaklaşımı (Adım 4): P_ij = rho * sqrt(Pi * Pj)
                P_ij = self.rho * np.sqrt(np.maximum(0, Pi * Pj))
                P_ji = P_ij.T
                
                # İnovasyon Kovaryansı: S_ij = Pi + Pj - P_ij - P_ji
                S_ij = Pi + Pj - P_ij - P_ji
                
                try:
                    S_inv = np.linalg.inv(S_ij)
                    # log|S_ij| hesaplaması stabilitesini korumak için sign ve logdet kullanıyoruz
                    sign, logdet = np.linalg.slogdet(S_ij)
                    if sign <= 0:
                        logdet = 0 
                except np.linalg.LinAlgError:
                    continue # Matris tersi alınamazsa pas geç

                diff = gt.state - local_state
                mahalanobis_sq = (diff.T @ S_inv @ diff).item()
                
                # Rapordaki Maliyet Fonksiyonu: Mahalanobis + log|S|
                cost = mahalanobis_sq + logdet
                
                # Gate kontrolü (Sadece makul uzaklıktakileri eşleştir)
                if mahalanobis_sq < gate_threshold:
                    cost_matrix[i, 0] = cost

            # Hungarian Algoritması ile en iyi eşleşmeyi bul
            # GÜNCELLEME: Eğer hiçbir track gating'i geçemediyse matris tamamen np.inf kalır.
            # Bu durumda atama yapmayı denemeden doğrudan yeni track açılışına yönlendiriyoruz.
            if np.all(cost_matrix == np.inf):
                best_match_idx = None
            else:
                # Hungarian Algoritması ile en iyi eşleşmeyi bul
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                best_match_idx = None
                if len(row_ind) > 0:
                    idx = row_ind[0]
                    if cost_matrix[idx, 0] != np.inf:
                        best_match_idx = idx
            if best_match_idx is not None:
                # 3. LMMSE Füzyon (Adım 5)
                gt = active_gts[best_match_idx]
                Pi = gt.cov
                Pj = local_cov
                P_ij = self.rho * np.sqrt(np.maximum(0, Pi * Pj))
                P_ji = P_ij.T
                S_ij = Pi + Pj - P_ij - P_ji
                
                try:
                    S_inv = np.linalg.inv(S_ij)
                    # Kalman Kazancı (LMMSE için formülize edilmiş hali)
                    K = (Pi - P_ij) @ S_inv
                    
                    # State ve Kovaryans Güncellemesi
                    gt.state = gt.state + K @ (local_state - gt.state)
                    gt.cov = Pi - K @ (Pi - P_ij).T
                    
                    # Existence Probability Güncellemesi (Bayesian yaklaşımına benzer)
                    # Yüksek TQ'lu bir ölçüm geldiyse olasılık artar
                    meas_prob = 0.5 + 0.45 * ((local_tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
                    gt.existence_prob = gt.existence_prob + (1 - gt.existence_prob) * meas_prob
                    gt.update_status()
                    return
                except np.linalg.LinAlgError:
                    pass # Tersi alınamazsa yeni track açılışına düş

        # 4. Eşleşme bulunamadıysa Yeni Track başlat (Adım 6 - Track Başlatma)
        new_gt = GlobalTrack(meas_time, local_state, local_cov, local_tq)
        self.global_tracks.append(new_gt)


# ---------------------------------------------------------
# 4. SİMÜLASYONU ÇALIŞTIRMA
# ---------------------------------------------------------
if __name__ == "__main__":
    # Önceki aşamada oluşturduğumuz zorlu veriyi yükle
    try:
        df = pd.read_csv("link16_sensor_tracks_hard.csv")
    except FileNotFoundError:
        print("HATA: 'link16_sensor_tracks_hard.csv' dosyası bulunamadı. Lütfen önce veri üretim kodunu çalıştırın.")
        exit()

    # Zaman damgasına göre asenkron akışı sağla
    df = df.sort_values("time").reset_index(drop=True)
    
    fusion_center = FusionCenter()
    
    output_records = []

    print("Asenkron sensör verileri işleniyor...")
    for idx, row in df.iterrows():
        t = row["time"]
        
        # Local state vektörü
        local_state = np.array([row["x"], row["vx"], row["y"], row["vy"]]).reshape(4, 1)
        
        # TQ'dan hesaplanmış lokal ölçüm kovaryansı
        local_cov = tq_to_cov(row["track_quality"])
        
        fusion_center.process_measurement(t, local_state, local_cov, row["track_quality"])
        
        # Her 10 saniyede bir CONFIRMED track'lerin durumunu kaydet (Uygulamaya gönderilen çıktı)
        if idx % 50 == 0:
            for gt in fusion_center.global_tracks:
                if gt.state_status == "CONFIRMED":
                    output_records.append({
                        "time": t,
                        "global_track_id": gt.id,
                        "x": gt.state[0, 0],
                        "y": gt.state[2, 0],
                        "vx": gt.state[1, 0],
                        "vy": gt.state[3, 0],
                        "prob": round(gt.existence_prob, 3)
                    })

    fused_df = pd.DataFrame(output_records)
    if not fused_df.empty:
        print(f"\nFüzyon tamamlandı. {len(fused_df)} adet CONFIRMED global track kaydı oluşturuldu.")
        print("Örnek Çıktı:")
        print(fused_df.tail(10).to_string(index=False))
        fused_df.to_csv("global_fused_tracks.csv", index=False)
    else:
        print("\nHiç CONFIRMED track oluşmadı (Eşikler çok yüksek veya veri gürültüsü çok fazla olabilir).")