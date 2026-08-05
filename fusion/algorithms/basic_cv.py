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

# YENİ: Birleştirme (Merge) Eşikleri
DUPLICATE_DIST_M = 150.0
DUPLICATE_VEL_MPS = 30.0

def tq_to_cov(tq):
    """TQ değerini 6x6 ölçüm kovaryans matrisine dönüştürür."""
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    sp = SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac
    sv = SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac
    return np.diag([sp**2, sv**2, sp**2, sv**2, sp**2, sv**2])


def measurement_cov_from_row(row):
    """Use sensor-provided uncertainty when it is available."""
    if "sigma_pos_m" in row and "sigma_vel_mps" in row:
        sp = float(row["sigma_pos_m"])
        sv = float(row["sigma_vel_mps"])
        if np.isfinite(sp) and np.isfinite(sv) and sp > 0 and sv > 0:
            return np.diag([sp**2, sv**2, sp**2, sv**2, sp**2, sv**2])
    return tq_to_cov(row.get("track_quality", TQ_MAX))

# ---------------------------------------------------------
# 2. GLOBAL TRACK SINIFI VE TRACK MANAGEMENT (Rapordaki Adım 6)
# ---------------------------------------------------------
class GlobalTrack:
    _id_counter = 0

    def __init__(self, time, state, cov, initial_tq):
        GlobalTrack._id_counter += 1
        self.id = f"GT-{GlobalTrack._id_counter:04d}"
        self.time = time
        self.state = state.reshape(6, 1) # [x, vx, y, vy, z, vz]^T
        self.cov = cov
        
        # YENİ: Birleştirme (merging) önceliği için hit sayacı
        self.hits_count = 1
        
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
            [1, dt, 0,  0,  0,  0],
            [0, 1,  0,  0,  0,  0],
            [0, 0,  1, dt,  0,  0],
            [0, 0,  0,  1,  0,  0],
            [0, 0,  0,  0,  1, dt],
            [0, 0,  0,  0,  0,  1],
        ])

        # Process Noise (Q) - Küçük bir belirsizlik eklenir
        q_pos = (0.5 * 1.0 * dt**2)**2
        q_vel = (1.0 * dt)**2
        Q = np.diag([q_pos, q_vel, q_pos, q_vel, q_pos, q_vel])

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

    def _tracks_are_duplicate(self, t1, t2, chi2_thresh=16.0):
        """
        İki track'in konumlarını Mahalanobis, hızlarını Öklid ile kıyaslar.
        Dikkat: 6D durumda pozisyonlar 0,2,4; hızlar 1,3,5 indekslerdedir.
        """
        # 1. Konum için Mahalanobis Mesafesi (0: x, 2: y, 4: z)
        dx = np.array([
            [float(t1.state[0,0]) - float(t2.state[0,0])],
            [float(t1.state[2,0]) - float(t2.state[2,0])],
            [float(t1.state[4,0]) - float(t2.state[4,0])],
        ])
        
        P_sum = t1.cov[np.ix_([0,2,4],[0,2,4])] + t2.cov[np.ix_([0,2,4],[0,2,4])]
        
        try:
            d2 = float((dx.T @ np.linalg.inv(P_sum) @ dx).item())
        except np.linalg.LinAlgError:
            return False

        # 2. Hız için basit Öklid Mesafesi (1: vx, 3: vy, 5: vz)
        dvx = float(t1.state[1,0]) - float(t2.state[1,0])
        dvy = float(t1.state[3,0]) - float(t2.state[3,0])
        dvz = float(t1.state[5,0]) - float(t2.state[5,0])
        vel_diff = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

        return d2 < chi2_thresh and vel_diff <= DUPLICATE_VEL_MPS

    def _merge_duplicates(self, current_time):
        """Birbirine çok yakın (konum ve hız) ve paralel ilerleyen track'leri birleştirir."""
        confirmed_tracks = [gt for gt in self.global_tracks if gt.state_status == "CONFIRMED"]
        tentative_tracks = [gt for gt in self.global_tracks if gt.state_status == "TENTATIVE"]
        to_delete = set()
        
        # 1. Aşama: CONFIRMED <-> CONFIRMED Kontrolü
        for i in range(len(confirmed_tracks)):
            for j in range(i + 1, len(confirmed_tracks)):
                t1, t2 = confirmed_tracks[i], confirmed_tracks[j]
                
                if t1.id in to_delete or t2.id in to_delete:
                    continue
                    
                if self._tracks_are_duplicate(t1, t2, chi2_thresh=16.0):
                    if t1.hits_count > t2.hits_count or (t1.hits_count == t2.hits_count and t1.existence_prob >= t2.existence_prob):
                        keeper, weaker = t1, t2
                    else:
                        keeper, weaker = t2, t1
                        
                    keeper.hits_count += weaker.hits_count # Çalınan ölçüm gücünü geri aktar
                    
                    to_delete.add(weaker.id)

        # 2. Aşama: CONFIRMED <-> TENTATIVE Kontrolü (Erken Temizlik)
        for c in confirmed_tracks:
            if c.id in to_delete:
                continue
            for t in tentative_tracks:
                if t.id in to_delete:
                    continue

                if self._tracks_are_duplicate(c, t, chi2_thresh=16.0):
                    c.hits_count += t.hits_count
                    
                    to_delete.add(t.id)

        # Silinecekleri ana listeden çıkar
        self.global_tracks = [gt for gt in self.global_tracks if gt.id not in to_delete]

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
                    
                    # YENİ: Hit sayacını artır
                    gt.hits_count += 1
                    
                    # Existence Probability Güncellemesi (Bayesian yaklaşımına benzer)
                    # Yüksek TQ'lu bir ölçüm geldiyse olasılık artar
                    meas_prob = 0.5 + 0.45 * ((local_tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
                    gt.existence_prob = gt.existence_prob + (1 - gt.existence_prob) * meas_prob
                    gt.update_status()
                    
                    # Eşleştirme yapıldı, fonksiyonu sonlandırmadan önce merge işlemini çağır
                    self._merge_duplicates(meas_time)
                    return
                except np.linalg.LinAlgError:
                    pass # Tersi alınamazsa yeni track açılışına düş

        # 4. Eşleşme bulunamadıysa Yeni Track başlat (Adım 6 - Track Başlatma)
        new_gt = GlobalTrack(meas_time, local_state, local_cov, local_tq)
        self.global_tracks.append(new_gt)
        
        # Döngü sonu merge kontrolü
        self._merge_duplicates(meas_time)


# ---------------------------------------------------------
# 4. SİMÜLASYONU ÇALIŞTIRMA
# ---------------------------------------------------------

def run_basic_fusion(
    sensor_csv: str = "radar_sensor_tracks.csv",
    output_csv: str = "basic_fusion_sonuclari.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print(f"Temel füzyon çalıştırılıyor: {sensor_csv}")

    df = pd.read_csv(sensor_csv)
    df = df.sort_values("time").reset_index(drop=True)

    GlobalTrack._id_counter = 0
    fusion_center = FusionCenter()
    output_records = []

    if verbose:
        print("Asenkron sensör verileri işleniyor...")

    # Process a complete sensor scan before emitting a snapshot.  The previous
    # ``idx % 50`` sampling made output density depend on CSV row count and
    # therefore produced misleading recall values.
    for t, group in df.groupby("time", sort=True):
        for _, row in group.iterrows():
            local_state = np.array([
                row["x"],
                row["vx"],
                row["y"],
                row["vy"],
                row.get("z", 0.0),
                row.get("vz", 0.0),
            ]).reshape(6, 1)
            local_cov = measurement_cov_from_row(row)
            tq = row.get("track_quality", TQ_MAX)
            fusion_center.process_measurement(t, local_state, local_cov, tq)

        for gt in fusion_center.global_tracks:
            if gt.state_status == "CONFIRMED":
                sigmas = np.sqrt(np.maximum(np.diag(gt.cov), 1e-12))
                output_records.append({
                    "time": t,
                    "global_track_id": gt.id,
                    "x": gt.state[0, 0],
                    "y": gt.state[2, 0],
                    "z": gt.state[4, 0],
                    "vx": gt.state[1, 0],
                    "vy": gt.state[3, 0],
                    "vz": gt.state[5, 0],
                    "pos_sigma_m": sigmas[0],
                    "vel_sigma_mps": sigmas[1],
                    "sigma_x_m": sigmas[0],
                    "sigma_vx_mps": sigmas[1],
                    "sigma_y_m": sigmas[2],
                    "sigma_vy_mps": sigmas[3],
                    "sigma_z_m": sigmas[4],
                    "sigma_vz_mps": sigmas[5],
                    "prob": round(gt.existence_prob, 3),
                })

    fused_df = pd.DataFrame(output_records)
    fused_df.to_csv(output_csv, index=False)

    if verbose:
        if not fused_df.empty:
            print(f"\nTemel füzyon tamamlandı. {len(fused_df)} adet CONFIRMED global track kaydı oluşturuldu.")
            print("Örnek Çıktı:")
            print(fused_df.tail(10).to_string(index=False))
        else:
            print("\nHiç CONFIRMED track oluşmadı (Eşikler çok yüksek veya veri gürültüsü çok fazla olabilir).")
        print(f"Çıktı dosyası: {output_csv}")

    return fused_df
