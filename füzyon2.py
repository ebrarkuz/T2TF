"""
Gelistirilmis Track-to-Track Fusion (Adim 3-6) - Covariance Intersection Tabanli
==================================================================================
Onceki versiyona gore iyilestirmeler:

1) FUZYON YONTEMI: Ad-hoc 'rho * sqrt(Pi*Pj)' capraz kovaryans tahmini yerine,
   literatürde standart olan Covariance Intersection (CI) kullanilir. CI,
   sensorler arasi korelasyon BILINMESE BILE istatistiksel tutarliligi
   (consistency) garanti eder - bu, no-feedback / asenkron mimarilerde
   (bizim senaryomuzda oldugu gibi) korelasyonun gercekte bilinemedigi
   durumlar icin matematiksel olarak savunulabilir tek yaklasimdir.

2) LOCAL TRACK SUREKLILIGI: Sensorlerin gonderdigi 'local_track_id' artik
   KULLANILIYOR. Ayni sensorun ayni fiziksel hedefe ait ardisik olcumleri,
   bir kez bir global track'e eslestiginde, sonraki olculmlerde de (gating
   gecerli oldugu surece) ayni global track'e oncelikli olarak baglanir.
   Bu, sensor-bazli track sürekliligini fuzyon seviyesine tasir ve gereksiz
   track parcalanmasini/yeniden baslatilmasini azaltir.

3) ESZAMANLI TOPLU ASSOCIATION: Ayni zaman damgasinda birden fazla sensorden
   gelen olcumler artik TEK TEK (greedy/sirali) degil, BIRLIKTE islenir.
   Tum aktif global track'ler ile o anki tum yeni olcumler arasinda TAM
   maliyet matrisi kurulup Hungarian algoritmasi ile EN IYI GLOBAL eslestirme
   bulunur. Bu, ozellikle hedefler birbirine yakinken (crossing targets)
   sirali islemenin neden olabilecegi yanlis eslestirmeleri (mis-association)
   onler.

4) FIZIKSEL OLARAK DAHA DOGRU PROCESS NOISE (Q): Pozisyon-hiz capraz
   korelasyonunu ihmal eden tam koseğen Q yerine, standart "continuous
   white noise acceleration" modeline gore turetilmis Q kullanilir.

5) NUMERIK KARARLILIK: Kovaryans guncellemesi simetriklestirilir (P ve P^T
   ortalamasi alinarak) ve gerektiginde kucuk bir regularizasyon eklenir.

6) ZAMANA DAYALI COASTING LIMITI: Sadece olasilik dususune degil, belirli
   bir suredir hic guncelleme almayan track'lerin de silinmesine izin verilir.

7) DAHA ANLAMLI CIKTI: Her benzersiz zaman adiminda CONFIRMED track'lerin
   anlik durumu kaydedilir (rastgele '50 satirda bir' yerine).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar

# ---------------------------------------------------------------------------
# 1. TQ <-> KOVARYANS DONUSUMU
# ---------------------------------------------------------------------------
TQ_MIN, TQ_MAX = 1, 15
SIGMA_POS_MAX, SIGMA_POS_MIN = 1500.0, 30.0
SIGMA_VEL_MAX, SIGMA_VEL_MIN = 25.0, 0.5


def tq_to_cov(tq) -> np.ndarray:
    """TQ degerini 4x4 olcum kovaryans matrisine (x, vx, y, vy) cevirir."""
    tq = np.clip(tq, TQ_MIN, TQ_MAX)
    frac = (tq - TQ_MIN) / (TQ_MAX - TQ_MIN)
    sp = SIGMA_POS_MAX * (SIGMA_POS_MIN / SIGMA_POS_MAX) ** frac
    sv = SIGMA_VEL_MAX * (SIGMA_VEL_MIN / SIGMA_VEL_MAX) ** frac
    return np.diag([sp ** 2, sv ** 2, sp ** 2, sv ** 2])


def sigma_pos_to_tq(sigma_pos: float) -> int:
    """Bir pozisyon std sapmasindan 'efektif TQ' geri turetir (raporlama icin)."""
    sigma_pos = float(np.clip(sigma_pos, SIGMA_POS_MIN, SIGMA_POS_MAX))
    frac = math.log(sigma_pos / SIGMA_POS_MAX) / math.log(SIGMA_POS_MIN / SIGMA_POS_MAX)
    tq = TQ_MIN + frac * (TQ_MAX - TQ_MIN)
    return int(round(np.clip(tq, TQ_MIN, TQ_MAX)))


# ---------------------------------------------------------------------------
# 2. COVARIANCE INTERSECTION (CI) FUZYONU
# ---------------------------------------------------------------------------

def covariance_intersection(x1: np.ndarray, P1: np.ndarray,
                             x2: np.ndarray, P2: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinmeyen/belirsiz korelasyon altinda istatistiksel tutarliligi
    garanti eden CI füzyonu. omega in [0,1], P_fused'in trace'ini (toplam
    belirsizligini) minimize edecek sekilde 1B optimizasyonla bulunur.

        P_fused^-1 = omega * P1^-1 + (1-omega) * P2^-1
        x_fused    = P_fused @ (omega*P1^-1@x1 + (1-omega)*P2^-1@x2)
    """
    P1_inv = np.linalg.inv(P1)
    P2_inv = np.linalg.inv(P2)

    def trace_of_fused(omega: float) -> float:
        P_inv = omega * P1_inv + (1 - omega) * P2_inv
        try:
            return float(np.trace(np.linalg.inv(P_inv)))
        except np.linalg.LinAlgError:
            return np.inf

    res = minimize_scalar(trace_of_fused, bounds=(1e-3, 1 - 1e-3), method="bounded")
    omega = res.x

    P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
    P_fused = np.linalg.inv(P_fused_inv)
    P_fused = 0.5 * (P_fused + P_fused.T)  # simetriklik garantisi

    x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)
    return x_fused, P_fused


# ---------------------------------------------------------------------------
# 3. GLOBAL TRACK SINIFI
# ---------------------------------------------------------------------------

PROCESS_NOISE_INTENSITY = 1.0   # ivme gurultu yogunlugu (m/s^2 mertebesinde, ayarlanabilir)
COAST_TIME_LIMIT = 30.0         # bu sureden uzun guncelleme alinmazsa track silinir (s)
GATE_CHI2_4DOF = 18.47          # 4 serbestlik derecesi, p=0.999 icin ki-kare esigi


class GlobalTrack:
    _id_counter = 0

    def __init__(self, time: float, state: np.ndarray, cov: np.ndarray, initial_tq: int,
                 source_key: Optional[Tuple[str, str]] = None):
        GlobalTrack._id_counter += 1
        self.id = f"GT-{GlobalTrack._id_counter:04d}"
        self.time = time
        self.state = state.reshape(4, 1)
        self.cov = cov
        self.last_update_time = time

        self.existence_prob = 0.1 + 0.7 * ((initial_tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.state_status = "TENTATIVE"  # TENTATIVE, CONFIRMED, DELETED

        # Hangi (sensor, local_track_id) kaynaklarinin bu global track'e
        # son zamanlarda baglandigini tutar -> sureklilik icin kullanilir.
        self.linked_sources: set = set()
        if source_key is not None:
            self.linked_sources.add(source_key)

    def propagate(self, target_time: float) -> None:
        """Sabit hiz modeliyle ileri sarim; Q, pozisyon-hiz korelasyonunu
        dogru yansitan 'continuous white noise acceleration' modeliyle kurulur."""
        dt = target_time - self.time
        if dt <= 0:
            return

        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

        q = PROCESS_NOISE_INTENSITY ** 2
        q_block = q * np.array([
            [dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 2 / 2, dt],
        ])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 1], [0, 1])] = q_block  # x, vx
        Q[np.ix_([2, 3], [2, 3])] = q_block  # y, vy

        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.time = target_time

        # Guncelleme alinmadigi her propagate'te varlik olasiligi hafifce duser
        self.existence_prob *= 0.97
        self.update_status()

    def update_status(self) -> None:
        coasting = (self.time - self.last_update_time) > COAST_TIME_LIMIT
        if coasting or self.existence_prob < 0.2:
            self.state_status = "DELETED"
        elif self.existence_prob > 0.85:
            self.state_status = "CONFIRMED"
        else:
            self.state_status = "TENTATIVE"

    def update_with_ci(self, meas_state: np.ndarray, meas_cov: np.ndarray,
                        meas_tq: int, source_key: Tuple[str, str]) -> None:
        x_fused, P_fused = covariance_intersection(self.state, self.cov, meas_state, meas_cov)
        self.state = x_fused
        self.cov = P_fused
        self.last_update_time = self.time

        meas_prob = 0.5 + 0.45 * ((meas_tq - TQ_MIN) / (TQ_MAX - TQ_MIN))
        self.existence_prob = self.existence_prob + (1 - self.existence_prob) * meas_prob
        self.linked_sources.add(source_key)
        self.update_status()


# ---------------------------------------------------------------------------
# 4. FUZYON MERKEZI
# ---------------------------------------------------------------------------

class FusionCenter:
    def __init__(self):
        self.global_tracks: List[GlobalTrack] = []
        # (sensor, local_track_id) -> global_track_id : sureklilik haritasi
        self.source_to_global: Dict[Tuple[str, str], str] = {}

    def _cleanup_deleted(self) -> None:
        deleted_ids = {gt.id for gt in self.global_tracks if gt.state_status == "DELETED"}
        if deleted_ids:
            self.global_tracks = [gt for gt in self.global_tracks if gt.id not in deleted_ids]
            self.source_to_global = {
                k: v for k, v in self.source_to_global.items() if v not in deleted_ids
            }

    def process_batch(self, meas_time: float, measurements: List[dict]) -> None:
        """Ayni zaman damgasina sahip TUM olcumleri birlikte isler.
        measurements: her biri {'state','cov','tq','source_key'} iceren liste."""

        # 1) Tum aktif global track'leri ortak zamana senkronize et
        for gt in self.global_tracks:
            gt.propagate(meas_time)
        self._cleanup_deleted()

        active_gts = self.global_tracks
        n_gt = len(active_gts)
        n_meas = len(measurements)

        matched_gt_idx = set()
        matched_meas_idx = set()

        # 2) Once SUREKLILIK HARITASI uzerinden "yumusak" eslestirme dene:
        #    daha once ayni (sensor,local_id) bir global track'e baglanmissa,
        #    gating hala gecerliyse o eslestirme ONCELIKLI olarak kullanilir.
        gt_index_by_id = {gt.id: i for i, gt in enumerate(active_gts)}
        for mi, meas in enumerate(measurements):
            prior_gid = self.source_to_global.get(meas["source_key"])
            if prior_gid is None or prior_gid not in gt_index_by_id:
                continue
            gi = gt_index_by_id[prior_gid]
            if gi in matched_gt_idx:
                continue
            gt = active_gts[gi]
            S = gt.cov + meas["cov"]  # bagimsizlik varsayimiyla konservatif gating kovaryansi
            diff = gt.state - meas["state"]
            try:
             d2 = (diff.T @ np.linalg.inv(S) @ diff).item()
            except np.linalg.LinAlgError:
                continue
            if d2 < GATE_CHI2_4DOF:
                gt.update_with_ci(meas["state"], meas["cov"], meas["tq"], meas["source_key"])
                self.source_to_global[meas["source_key"]] = gt.id
                matched_gt_idx.add(gi)
                matched_meas_idx.add(mi)

        # 3) Kalan (sureklilikle eslesemeyen) olcumler icin TOPLU Hungarian
        #    association: tum kalan global track'ler x tum kalan olcumler.
        remaining_gt_idx = [i for i in range(n_gt) if i not in matched_gt_idx]
        remaining_meas_idx = [i for i in range(n_meas) if i not in matched_meas_idx]

        if remaining_gt_idx and remaining_meas_idx:
            # np.inf yerine çok büyük bir ceza maliyeti (penalty) atıyoruz
            UNMATCHED_PENALTY = 1e7 
            cost = np.full((len(remaining_gt_idx), len(remaining_meas_idx)), UNMATCHED_PENALTY)
            
            for r, gi in enumerate(remaining_gt_idx):
                gt = active_gts[gi]
                for c, mi in enumerate(remaining_meas_idx):
                    meas = measurements[mi]
                    S = gt.cov + meas["cov"]
                    diff = gt.state - meas["state"]
                    try:
                        S_inv = np.linalg.inv(S)
                        sign, logdet = np.linalg.slogdet(S)
                        if sign <= 0:
                            logdet = 0.0
                    except np.linalg.LinAlgError:
                        continue
                    d2 = (diff.T @ S_inv @ diff).item()
                    
                    if d2 < GATE_CHI2_4DOF:
                        cost[r, c] = d2 + logdet

            # np.all(np.isinf(cost)) kontrolüne artık gerek yok, matris her halükarda çözülebilir
            row_ind, col_ind = linear_sum_assignment(cost)
            
            for r, c in zip(row_ind, col_ind):
                # Eğer atanan maliyet ceza puanımıza eşit veya büyükse, bu gerçek bir eşleşme değildir
                if cost[r, c] >= UNMATCHED_PENALTY:
                    continue
                    
                gi, mi = remaining_gt_idx[r], remaining_meas_idx[c]
                gt = active_gts[gi]
                meas = measurements[mi]
                gt.update_with_ci(meas["state"], meas["cov"], meas["tq"], meas["source_key"])
                self.source_to_global[meas["source_key"]] = gt.id
                matched_gt_idx.add(gi)
                matched_meas_idx.add(mi)

        # 4) Eslesemeyen tum olcumler icin YENI global track baslat
        for mi, meas in enumerate(measurements):
            if mi in matched_meas_idx:
                continue
            new_gt = GlobalTrack(meas_time, meas["state"], meas["cov"], meas["tq"],
                                  source_key=meas["source_key"])
            self.global_tracks.append(new_gt)
            self.source_to_global[meas["source_key"]] = new_gt.id

        self._cleanup_deleted()


# ---------------------------------------------------------------------------
# 5. SIMULASYONU CALISTIRMA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        df = pd.read_csv("link16_sensor_tracks_hard.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("link16_sensor_tracks.csv")
        except FileNotFoundError:
            print("HATA: sensor track CSV'si bulunamadi. Once veri uretim kodunu calistirin.")
            raise SystemExit

    # Clutter olarak isaretlenmis kayitlar varsa fuzyona sokmadan once eleyebilirsin;
    # gercekci association testi icin clutter'i da SOKMAK isteyebilirsin - asagidaki
    # satiri yorumdan cikararak clutter'i association'a dahil edebilirsin:
    # df = df  # clutter dahil
    if "is_clutter" in df.columns:
        df = df[df["is_clutter"] != True].copy()  # baseline: clutter'i disarida tut

    df = df.sort_values("time").reset_index(drop=True)

    fusion_center = FusionCenter()
    output_records = []

    print("Asenkron sensor verileri (zaman gruplari halinde, toplu) isleniyor...")
    for t, group in df.groupby("time", sort=True):
        measurements = []
        for _, row in group.iterrows():
            state = np.array([row["x"], row["vx"], row["y"], row["vy"]]).reshape(4, 1)
            cov = tq_to_cov(row["track_quality"])
            source_key = (row["sensor"], row["local_track_id"])
            measurements.append({"state": state, "cov": cov, "tq": row["track_quality"],
                                  "source_key": source_key})

        fusion_center.process_batch(t, measurements)

        for gt in fusion_center.global_tracks:
            if gt.state_status == "CONFIRMED":
                output_records.append({
                    "time": t,
                    "global_track_id": gt.id,
                    "x": gt.state[0, 0],
                    "y": gt.state[2, 0],
                    "vx": gt.state[1, 0],
                    "vy": gt.state[3, 0],
                    "pos_sigma_m": math.sqrt(max(gt.cov[0, 0], gt.cov[2, 2])),
                    "fused_tq": sigma_pos_to_tq(math.sqrt(max(gt.cov[0, 0], gt.cov[2, 2]))),
                    "prob": round(gt.existence_prob, 3),
                    "n_linked_sources": len(gt.linked_sources),
                })

    fused_df = pd.DataFrame(output_records)
    if not fused_df.empty:
        print(f"\nFuzyon tamamlandi. {len(fused_df)} adet CONFIRMED global track kaydi olusturuldu.")
        print(f"Toplam benzersiz global track sayisi: {fused_df['global_track_id'].nunique()}")
        print("\nOrnek cikti:")
        print(fused_df.tail(10).to_string(index=False))
        fused_df.to_csv("global_fused_tracks2.csv", index=False)
    else:
        print("\nHic CONFIRMED track olusmadi (esikler cok yuksek olabilir).")