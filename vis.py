import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_tracking_performance():
    # 1. Dosyaları Yükle
    try:
        gt_df = pd.read_csv("ground_truth_hard.csv")
        fused_df = pd.read_csv("global_fused_tracks2.csv")
    except FileNotFoundError as e:
        print(f"HATA: Dosya bulunamadı. Lütfen dosya isimlerini kontrol edin.\nDetay: {e}")
        return

    # 2. Çizim Alanını Oluştur (1 satır, 2 sütun)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Track-to-Track Füzyon Performans Analizi", fontsize=16, fontweight='bold')

    # --- GRAFİK 1: X-Y KUŞBAKIŞI UZAYSAL GÖRÜNÜM ---
    
    # Gerçek Hedefleri Çiz (Arka planda kalın ve şeffaf)
    for target_name, group in gt_df.groupby("target"):
        ax1.plot(group["x"], group["y"], label=f"Gerçek: {target_name}", linewidth=6, alpha=0.3, color='black' if 'LINEAR' in target_name else 'gray')

    # Füzyon Çıktılarını Çiz (Her ID için farklı renk)
    unique_ids = fused_df["global_track_id"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids))) # 20 farklı renk paleti

    for (track_id, group), color in zip(fused_df.groupby("global_track_id"), colors):
        # Track rotasını çiz
        ax1.plot(group["x"], group["y"], marker='o', markersize=4, linestyle='-', label=f"{track_id}", color=color)
        
        # Karmaşayı önlemek için ID metnini sadece track'in başladığı yere yaz
        if not group.empty:
            ax1.text(group["x"].iloc[0], group["y"].iloc[0] + 200, track_id, fontsize=8, color=color, fontweight='bold')

    ax1.set_title("X-Y Konum Haritası (Gerçek Rota vs. Sistem İzleri)")
    ax1.set_xlabel("X (metre)")
    ax1.set_ylabel("Y (metre)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Legend (Sadece ilk 15 öğeyi gösterelim ki ekranı kaplamasın)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:15], labels[:15], loc='best', fontsize='small')


    # --- GRAFİK 2: ZAMAN VS Y-EKSENİ (TRACK KOPMALARINI GÖRMEK İÇİN) ---
    
    for target_name, group in gt_df.groupby("target"):
        ax2.plot(group["time"], group["y"], label=f"Gerçek: {target_name}", linewidth=6, alpha=0.3, color='black' if 'LINEAR' in target_name else 'gray')

    for (track_id, group), color in zip(fused_df.groupby("global_track_id"), colors):
        ax2.plot(group["time"], group["y"], marker='X', markersize=6, linestyle='', color=color)

    ax2.set_title("Zaman Ekseni (ID Switch / Kopma Analizi)")
    ax2.set_xlabel("Zaman (saniye)")
    ax2.set_ylabel("Y Ekseni Konumu (metre)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Ekrana Bas
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Başlık için biraz boşluk bırak
    plt.show()

if __name__ == "__main__":
    print("Grafik hazırlanıyor... Lütfen bekleyin.")
    visualize_tracking_performance()