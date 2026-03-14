import os
import librosa
import soundfile as sf
import numpy as np

# Veri artırma (Data Augmentation) Fonksiyonları

def add_noise(data, noise_factor=0.02):
    """Sese beyaz gürültü (white noise) ekler."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Kırpılmayı (clipping) önlemek için normalize edelim
    max_val = np.max(np.abs(augmented_data))
    if max_val > 1.0:
        augmented_data = augmented_data / max_val
    return augmented_data

def change_pitch(data, sr, n_steps):
    """Sesin perdesini (pitch) değiştirir (kalınlaştırır veya inceltir)."""
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def change_speed(data, speed_factor):
    """Sesin hızını değiştirir (hızlandırır veya yavaşlatır)."""
    return librosa.effects.time_stretch(y=data, rate=speed_factor)

def augment_dataset(base_dir="dataset_fixed"):
    print("Veri artırma (Data Augmentation) işlemi başlatılıyor...")
    
    for folder in ["wake", "nonwake"]:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"{folder_path} bulunamadı, atlanıyor.")
            continue
            
        print(f"\nİşleniyor: {folder} kategorisi...")
        
        # Orijinal dosyaları al (daha önce üretilmiş augmented dosyaları atla)
        original_files = [f for f in os.listdir(folder_path) if f.endswith(".wav") and not f.startswith("aug_")]
        
        for file in original_files:
            file_path = os.path.join(folder_path, file)
            try:
                # Sesi yükle
                data, sr = librosa.load(file_path, sr=16000)
                
                # Sadece tek boyuta indir
                data = np.squeeze(data)
                
                # 1. Gürültü eklenmiş versiyon (.02 ve .04)
                sf.write(os.path.join(folder_path, f"aug_noise_light_{file}"), add_noise(data, 0.02), sr)
                sf.write(os.path.join(folder_path, f"aug_noise_heavy_{file}"), add_noise(data, 0.04), sr)
                
                # 2. Pitch (Perde) değiştirilmiş versiyon (daha kalın -2 ve daha ince +2)
                sf.write(os.path.join(folder_path, f"aug_pitch_low_{file}"), change_pitch(data, sr, -2), sr)
                sf.write(os.path.join(folder_path, f"aug_pitch_high_{file}"), change_pitch(data, sr, 2), sr)
                
                # 3. Hız değiştirilmiş versiyon (%10 daha yavaş ve %10 daha hızlı)
                sf.write(os.path.join(folder_path, f"aug_speed_slow_{file}"), change_speed(data, 0.9), sr)
                sf.write(os.path.join(folder_path, f"aug_speed_fast_{file}"), change_speed(data, 1.1), sr)
                
                # İsterseniz bu iki yöntemi birleştirip daha da fazla varyasyon yapabilirsiniz.
                
            except Exception as e:
                print(f"Hata ({file}): {e}")
                
    print("\nVeri artırma tamamlandı! Artık daha fazla ses varyasyonu var.")
    print("Mevcut CSV dosyasını güncellemek için sırasıyla şunu çalıştırın:")
    print("python create_dataset.py")
    print("python train_model.py")

if __name__ == "__main__":
    augment_dataset()
