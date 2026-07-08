import os
import librosa
import numpy as np
import pandas as pd

data = []

dataset_path = "dataset_fixed"
target_length = 16000   # 1 saniye * 16000 Hz

for folder in ["wake", "nonwake"]:
    label = 1 if folder == "wake" else 0
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            y, sr = librosa.load(file_path, sr=16000)

            # Sessizliği kırp (VAD)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            if len(y_trimmed) > 4096:
                y = y_trimmed
            
            # Sesi 1 saniye ile sınırla ve kısa ise DOLDUR (padding)
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))
                
            # Normalize (-1 ile 1 arasına)
            y = librosa.util.normalize(y)

            # --- Zenginleştirilmiş Özellik Çıkarımı ---
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y=y)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # İstatistikleri hesapla ve birleştir
            features = np.concatenate((
                np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
                np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1),
                np.mean(rms, axis=1), np.std(rms, axis=1),
                np.mean(zcr, axis=1), np.std(zcr, axis=1),
                np.mean(centroid, axis=1), np.std(centroid, axis=1),
                np.mean(chroma, axis=1), np.std(chroma, axis=1)
            ))

            row = [file] + list(features) + [label]
            data.append(row)

columns = ["filename"] + \
          [f"mfcc{i+1}_mean" for i in range(20)] + \
          [f"mfcc{i+1}_std" for i in range(20)] + \
          [f"delta{i+1}_mean" for i in range(20)] + \
          [f"delta{i+1}_std" for i in range(20)] + \
          [f"delta2_{i+1}_mean" for i in range(20)] + \
          [f"delta2_{i+1}_std" for i in range(20)] + \
          ["rms_mean", "rms_std"] + \
          ["zcr_mean", "zcr_std"] + \
          ["centroid_mean", "centroid_std"] + \
          [f"chroma{i+1}_mean" for i in range(12)] + \
          [f"chroma{i+1}_std" for i in range(12)] + \
          ["label"]


df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)

print("dataset.csv oluşturuldu.")
print(df.head())
print("Toplam örnek sayısı:", len(df))
print(df["label"].value_counts())