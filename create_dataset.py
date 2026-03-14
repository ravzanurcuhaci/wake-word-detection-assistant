import os
import librosa
import numpy as np
import pandas as pd

data = []

dataset_path = "dataset_fixed"
target_length = 48000   # 3 saniye * 16000 Hz

for folder in ["wake", "nonwake"]:
    label = 1 if folder == "wake" else 0
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            y, sr = librosa.load(file_path, sr=16000)

            # sesi 3 saniyeye sabitle
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))

            # MFCC çıkar
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Zaman içindeki değişimi yakalamak için Delta ve Delta-Delta eklentisi (Çok Kritik!)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # özet özellikler
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            delta_mean = np.mean(delta_mfcc, axis=1)
            delta_std = np.std(delta_mfcc, axis=1)
            
            delta2_mean = np.mean(delta2_mfcc, axis=1)
            delta2_std = np.std(delta2_mfcc, axis=1)

            # Tüm özellikleri birleştir (13*6 = 78 özellik)
            features = np.concatenate((mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std))

            row = [file] + list(features) + [label]
            data.append(row)

columns = ["filename"] + \
          [f"mfcc{i+1}_mean" for i in range(13)] + \
          [f"mfcc{i+1}_std" for i in range(13)] + \
          [f"delta{i+1}_mean" for i in range(13)] + \
          [f"delta{i+1}_std" for i in range(13)] + \
          [f"delta2_{i+1}_mean" for i in range(13)] + \
          [f"delta2_{i+1}_std" for i in range(13)] + \
          ["label"]

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)

print("dataset.csv oluşturuldu.")
print(df.head())
print("Toplam örnek sayısı:", len(df))
print(df["label"].value_counts())