import os
import librosa
import numpy as np

wake_dir = "dataset_fixed/wake"

print("Analyzing training files volume levels:")
if not os.path.exists(wake_dir):
    print(f"Directory {wake_dir} not found.")
else:
    for i, file in enumerate(os.listdir(wake_dir)):
        if file.endswith(".wav"):
            path = os.path.join(wake_dir, file)
            audio, sr = librosa.load(path, sr=16000)
            
            # Tek boyuta indir
            audio = np.squeeze(audio)
            
            max_amp = np.max(np.abs(audio))
            mean_amp = np.mean(np.abs(audio))
            print(f"{file} -> Max: {max_amp:.4f}, Mean: {mean_amp:.4f}")
            
        if i >= 10:  # just test a few
            break
