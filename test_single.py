import librosa
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Yüklenen modeli eğitim verisinde nasılsa öyle test et
path = "dataset_fixed/wake/Mesut_Sarsıcı_Wake_1.wav"
audio, sr = librosa.load(path, sr=16000)
audio = np.squeeze(audio)

if len(audio) > 48000:
    audio = audio[:48000]
else:
    audio = np.pad(audio, (0, 48000 - len(audio)))

# Original
mfcc_orig = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
f_orig = np.concatenate((np.mean(mfcc_orig, axis=1), np.std(mfcc_orig, axis=1))).reshape(1, -1)
f_orig = scaler.transform(f_orig)
prob_orig = model.predict_proba(f_orig)[0][1]

# Normalized
audio_norm = audio / np.max(np.abs(audio))
mfcc_norm = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=13)
f_norm = np.concatenate((np.mean(mfcc_norm, axis=1), np.std(mfcc_norm, axis=1))).reshape(1, -1)
f_norm = scaler.transform(f_norm)
prob_norm = model.predict_proba(f_norm)[0][1]

print(f"Original Audio Prob: {prob_orig:.4f}")
print(f"Normalized Audio Prob: {prob_norm:.4f}")
