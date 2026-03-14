import os
import librosa
import numpy as np
import joblib

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
TARGET_LENGTH = 48000

def predict_file(filepath):
    audio, sr = librosa.load(filepath, sr=16000)
    audio = np.squeeze(audio)
    
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
        
    # max_amp = np.max(np.abs(audio))
    # if max_amp > 0:
    #     audio = audio / max_amp
        
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)
    features = scaler.transform(features)
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return pred, prob

# Test the wake directory
wake_dir = "dataset_fixed/wake"
if os.path.exists(wake_dir):
    print("Testing Wake Data (Should be Pred: 1, Prob > 0.5):")
    for i, file in enumerate(os.listdir(wake_dir)):
        if file.endswith(".wav"):
            path = os.path.join(wake_dir, file)
            pred, prob = predict_file(path)
            print(f"{file} -> Pred: {pred}, Olasılık: {prob:.2f}")
        if i >= 5: break

# Test the nonwake directory
nonwake_dir = "dataset_fixed/nonwake"
if os.path.exists(nonwake_dir):
    print("\nTesting Nonwake Data (Should be Pred: 0, Prob < 0.5):")
    for i, file in enumerate(os.listdir(nonwake_dir)):
        if file.endswith(".wav"):
            path = os.path.join(nonwake_dir, file)
            pred, prob = predict_file(path)
            print(f"{file} -> Pred: {pred}, Olasılık: {prob:.2f}")
        if i >= 5: break
