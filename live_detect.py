import time
import numpy as np
import sounddevice as sd
import librosa
import joblib
import warnings
import pyttsx3

# Sklearn'den gelen feature isimsiz (feature names) uyarısını gizle
warnings.filterwarnings("ignore", category=UserWarning)

# Sesli yanıt (TTS) Motorunu Başlat
engine = pyttsx3.init()
engine.setProperty('voice', 'trk/tr')  # Türkçe telaffuz
engine.setProperty('rate', 160)        # Normal konuşma hızı

# Ayarlar
SAMPLERATE = 16000
DURATION = 3  # saniye
TARGET_LENGTH = SAMPLERATE * DURATION

# Model ve scaler yükle
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features_from_audio(audio, sr=16000):
    # Tek boyuta indir
    audio = np.squeeze(audio)

    # Uzunluk sabitle
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

    # Ses düzeyini normalize et (Eğer ses çok kısıksa yükseltir)
    max_amp = np.max(np.abs(audio))
    
    # Sessizlik / Dip Gürültüsü kontrolü (Önemli!)
    # Eğer mikrofondan gelen en yüksek ses 0.02'den düşükse bu bir sessizliktir
    is_silent = max_amp < 0.02

    if max_amp > 0:
        audio = audio / max_amp

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Delta ve Delta-Delta
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    features = np.concatenate((mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std)).reshape(1, -1)
    return features, is_silent

print("Sistem hazır. 3 saniyelik dinleme döngüsü başladı.")
print("Wake word algılanırsa yanıt verecek. Algılanmazsa sessiz kalacak.")
print("Çıkmak için Ctrl+C.\n")

try:
    while True:
        print("Dinleniyor...")
        audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype="float32")
        sd.wait()

        features, is_silent = extract_features_from_audio(audio, sr=SAMPLERATE)
        
        # Eğer sadece dip gürültüsü varsa, modelin kafası karışmasın diye tahmin yapma
        if is_silent:
            print("[Debug] Çok sessiz, tahmin atlandı.")
            continue

        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        # probs[1] = wake olasılığı
        wake_prob = probs[1]

        # Daha güvenli karar için eşik
        print(f"[Debug] Pred: {prediction}, Olasılık: {wake_prob:.2f}")
        
        if prediction == 1 and wake_prob >= 0.49:
            print(f"--> Hey Pakize algılandı! (olasılık: {wake_prob:.2f})\n")
            # Sesli yanıt ver
            engine.say("Sizi dinliyorum")
            engine.runAndWait()
            
            # Yanıt verdikten sonra biraz bekle ki kendi sesini tekrar dinlemesin
            time.sleep(0.5)
        else:
            # Sessiz kalsın istiyorsan bunu tamamen kaldırabilirsin
            pass

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nProgram durduruldu.")