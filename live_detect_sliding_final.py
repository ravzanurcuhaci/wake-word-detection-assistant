import time
import queue
import numpy as np
import sounddevice as sd
import librosa
import joblib
import warnings
import pyttsx3

warnings.filterwarnings("ignore", category=UserWarning)

# Sesli yanıt Motoru
engine = pyttsx3.init()
engine.setProperty('voice', 'trk/tr')
engine.setProperty('rate', 160)

# Ayarlar
SAMPLERATE = 16000
WINDOW_DURATION = 3.0  # Toplam bakılacak pencere süresi
STEP_DURATION = 0.5    # Her adımda ne kadarlık yeni ses alınacak
VAD_THRESHOLD = 0.06   # Sessizlik eşiği (Yeni, daha güvenli eşik)

WINDOW_SAMPLES = int(SAMPLERATE * WINDOW_DURATION)
STEP_SAMPLES = int(SAMPLERATE * STEP_DURATION)

# Model ve scaler yükle
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Ses akışı için kuyruk (queue)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Uyarı] Ses okuma hatası: {status}")
    audio_queue.put(indata.copy().flatten())

def extract_features_from_audio(audio, sr=16000):
    # DİKKAT: Normalizasyon Mantığı (Eğitim ortamına eşitleme)
    max_amp = np.max(np.abs(audio))
    
    # Sadece ses yeterince yüksekse normalize et (sessizliği amplifiye etmemek için)
    if max_amp > 0.05:
        audio = (audio / max_amp) * 0.72

    # MFCC ve deltaları hesapla
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    features = np.concatenate((mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std)).reshape(1, -1)
    return features

def main():
    print("Sistem hazır. KESİNTİSİZ DİNLEME (Sliding Window) ve Akıllı Ses Denkleştirme devrede.")
    print("Çıkmak için Ctrl+C.\n")
    print("Dinleniyor...", end="", flush=True)
    
    # 3 saniyelik boş bir havuz (buffer) oluştur
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    
    # InputStream'i arka planda başlat
    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        channels=1,
        dtype="float32",
        blocksize=STEP_SAMPLES,
        callback=audio_callback
    )
    
    with stream:
        try:
            while True:
                # 1. Yeni 0.5 saniyelik sesi al
                new_audio_chunk = audio_queue.get()
                
                # 2. Havuzu güncelle (Kayan Pencere kısmı)
                buffer = np.roll(buffer, -STEP_SAMPLES)
                buffer[-STEP_SAMPLES:] = new_audio_chunk
                
                # 3. Kendi Kendini Susturma Filtresi (VAD - Voice Activity Detection)
                chunk_max_amp = np.max(np.abs(new_audio_chunk))
                
                if chunk_max_amp < VAD_THRESHOLD:
                    # Sessizlik var. Atla! İşlemciyi yorma.
                    continue
                
                # 4. Seste hareket varsa, son 3 saniyelik havuzdan özellikleri çıkar
                # Ses kesilmemiş, parçalanmamış bir şekilde en doğru noktasındayken
                features = extract_features_from_audio(buffer, sr=SAMPLERATE)
                features = scaler.transform(features)
                
                prediction = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                wake_prob = probs[1]
                
                # 5. Sonuç karar mekanizması
                # Sliding window ortamında model çok sık tetiklendiği için 0.65 makul ve güvenlidir
                if prediction == 1 and wake_prob >= 0.65:
                    print(f"\n--> 'Hey Pakize' algılandı! (Olasılık: {wake_prob:.2f})")
                    
                    # Sesli yanıt ver
                    engine.say("Sizi dinliyorum")
                    engine.runAndWait()
                    
                    # Yanıt verdiği sırada ekoyu dinlemesin diye ufak bir mola ver ve buffer'ı sıfırla
                    buffer.fill(0)
                    time.sleep(0.5)
                    
                    # Kuyrukta birikmiş çöpleri sil
                    with audio_queue.mutex:
                        audio_queue.queue.clear()
                    
                    print("\nDinleniyor...", end="", flush=True)

        except KeyboardInterrupt:
            print("\nProgram durduruldu.")

if __name__ == "__main__":
    main()
