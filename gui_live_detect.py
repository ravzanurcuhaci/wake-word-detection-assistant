import time
import queue
import threading
import warnings
import numpy as np
import sounddevice as sd
import librosa
import joblib
import pyttsx3
import tkinter as tk

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Sesli Yanıt Motoru
# =========================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

try:
    engine.setProperty("voice", "trk/tr")
except Exception:
    pass

# =========================
# Ayarlar
# =========================
SAMPLERATE = 16000
WINDOW_DURATION = 3.0
STEP_DURATION = 0.5
VAD_THRESHOLD = 0.06
DETECTION_THRESHOLD = 0.65
COOLDOWN_SECONDS = 2.0

WINDOW_SAMPLES = int(SAMPLERATE * WINDOW_DURATION)
STEP_SAMPLES = int(SAMPLERATE * STEP_DURATION)

# GUI görünürlük süreleri
SPEAKING_HOLD_SECONDS = 0.8
DETECTED_HOLD_SECONDS = 2.5

# =========================
# Model ve scaler yükle
# =========================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Queue'lar
# =========================
audio_queue = queue.Queue()
ui_queue = queue.Queue()

# =========================
# Ses callback
# =========================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Uyarı] Ses okuma hatası: {status}")
    audio_queue.put(indata.copy().flatten())

# =========================
# Özellik çıkarma
# =========================
def extract_features_from_audio(audio, sr=16000):
    audio = np.squeeze(audio)

    # Sessizliği kırp
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    # Delta hesaplaması en az 9 frame (yaklaşık 4000 sample) gerektirir.
    # Bu yüzden sadece yeterince uzunsa kırpılmış halini kullan.
    if len(audio_trimmed) > 4096:
        audio = audio_trimmed
        
    # Uzunluk sabitle (1 Saniye Padding/Truncating)
    TARGET_LENGTH = 16000
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = librosa.util.normalize(audio)

    # Özellik çıkarımı (150 özellik)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    features = np.concatenate((
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(centroid, axis=1), np.std(centroid, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1)
    )).reshape(1, -1)

    return features

# =========================
# TTS
# =========================
def speak_text(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS hatası:", e)

# =========================
# Dinleme thread
# =========================
def listener_loop():
    print("Sistem hazır. GUI dinleme başladı.")
    print("Çıkmak için pencereyi kapatabilir veya Ctrl+C kullanabilirsin.")
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    last_detection_time = 0

    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        channels=1,
        dtype="float32",
        blocksize=STEP_SAMPLES,
        callback=audio_callback
    )

    with stream:
        while True:
            new_audio_chunk = audio_queue.get()

            # Buffer güncelle
            buffer = np.roll(buffer, -STEP_SAMPLES)
            buffer[-STEP_SAMPLES:] = new_audio_chunk

            # Ses var mı kontrolü
            chunk_max_amp = np.max(np.abs(new_audio_chunk))

            if chunk_max_amp < VAD_THRESHOLD:
                ui_queue.put(("idle", None))
                continue
            else:
                # Sadece konuşma var bilgisi
                ui_queue.put(("speaking", chunk_max_amp))

            # Cooldown
            now = time.time()
            if now - last_detection_time < COOLDOWN_SECONDS:
                continue

            # Özellik çıkar + tahmin
            features = extract_features_from_audio(buffer, sr=SAMPLERATE)
            features = scaler.transform(features)

            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            wake_prob = probs[1]

            if prediction == 1 and wake_prob >= DETECTION_THRESHOLD:
                last_detection_time = now
                print(f"--> 'Hey Pakize' algılandı! (Olasılık: {wake_prob:.2f})")

                ui_queue.put(("detected", wake_prob))

                speak_text("Sizi dinliyorum")

                buffer.fill(0)
                time.sleep(0.5)

                with audio_queue.mutex:
                    audio_queue.queue.clear()

# =========================
# GUI
# =========================
class WakeWordGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hey Pakize")
        self.root.geometry("500x600")
        self.root.configure(bg="black")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=500,
            height=600,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack()

        self.current_mode = "idle"
        self.last_speaking_time = 0
        self.detected_until = 0
        self.last_prob = None
        self.current_amp = 0.0

        self.draw_idle_screen()
        self.poll_ui_queue()

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw_mic_icon(self, body_color="white", accent_color="white"):
        # Mikrofon kapsülü
        self.canvas.create_oval(
            195, 150, 305, 300,
            fill=body_color, outline=body_color, width=2
        )

        # İç ızgara çizgileri
        for y in range(175, 285, 18):
            self.canvas.create_line(215, y, 285, y, fill="black", width=2)

        # Sap
        self.canvas.create_rectangle(
            235, 295, 265, 380,
            fill=body_color, outline=body_color
        )

        # Alt bağlantı
        self.canvas.create_line(
            250, 380, 250, 425,
            fill=accent_color, width=6
        )

        # Dış taşıyıcı U şekli
        self.canvas.create_arc(
            170, 300, 330, 470,
            start=205, extent=130,
            style="arc", outline=accent_color, width=6
        )

        # Alt taban
        self.canvas.create_line(
            200, 470, 300, 470,
            fill=accent_color, width=6
        )

    def draw_sound_bars(self, amp=0.1):
        amp = max(0.0, min(1.0, float(amp)))

        center_x = 250
        base_y = 545
        bar_width = 16
        gap = 10

        heights_min = [18, 28, 38, 28, 18]
        heights_max = [45, 70, 95, 70, 45]

        total_width = 5 * bar_width + 4 * gap
        start_x = center_x - total_width // 2

        for i in range(5):
            x1 = start_x + i * (bar_width + gap)
            x2 = x1 + bar_width
            h = heights_min[i] + (heights_max[i] - heights_min[i]) * amp
            y1 = base_y - h
            y2 = base_y

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="lime", outline="lime"
            )

    def draw_robot_icon(self):
        self.canvas.create_rectangle(160, 120, 340, 280, fill="#2c2c2c", outline="white", width=3)
        self.canvas.create_line(250, 80, 250, 120, fill="white", width=4)
        self.canvas.create_oval(238, 60, 262, 84, fill="red", outline="red")
        self.canvas.create_oval(195, 160, 235, 200, fill="cyan", outline="cyan")
        self.canvas.create_oval(265, 160, 305, 200, fill="cyan", outline="cyan")
        self.canvas.create_rectangle(200, 225, 300, 250, fill="white", outline="white")

        for x in range(210, 300, 18):
            self.canvas.create_line(x, 225, x, 250, fill="black", width=2)

        self.canvas.create_rectangle(135, 160, 160, 220, fill="gray", outline="white")
        self.canvas.create_rectangle(340, 160, 365, 220, fill="gray", outline="white")

    def draw_idle_screen(self):
        self.clear_canvas()
        self.current_mode = "idle"

        self.canvas.create_text(
            250, 80,
            text="HEY PAKİZE",
            fill="white",
            font=("Arial", 24, "bold")
        )

        self.canvas.create_text(
            250, 120,
            text="Mikrofon açık",
            fill="gray85",
            font=("Arial", 15)
        )

        self.draw_mic_icon(body_color="white", accent_color="white")

        self.canvas.create_text(
            250, 530,
            text="Bekleniyor...",
            fill="gray75",
            font=("Arial", 18, "bold")
        )

    def draw_speaking_screen(self, amp=0.1):
        self.clear_canvas()
        self.current_mode = "speaking"

        self.canvas.create_text(
            250, 80,
            text="HEY PAKİZE",
            fill="white",
            font=("Arial", 24, "bold")
        )

        self.canvas.create_text(
            250, 120,
            text="Konuşma algılandı",
            fill="gray85",
            font=("Arial", 15)
        )

        self.draw_mic_icon(body_color="white", accent_color="white")
        self.draw_sound_bars(amp=amp)

        self.canvas.create_text(
            250, 580,
            text="Dinleniyor...",
            fill="lime",
            font=("Arial", 18, "bold")
        )

    def draw_detected_screen(self, prob=None):
        self.clear_canvas()
        self.current_mode = "detected"

        self.draw_robot_icon()

        self.canvas.create_text(
            250, 350,
            text="Hey Pakize algılandı",
            fill="lightgreen",
            font=("Arial", 20, "bold")
        )

        self.canvas.create_text(
            250, 390,
            text="Sizi dinliyorum",
            fill="white",
            font=("Arial", 24, "bold")
        )

        if prob is not None:
            self.canvas.create_text(
                250, 435,
                text=f"Algılama olasılığı: {prob:.2f}",
                fill="lightgreen",
                font=("Arial", 16)
            )

    def poll_ui_queue(self):
        now = time.time()

        try:
            while True:
                state, data = ui_queue.get_nowait()

                if state == "speaking":
                    self.last_speaking_time = now
                    amp = min(float(data) / 0.30, 1.0) if data is not None else 0.1
                    self.current_amp = amp

                    if now >= self.detected_until:
                        self.draw_speaking_screen(amp=self.current_amp)

                elif state == "detected":
                    self.last_prob = data
                    self.detected_until = now + DETECTED_HOLD_SECONDS
                    self.draw_detected_screen(prob=data)

                elif state == "idle":
                    pass

        except queue.Empty:
            pass

        if now < self.detected_until:
            if self.current_mode != "detected":
                self.draw_detected_screen(prob=self.last_prob)

        elif now - self.last_speaking_time < SPEAKING_HOLD_SECONDS:
            self.draw_speaking_screen(amp=self.current_amp)

        else:
            if self.current_mode != "idle":
                self.draw_idle_screen()

        self.root.after(100, self.poll_ui_queue)

# =========================
# Ana program
# =========================
def main():
    root = tk.Tk()
    app = WakeWordGUI(root)

    listener_thread = threading.Thread(target=listener_loop, daemon=True)
    listener_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()