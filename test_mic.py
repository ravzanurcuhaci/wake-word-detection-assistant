import sounddevice as sd
import numpy as np

# Record audio
SAMPLERATE = 16000
DURATION = 3
print("Recording test audio for 3 seconds...")
audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype="float32")
sd.wait()

print("Max amplitude:", np.max(np.abs(audio)))
print("Mean amplitude:", np.mean(np.abs(audio)))
print("Unique values count (rough check for silence):", len(np.unique(audio)))
