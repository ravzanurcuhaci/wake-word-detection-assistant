import sounddevice as sd
import soundfile as sf
import os
import time

SAMPLERATE = 16000
DURATION = 3  # 3 saniyelik kayıt
CHANNELS = 1

def record_audio(folder_name, prefix, count):
    # Klasör yoksa oluştur
    os.makedirs(folder_name, exist_ok=True)
    
    print(f"\n--- {prefix.upper()} İÇİN SES KAYDI ---")
    print(f"Toplam {count} adet kayıt alınacak.")
    print("Her kayıt 3 saniye sürecek.\n")
    
    for i in range(1, count + 1):
        input(f"Kayıt {i}/{count} için hazır olduğunuzda ENTER'a basın...")
        
        print("Kayıt başlıyor... Lütfen konuşun!")
        audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype='float32')
        
        # Ekranda 3 saniye geri sayım göster
        for s in range(DURATION, 0, -1):
            print(f"{s}...")
            time.sleep(1)
            
        sd.wait()
        print("Kayıt tamamlandı!\n")
        
        # Dosyayı kaydet
        filename = f"{prefix}_{int(time.time())}.wav"
        filepath = os.path.join(folder_name, filename)
        sf.write(filepath, audio, SAMPLERATE)
        print(f"Kaydedildi: {filepath}")

if __name__ == "__main__":
    print("Veri Seti Toplama Aracına Hoş Geldiniz!")
    print("1. 'Hey Pakize' (Wake Word) kaydet")
    print("2. 'Yanlış Kelimeler / Gürültü' (Non-Wake Word) kaydet")
    
    secim = input("Seçiminiz (1 veya 2): ")
    
    if secim == "1":
        adet = int(input("Kaç adet 'Hey Pakize' kaydetmek istiyorsunuz? (Örn: 5): "))
        record_audio("dataset_fixed/wake", "yeni_pakize", adet)
    elif secim == "2":
        adet = int(input("Kaç adet yanlış kelime (örn: Hey Habibe, öksürük vb.) kaydetmek istiyorsunuz? (Örn: 10): "))
        record_audio("dataset_fixed/nonwake", "yeni_yanlis", adet)
    else:
        print("Geçersiz seçim.")
        
    print("\nİşlem bitti! Yeni verileri eğitime dahil etmek için şu komutları sırasıyla çalıştırın:")
    print("1) python create_dataset.py")
    print("2) python train_model.py")
