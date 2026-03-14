🎙️ Hey Pakize - Wake Word Detection Assistant (Python & Machine Learning)
Python ile geliştirilmiş, anlık ses işleme ve makine öğrenmesi modelleri kullanarak "Hey Pakize" uyanma kelimesini (wake word) algılayan akıllı bir asistan altyapısıdır. Proje; veri toplama, ses sinyali işleme, model eğitimi ve gerçek zamanlı mikrofondan algılama süreçlerini uçtan uca içerir.

🚀 Özellikler
🎤 Gerçek Zamanlı Algılama: Mikrofondan anlık ses akışı ile "Hey Pakize" kelimesini dinleme ve tepki verme.
📊 Veri Toplama ve Çoğaltma: Kendi ses verilerinizi kaydetme ve veri artırma (data augmentation) ile veri setini zenginleştirme.
🧠 Makine Öğrenmesi Modeli: MFCC özellikleri çıkarılarak eğitilmiş destek vektör makineleri (SVM) ve diğer algoritmalar ile yüksek doğruluklu tahmin.
⚡ Hızlı Yanıt Sistemi: "Hey Pakize" algılandığında Text-to-Speech (TTS) motoru ile anında sesli geri bildirim ("Sizi dinliyorum").
🔉 Gürültü ve Sessizlik Filtresi: Arka plan gürültülerini ve sessizlik anlarını filtreleyerek model hatalarını (false positive) azaltma.

🔍 Mimari ve Çalışma Mantığı
Bu projede ses işleme ve makine öğrenmesi boru hattı (pipeline) birleştirilerek sorumluluklar birbirinden ayrılmıştır:

- Veri Toplama (Data Collection): `record_data.py` ve diğer scriptler ile çevreden ses verileri toplanıp etiketlenir.
- Veri Artırma (Data Augmentation): `augment_data.py` ile mevcut ses verileri hız, perde ve gürültü eklenerek çoğaltılır, genellenebilirlik artırılır.
- Özellik Çıkarımı (Feature Extraction): Ses dosyalarından Librosa kütüphanesi ile MFCC, Delta ve Delta-Delta özellikleri çıkarılarak modele uygun sayısal verilere dönüştürülür.
- Model Eğitimi (Model Training): Çıkarılan özellikler kullanılarak SVM, Random Forest, MLP gibi sınıflandırma algoritmaları ile eğitilir ve tahmin ağırlıkları kaydedilir.
- Canlı Dinleme (Live Detection): `live_detect.py` ile sistem mikrofondan 3 saniyelik parçalar kaydeder, tahminde bulunur ve tespit durumunda TTS (pyttsx3) ile yanıt verir.

🛠️ Kullanılan Teknolojiler
- Dil: Python
- Ses İşleme: Librosa, SoundDevice, PyAudio
- Makine Öğrenmesi: Scikit-Learn (SVM, Random Forest, MLP, vs.)
- Veri Bilimi: NumPy, Pandas
- Sesli Yanıt (TTS): pyttsx3
- Model Kayıt: Joblib

📂 Mimari Yapı
```text
take_word_project/
 ├─ dataset_fixed/         # İşlenmiş ve artırılmış ses verileri (Wake/Not Wake)
 ├─ record_data.py         # Mikrofondan ses kaydetme aracı
 ├─ create_dataset.py      # Ses verilerinden MFCC özelliklerini çıkarıp dataset.csv'yi oluşturma
 ├─ augment_data.py        # Mevcut ses verilerini çoğaltma araçları
 ├─ train_model.py         # Veri setini kullanarak modeli eğitme
 ├─ compare_models.py      # Farklı makine öğrenmesi algoritmalarını karşılaştırma
 ├─ live_detect.py         # Gerçek zamanlı algılama ve sesli asistan yanıtı
 ├─ test_*.py              # Modeli, sesi ve mikrofonu test etme betikleri
 ├─ dataset.csv            # Çıkarılan sayısal özelliklerin CSV tablosu
 ├─ svm_model.pkl          # Eğitilmiş ve kaydedilmiş yapay zeka modeli
 └─ scaler.pkl             # Veri standardizasyonu için scaler
```

▶️ Kurulum & Çalıştırma
Repoyu klonla:
```bash
git clone https://github.com/ravzanurcuhaci/wake-word-detection-assistant.git
```
Proje dizinine gir:
```bash
cd take_word_project
```
Sanal ortam oluştur ve aktif et (Önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux için
# venv\Scripts\activate   # Windows için
```
Gerekli kütüphaneleri yükle:
```bash
pip install numpy pandas scikit-learn librosa sounddevice pyaudio pyttsx3 joblib
```
Uygulamayı çalıştır (Canlı Dinleme):
```bash
python live_detect.py
```

🎯 Amaç
Bu proje;

- Ses sinyali işleme ve MFCC özellik çıkarımı mantığını kavramak,
- Makine öğrenmesi modelleri ile kendi verin üzerinden sınıflandırma problemlerini çözmek,
- Mikrofondan alınan anlık sesleri (stream) işleyerek asenkron bir asistan altyapısı kurmak,
- Özel uyanma kelimesi (wake word) tespiti ve sesli geri bildirim sistemlerini geliştirmek amacıyla tasarlanmıştır.
