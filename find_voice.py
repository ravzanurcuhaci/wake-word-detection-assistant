import pyttsx3
#sesi türkçe  gibi söylemesi için
engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Sistemdeki Yüklü Sesler:")
for voice in voices:
    if 'turkish' in voice.name.lower() or 'tr' in voice.id.lower():
        print(f"BULUNDU: ID: {voice.id} - İsim: {voice.name}")
    else:
        # print(f"Bulunamadı: ID: {voice.id} - İsim: {voice.name}")
        pass

print("\nTürkçe ses bulunamazsa işletim sisteminde yüklü değildir.")
