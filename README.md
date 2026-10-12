# CardioPilot

CardioPilot, 12 derivasyonlu EKG kayıtlarını yapay zekâ ile analiz ederek olası bulgu sınıflarını ve model güven bilgisini hekime açıklanabilir biçimde sunmayı hedefleyen araştırma amaçlı bir klinik karar-destek prototipidir.

## İlk sürümün amacı

- Anonim açık veri setindeki EKG kayıtlarını okumak ve görselleştirmek.
- Seçilen EKG bulgu sınıfları için bir makine öğrenmesi modeli eğitmek.
- Sonucu web arayüzünde, olasılık bilgisi ve açık uyarılarla göstermek.
- Doktorun incelemesine yardımcı olacak PDF ön rapor üretmek.

## İlk sürümün sınırları

- Sistem tanı koymaz ve doktorun kararının yerine geçmez.
- Gerçek hasta verisi kullanılmaz; yalnızca anonim araştırma verileri kullanılır.
- Hastane bilgi sistemi, mobil uygulama ve MR/EEG entegrasyonu ilk sürümün kapsamı dışındadır.

## Hedef kullanıcı

Kardiyoloji uzmanı veya EKG inceleyen sağlık çalışanı.

## Teknoloji planı

- Yapay zekâ ve API: Python, PyTorch, FastAPI
- Web arayüzü: TypeScript, React, Next.js
- Veritabanı: PostgreSQL
- Dağıtım: Docker

