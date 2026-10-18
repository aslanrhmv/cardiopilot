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

## Proje yapısı

- `frontend/`: Kullanıcı arayüzü; ileride React ve Next.js ile geliştirilecek.
- `backend/`: Uygulamanın API ve sunucu tarafı; ileride Python ve FastAPI ile geliştirilecek.
- `model/`: EKG veri işleme ve yapay zekâ modeli; araştırma verileri ile geliştirilecek.

Bu klasörler başlangıç hazırlığıdır; henüz uygulama veya eğitilmiş model içermez.

## Geliştirme araçlarını kontrol etme

PowerShell üzerinde aşağıdaki komutlar sürüm bilgisi göstermelidir:

```powershell
python --version
python -m pip --version
node --version
npm.cmd --version
git --version
docker --version
```

Python ve paketlerini projeye özel bir sanal ortamda kullanmak için proje klasöründe:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip --version
```

Sanal ortam `.gitignore` kapsamındadır ve GitHub'a yüklenmez.

