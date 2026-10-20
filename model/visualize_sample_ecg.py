"""PTB-XL örnek EKG kaydını görselleştirir.

Bu betik yalnızca araştırma/eğitim amaçlı sinyal incelemesi yapar.
Tıbbi tanı veya klinik karar vermez.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = PROJECT_ROOT / "data" / "raw" / "sample_ecg" / "00001_lr"

record = wfdb.rdrecord(str(RECORD_PATH))
signal = record.p_signal
sampling_frequency = record.fs

print("Derivasyonlar:", ", ".join(record.sig_name))
print("Sinyal boyutu:", signal.shape)
print("Örnekleme hızı:", sampling_frequency, "Hz")

# II derivasyonu, klinik EKG incelemelerinde sık kullanılan kanallardan biridir.
lead_name = "II"
lead_index = record.sig_name.index(lead_name)
time_seconds = np.arange(signal.shape[0]) / sampling_frequency

plt.figure(figsize=(12, 4))
plt.plot(time_seconds, signal[:, lead_index], linewidth=0.8)
plt.title("PTB-XL Örnek Kaydı - Derivasyon II")
plt.xlabel("Zaman (saniye)")
plt.ylabel("Genlik (mV)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
