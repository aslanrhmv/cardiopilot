from pathlib import Path

import numpy as np
import wfdb


def normalize_per_lead(signal):
    """Her EKG derivasyonunu ayrı ayrı z-skor ile normalleştirir."""
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)

    # Sıfır standart sapma ihtimaline karşı güvenli değer kullanılır.
    safe_std = np.where(std == 0, 1, std)

    return (signal - mean) / safe_std


project_root = Path(__file__).resolve().parents[1]

input_record = project_root / "data" / "raw" / "sample_ecg" / "00001_lr"
output_dir = project_root / "work" / "preprocessed"
output_dir.mkdir(parents=True, exist_ok=True)

signal, fields = wfdb.rdsamp(str(input_record))
normalized_signal = normalize_per_lead(signal).astype(np.float32)

output_path = output_dir / "00001_lr_normalized.npz"

np.savez_compressed(
    output_path,
    signal=normalized_signal,
    lead_names=np.array(fields["sig_name"]),
    sampling_rate=fields["fs"],
)

print("Normalleştirilmiş EKG kaydedildi:", output_path)
print("Sinyal boyutu:", normalized_signal.shape)
print("Örnekleme hızı:", fields["fs"], "Hz")