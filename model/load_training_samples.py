from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
RECORD_IDS = [1, 22, 32, 45, 50]


def normalize_per_lead(signal):
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)
    safe_std = np.where(std == 0, 1, std)

    return (signal - mean) / safe_std


project_root = Path(__file__).resolve().parents[1]
sample_dir = project_root / "data" / "raw" / "training_samples"
labels_path = project_root / "work" / "prepared" / "ptbxl_five_class_labels.csv"
output_dir = project_root / "work" / "batches"

output_dir.mkdir(parents=True, exist_ok=True)

label_table = pd.read_csv(labels_path).set_index("ecg_id")
selected_labels = label_table.loc[RECORD_IDS, TARGET_LABELS]

signals = []
lead_names = None
sampling_rate = None

for record_id in RECORD_IDS:
    record_path = sample_dir / f"{record_id:05d}_lr"

    signal, fields = wfdb.rdsamp(str(record_path))
    normalized_signal = normalize_per_lead(signal).astype(np.float32)

    signals.append(normalized_signal)

    if lead_names is None:
        lead_names = fields["sig_name"]
        sampling_rate = fields["fs"]

batch_signals = np.stack(signals)
batch_labels = selected_labels.to_numpy(dtype=np.uint8)

output_path = output_dir / "training_samples_batch.npz"

np.savez_compressed(
    output_path,
    signals=batch_signals,
    labels=batch_labels,
    ecg_ids=np.array(RECORD_IDS),
    label_names=np.array(TARGET_LABELS),
    lead_names=np.array(lead_names),
    sampling_rate=sampling_rate,
)

print("Mini batch kaydedildi:", output_path)
print("Sinyal boyutu:", batch_signals.shape)
print("Etiket boyutu:", batch_labels.shape)

print("\nKayitlarin etiketleri:")
print(selected_labels.to_string())