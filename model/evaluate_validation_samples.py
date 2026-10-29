from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_model import CardioPilotECGNet, TARGET_LABELS


project_root = Path(__file__).resolve().parents[1]

samples_path = project_root / "work" / "validation" / "validation_samples.csv"
signal_dir = project_root / "data" / "raw" / "validation_samples"
model_path = project_root / "work" / "models" / "multiepoch_training.pt"
output_path = project_root / "work" / "validation" / "validation_predictions.csv"

samples = pd.read_csv(samples_path)

signals = []
targets = []

for _, sample in samples.iterrows():
    record_name = Path(sample["filename_lr"]).name
    signal, _ = wfdb.rdsamp(str(signal_dir / record_name))

    # Eğitimdekiyle aynı normalleştirme uygulanır.
    normalized_signal = (
        signal - signal.mean(axis=0)
    ) / signal.std(axis=0)

    signals.append(normalized_signal.astype(np.float32))
    targets.append(sample[TARGET_LABELS].to_numpy(dtype=np.float32))

signals_tensor = torch.from_numpy(np.stack(signals)).permute(0, 2, 1)
targets_array = np.stack(targets)

checkpoint = torch.load(model_path, map_location="cpu")

model = CardioPilotECGNet(number_of_labels=len(TARGET_LABELS))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(signals_tensor)
    probabilities = torch.sigmoid(logits).numpy()

results = samples[["ecg_id", "filename_lr"]].copy()

for index, label in enumerate(TARGET_LABELS):
    results[f"target_{label}"] = targets_array[:, index].astype(int)
    results[f"probability_{label}"] = probabilities[:, index]

results.to_csv(output_path, index=False)

print("Dogrulama sinyal boyutu:", tuple(signals_tensor.shape))
print("\nHer kayit icin gercek etiket ve en yuksek model olasiligi:")

for row_index, row in results.iterrows():
    true_labels = [
        label for label in TARGET_LABELS
        if row[f"target_{label}"] == 1
    ]

    highest_label = max(
        TARGET_LABELS,
        key=lambda label: row[f"probability_{label}"],
    )

    highest_probability = row[f"probability_{highest_label}"]

    print(
        f"EKG {int(row['ecg_id'])} | "
        f"gercek={','.join(true_labels)} | "
        f"en yuksek_cikti={highest_label} "
        f"({highest_probability:.4f})"
    )

print(f"\nSonuclar kaydedildi: {output_path}")
print(
    "Not: Bu olasiliklar kucuk egitim ornegiyle olusan "
    "arastirma prototipi ciktisidir; klinik teshis degildir."
)