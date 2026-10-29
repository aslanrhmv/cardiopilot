from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from torch import nn

from ecg_model import CardioPilotECGNet, TARGET_LABELS


torch.manual_seed(42)

NUMBER_OF_EPOCHS = 100
THRESHOLD = 0.5

project_root = Path(__file__).resolve().parents[1]
experiment_dir = project_root / "work" / "experiments"
raw_data_dir = project_root / "data" / "raw"
model_dir = project_root / "work" / "models"

train_list_path = experiment_dir / "train_experiment_samples.csv"
validation_list_path = experiment_dir / "validation_experiment_samples.csv"

train_signal_dir = raw_data_dir / "experiment_train"
validation_signal_dir = raw_data_dir / "experiment_validation"

metrics_path = experiment_dir / "experiment_metrics.csv"
predictions_path = experiment_dir / "experiment_validation_predictions.csv"
chart_path = experiment_dir / "experiment_loss.png"
model_path = model_dir / "experiment_model.pt"


def load_group(list_path, signal_dir):
    samples = pd.read_csv(list_path)

    signals = []
    targets = []

    for _, sample in samples.iterrows():
        record_name = Path(sample["filename_lr"]).name
        signal, _ = wfdb.rdsamp(str(signal_dir / record_name))

        normalized_signal = (
            signal - signal.mean(axis=0)
        ) / signal.std(axis=0)

        signals.append(normalized_signal.astype(np.float32))
        targets.append(sample[TARGET_LABELS].to_numpy(dtype=np.float32))

    signals_tensor = torch.from_numpy(
        np.stack(signals)
    ).permute(0, 2, 1).float()

    targets_tensor = torch.from_numpy(np.stack(targets)).float()

    return samples, signals_tensor, targets_tensor


train_samples, train_signals, train_targets = load_group(
    train_list_path,
    train_signal_dir,
)

validation_samples, validation_signals, validation_targets = load_group(
    validation_list_path,
    validation_signal_dir,
)

model = CardioPilotECGNet(number_of_labels=len(TARGET_LABELS))
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []

model.train()

for epoch in range(1, NUMBER_OF_EPOCHS + 1):
    logits = model(train_signals)
    loss = loss_function(logits, train_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch in [1, 10, 50, 100]:
        print(f"Tur {epoch:03d} - egitim hatasi: {loss.item():.6f}")

model.eval()

with torch.no_grad():
    validation_logits = model(validation_signals)
    validation_probabilities = torch.sigmoid(
        validation_logits
    ).numpy()

validation_targets_array = validation_targets.numpy().astype(int)
validation_predictions = (
    validation_probabilities >= THRESHOLD
).astype(int)

metric_rows = []

for label_index, label in enumerate(TARGET_LABELS):
    true_values = validation_targets_array[:, label_index]
    predicted_values = validation_predictions[:, label_index]

    true_positive = int(
        ((true_values == 1) & (predicted_values == 1)).sum()
    )
    false_positive = int(
        ((true_values == 0) & (predicted_values == 1)).sum()
    )
    false_negative = int(
        ((true_values == 1) & (predicted_values == 0)).sum()
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    metric_rows.append(
        {
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": int(true_values.sum()),
        }
    )

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(metrics_path, index=False)

predictions = validation_samples[
    ["ecg_id", "filename_lr", "selection_label"]
].copy()

for label_index, label in enumerate(TARGET_LABELS):
    predictions[f"target_{label}"] = (
        validation_targets_array[:, label_index]
    )
    predictions[f"probability_{label}"] = (
        validation_probabilities[:, label_index]
    )
    predictions[f"prediction_{label}"] = (
        validation_predictions[:, label_index]
    )

predictions.to_csv(predictions_path, index=False)

plt.figure(figsize=(8, 5))
plt.plot(range(1, NUMBER_OF_EPOCHS + 1), loss_history)
plt.title("CardioPilot Deney Egitim Hatasi")
plt.xlabel("Egitim turu")
plt.ylabel("BCEWithLogitsLoss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()

model_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "target_labels": TARGET_LABELS,
        "number_of_epochs": NUMBER_OF_EPOCHS,
        "threshold": THRESHOLD,
    },
    model_path,
)

print("\nEgitim sinyal boyutu:", tuple(train_signals.shape))
print("Dogrulama sinyal boyutu:", tuple(validation_signals.shape))
print("\nKucuk deney dogrulama metrikleri:")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nMetrikler kaydedildi: {metrics_path}")
print(f"Tahminler kaydedildi: {predictions_path}")
print(f"Grafik kaydedildi: {chart_path}")
print(f"Model kaydedildi: {model_path}")
print(
    "\nNot: Bu metrikler cok kucuk deney seti icindir; "
    "klinik performans veya tani sonucu degildir."
)