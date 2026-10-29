import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ecg_model import CardioPilotECGNet, TARGET_LABELS


torch.manual_seed(42)

NUMBER_OF_EPOCHS = 50

project_root = Path(__file__).resolve().parents[1]
batch_path = project_root / "work" / "batches" / "training_samples_batch.npz"
output_dir = project_root / "work" / "training_history"
model_dir = project_root / "work" / "models"

history_path = output_dir / "training_loss.csv"
chart_path = output_dir / "training_loss.png"
model_path = model_dir / "multiepoch_training.pt"

batch_data = np.load(batch_path)

# NumPy: (batch, zaman, derivasyon)
# PyTorch: (batch, derivasyon, zaman)
signals = torch.from_numpy(batch_data["signals"]).permute(0, 2, 1).float()
targets = torch.from_numpy(batch_data["labels"]).float()

model = CardioPilotECGNet(number_of_labels=len(TARGET_LABELS))
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []

model.train()

for epoch in range(1, NUMBER_OF_EPOCHS + 1):
    logits = model(signals)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch in [1, 10, 25, 50]:
        print(f"Tur {epoch:02d} - hata: {loss.item():.6f}")

output_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss"])

    for epoch, loss in enumerate(loss_history, start=1):
        writer.writerow([epoch, f"{loss:.8f}"])

plt.figure(figsize=(8, 5))
plt.plot(range(1, NUMBER_OF_EPOCHS + 1), loss_history, color="#1f77b4")
plt.title("CardioPilot Kucuk Batch Egitim Hatasi")
plt.xlabel("Egitim turu")
plt.ylabel("BCEWithLogitsLoss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "target_labels": TARGET_LABELS,
        "sampling_rate": int(batch_data["sampling_rate"]),
        "number_of_epochs": NUMBER_OF_EPOCHS,
        "final_loss": loss_history[-1],
    },
    model_path,
)

print(f"\nBaslangic hatasi: {loss_history[0]:.6f}")
print(f"Son hata: {loss_history[-1]:.6f}")
print(f"Kayıp geçmişi kaydedildi: {history_path}")
print(f"Grafik kaydedildi: {chart_path}")
print(f"Model kaydedildi: {model_path}")