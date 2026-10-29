from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg_model import CardioPilotECGNet, TARGET_LABELS


torch.manual_seed(42)

project_root = Path(__file__).resolve().parents[1]
batch_path = project_root / "work" / "batches" / "training_samples_batch.npz"
model_output_dir = project_root / "work" / "models"
model_output_path = model_output_dir / "first_training_step.pt"

batch_data = np.load(batch_path)

# NumPy: (batch, zaman, derivasyon)
# PyTorch: (batch, derivasyon, zaman)
signals = torch.from_numpy(batch_data["signals"]).permute(0, 2, 1).float()
targets = torch.from_numpy(batch_data["labels"]).float()

model = CardioPilotECGNet(number_of_labels=len(TARGET_LABELS))

# Çok etiketli problemde her etiket için ayrı hata hesaplanır.
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

logits_before = model(signals)
loss_before = loss_function(logits_before, targets)

optimizer.zero_grad()
loss_before.backward()
optimizer.step()

with torch.no_grad():
    logits_after = model(signals)
    loss_after = loss_function(logits_after, targets)
    probabilities_after = torch.sigmoid(logits_after)

model_output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "target_labels": TARGET_LABELS,
        "sampling_rate": int(batch_data["sampling_rate"]),
    },
    model_output_path,
)

print("Sinyal boyutu:", tuple(signals.shape))
print("Etiket boyutu:", tuple(targets.shape))
print(f"Egitim oncesi hata: {loss_before.item():.6f}")
print(f"Bir adim sonrasi hata: {loss_after.item():.6f}")
print(f"Model kaydedildi: {model_output_path}")

print("\nIlk EKG icin guncellenmis olasiliklar:")
for label, target, probability in zip(
    TARGET_LABELS,
    targets[0],
    probabilities_after[0],
):
    print(
        f"{label}: hedef={int(target.item())}, "
        f"olasilik={probability.item():.4f}"
    )