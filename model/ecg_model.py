from pathlib import Path

import numpy as np
import torch
from torch import nn


TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


class CardioPilotECGNet(nn.Module):
    def __init__(self, number_of_labels=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, number_of_labels)

    def forward(self, signals):
        features = self.features(signals)
        features = features.squeeze(-1)
        return self.classifier(features)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    batch_path = project_root / "work" / "batches" / "training_samples_batch.npz"

    batch_data = np.load(batch_path)

    # NumPy: (batch, zaman, derivasyon)
    # PyTorch Conv1d: (batch, derivasyon, zaman)
    signals = torch.from_numpy(batch_data["signals"]).permute(0, 2, 1)

    model = CardioPilotECGNet(number_of_labels=len(TARGET_LABELS))
    model.eval()

    with torch.no_grad():
        logits = model(signals)
        probabilities = torch.sigmoid(logits)

    print("Model girdisi boyutu:", tuple(signals.shape))
    print("Model ham cikti boyutu:", tuple(logits.shape))
    print("Olasilik boyutu:", tuple(probabilities.shape))

    print("\nIlk EKG icin egitilmemis model olasiliklari:")
    for label, probability in zip(TARGET_LABELS, probabilities[0]):
        print(f"{label}: {float(probability):.4f}")