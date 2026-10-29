from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLDS = [0.30, 0.50, 0.70]

project_root = Path(__file__).resolve().parents[1]
input_path = (
    project_root
    / "work"
    / "experiments"
    / "experiment_validation_predictions.csv"
)

output_dir = project_root / "work" / "experiments"
metrics_path = output_dir / "threshold_metrics.csv"
chart_path = output_dir / "threshold_comparison.png"

data = pd.read_csv(input_path)

rows = []

for threshold in THRESHOLDS:
    label_metrics = []

    for label in TARGET_LABELS:
        true_values = data[f"target_{label}"]
        predicted_values = (
            data[f"probability_{label}"] >= threshold
        ).astype(int)

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

        metric = {
            "threshold": threshold,
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        rows.append(metric)
        label_metrics.append(metric)

    rows.append(
        {
            "threshold": threshold,
            "label": "MACRO_AVERAGE",
            "precision": sum(
                item["precision"] for item in label_metrics
            ) / len(label_metrics),
            "recall": sum(
                item["recall"] for item in label_metrics
            ) / len(label_metrics),
            "f1_score": sum(
                item["f1_score"] for item in label_metrics
            ) / len(label_metrics),
        }
    )

metrics = pd.DataFrame(rows)
metrics.to_csv(metrics_path, index=False)

summary = metrics[
    metrics["label"] == "MACRO_AVERAGE"
].copy()

print("Esiklere gore makro ortalama metrikleri:")
print(
    summary[
        ["threshold", "precision", "recall", "f1_score"]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

plt.figure(figsize=(8, 5))
plt.plot(
    summary["threshold"],
    summary["precision"],
    marker="o",
    label="Precision",
)
plt.plot(
    summary["threshold"],
    summary["recall"],
    marker="o",
    label="Recall",
)
plt.plot(
    summary["threshold"],
    summary["f1_score"],
    marker="o",
    label="F1 score",
)

plt.title("CardioPilot Esik Karsilastirmasi")
plt.xlabel("Olasilik esigi")
plt.ylabel("Makro ortalama metrik")
plt.xticks(THRESHOLDS)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"\nDetayli metrikler kaydedildi: {metrics_path}")
print(f"Grafik kaydedildi: {chart_path}")
print(
    "Not: Bu analiz kucuk deney veri seti icindir; "
    "klinik karar esigi belirlemez."
)