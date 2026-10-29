from pathlib import Path

import pandas as pd


TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLES_PER_LABEL = 3

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "work" / "prepared" / "ptbxl_five_class_labels.csv"
output_dir = project_root / "work" / "experiments"

data = pd.read_csv(input_path)


def select_samples(split_name):
    split_data = data[data["split"] == split_name].copy()
    selected_rows = []
    used_ecg_ids = set()

    for label in TARGET_LABELS:
        candidates = split_data[
            (split_data[label] == 1)
            & (~split_data["ecg_id"].isin(used_ecg_ids))
        ].sort_values("ecg_id")

        for _, row in candidates.head(SAMPLES_PER_LABEL).iterrows():
            selected_row = row.copy()
            selected_row["selection_label"] = label
            selected_rows.append(selected_row)
            used_ecg_ids.add(row["ecg_id"])

    return pd.DataFrame(selected_rows)


train_samples = select_samples("train")
validation_samples = select_samples("validation")

output_dir.mkdir(parents=True, exist_ok=True)

train_path = output_dir / "train_experiment_samples.csv"
validation_path = output_dir / "validation_experiment_samples.csv"

train_samples.to_csv(train_path, index=False)
validation_samples.to_csv(validation_path, index=False)

print("Egitim kayit sayisi:", len(train_samples))
print("Dogrulama kayit sayisi:", len(validation_samples))

print("\nEgitim grubuna gore secim:")
print(train_samples["selection_label"].value_counts().sort_index().to_string())

print("\nDogrulama grubuna gore secim:")
print(validation_samples["selection_label"].value_counts().sort_index().to_string())

print(f"\nEgitim listesi: {train_path}")
print(f"Dogrulama listesi: {validation_path}")