from pathlib import Path

import pandas as pd


TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "work" / "prepared" / "ptbxl_five_class_labels.csv"
output_dir = project_root / "work" / "validation"
output_path = output_dir / "validation_samples.csv"

data = pd.read_csv(input_path)
validation_data = data[data["split"] == "validation"].copy()

selected_rows = []
used_ecg_ids = set()

for label in TARGET_LABELS:
    candidates = validation_data[
        (validation_data[label] == 1)
        & (~validation_data["ecg_id"].isin(used_ecg_ids))
    ]

    selected_row = candidates.sort_values("ecg_id").iloc[0]
    selected_rows.append(selected_row)
    used_ecg_ids.add(selected_row["ecg_id"])

selected = pd.DataFrame(selected_rows)

output_dir.mkdir(parents=True, exist_ok=True)
selected.to_csv(output_path, index=False)

print("Secilen dogrulama kayitlari:")
print(selected[["ecg_id", "filename_lr"] + TARGET_LABELS].to_string(index=False))
print(f"\nDosya kaydedildi: {output_path}")