import ast
from pathlib import Path

import pandas as pd

TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def split_name(strat_fold):
    if 1 <= strat_fold <= 8:
        return "train"
    if strat_fold == 9:
        return "validation"
    return "test"


project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
output_dir = project_root / "work" / "prepared"

database = pd.read_csv(raw_dir / "ptbxl_database.csv")
statements = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)

code_to_class = statements["diagnostic_class"].dropna().to_dict()


def extract_labels(scp_codes_text):
    scp_codes = ast.literal_eval(scp_codes_text)

    return {
        code_to_class[code]
        for code, score in scp_codes.items()
        if score > 0
        and code in code_to_class
        and code_to_class[code] in TARGET_LABELS
    }


label_sets = database["scp_codes"].map(extract_labels)

prepared = database[
    ["ecg_id", "patient_id", "filename_lr", "strat_fold"]
].copy()

prepared["split"] = prepared["strat_fold"].map(split_name)

for label in TARGET_LABELS:
    prepared[label] = label_sets.map(lambda labels: int(label in labels))

output_path = output_dir / "ptbxl_five_class_labels.csv"
prepared.to_csv(output_path, index=False)

print("Hazır tablo oluşturuldu:", output_path)
print("\nGruplara göre kayıt sayısı:")
print(prepared["split"].value_counts().to_string())

print("\nHedef etiketlerin kayıt sayıları:")
print(prepared[TARGET_LABELS].sum().sort_values(ascending=False).to_string())

multi_label_count = (prepared[TARGET_LABELS].sum(axis=1) > 1).sum()
print("\nBirden fazla etiketi olan kayıt:", multi_label_count)