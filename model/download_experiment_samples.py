from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"

project_root = Path(__file__).resolve().parents[1]
experiment_dir = project_root / "work" / "experiments"
raw_data_dir = project_root / "data" / "raw"

sample_groups = {
    "train": (
        experiment_dir / "train_experiment_samples.csv",
        raw_data_dir / "experiment_train",
    ),
    "validation": (
        experiment_dir / "validation_experiment_samples.csv",
        raw_data_dir / "experiment_validation",
    ),
}

for group_name, (list_path, output_dir) in sample_groups.items():
    samples = pd.read_csv(list_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{group_name.upper()} grubu indiriliyor:")

    for _, sample in samples.iterrows():
        filename = sample["filename_lr"]

        for extension in [".dat", ".hea"]:
            source_url = f"{BASE_URL}/{filename}{extension}"
            destination = output_dir / f"{Path(filename).name}{extension}"

            if destination.exists():
                print(f"Zaten var: {destination.name}")
                continue

            response = requests.get(source_url, timeout=60)
            response.raise_for_status()

            destination.write_bytes(response.content)
            print(f"Indirildi: {destination.name}")

print("\nTum deney EKG dosyalari hazir.")