from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "work" / "validation" / "validation_samples.csv"
output_dir = project_root / "data" / "raw" / "validation_samples"

samples = pd.read_csv(input_path)

output_dir.mkdir(parents=True, exist_ok=True)

for _, sample in samples.iterrows():
    filename = sample["filename_lr"]

    for extension in [".dat", ".hea"]:
        source_url = f"{BASE_URL}/{filename}{extension}"
        destination = output_dir / f"{Path(filename).name}{extension}"

        response = requests.get(source_url, timeout=60)
        response.raise_for_status()

        destination.write_bytes(response.content)
        print(f"Indirildi: {destination.name}")

print(f"\nToplam dosya klasoru: {output_dir}")