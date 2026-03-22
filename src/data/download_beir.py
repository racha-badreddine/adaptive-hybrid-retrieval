from pathlib import Path
import os
import time
import zipfile

import requests


BEIR_DATASET_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
    "trec-covid": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
}

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_beir_dataset(data_dir: str, dataset_name: str) -> Path:
    """
    Download and extract a BEIR dataset into data_dir.

    Args:
        data_dir: root data directory
        dataset_name: BEIR dataset name, e.g. 'scifact'

    Returns:
        Path to extracted dataset directory
    """
    dataset_url = BEIR_DATASET_URLS.get(
        dataset_name,
        f"{BEIR_BASE_URL}/{dataset_name}.zip",
    )

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_path = data_root / dataset_name
    zip_path = data_root / f"{dataset_name}.zip"

    if dataset_path.exists():
        print(f"Dataset already exists at: {dataset_path}")
        return dataset_path

    print(f"Downloading {dataset_name}...")
    download_file(dataset_url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    print(f"Removing zip file: {zip_path}")
    # Windows can briefly keep a lock after extraction; retry a few times.
    for attempt in range(5):
        try:
            os.remove(zip_path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)

    print(f"Dataset ready at: {dataset_path}")
    return dataset_path