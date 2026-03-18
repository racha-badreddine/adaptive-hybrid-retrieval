from pathlib import Path
import os
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
    if dataset_name not in BEIR_DATASET_URLS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(BEIR_DATASET_URLS.keys())}"
        )

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_path = data_root / dataset_name
    zip_path = data_root / f"{dataset_name}.zip"

    if dataset_path.exists():
        print(f"Dataset already exists at: {dataset_path}")
        return dataset_path

    print(f"Downloading {dataset_name}...")
    download_file(BEIR_DATASET_URLS[dataset_name], zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    print(f"Removing zip file: {zip_path}")
    os.remove(zip_path)

    print(f"Dataset ready at: {dataset_path}")
    return dataset_path