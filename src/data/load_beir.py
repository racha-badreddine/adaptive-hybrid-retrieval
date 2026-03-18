from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import os


def load_beir_dataset(data_dir: str, dataset_name: str, split: str = "test"):
    """
    Load a BEIR dataset from local storage.

    Args:
        data_dir: Root directory where datasets are stored.
        dataset_name: Name of the BEIR dataset (e.g., 'scifact', 'quora').
        split: Dataset split to load ('train', 'dev', 'test').

    Returns:
        corpus: dict of documents
        queries: dict of queries
        qrels: dict of relevance judgments
    """
    dataset_path = Path(data_dir) / dataset_name

    if not dataset_path.exists():
        print(f"Dataset '{dataset_name}' not found. Downloading to {data_dir}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)
    return corpus, queries, qrels