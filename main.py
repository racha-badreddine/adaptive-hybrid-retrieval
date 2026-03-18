from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from beir.retrieval.evaluation import EvaluateRetrieval


def main():
    dataset_name = "scifact"
    data_dir = "data"

    download_beir_dataset(data_dir=data_dir, dataset_name=dataset_name)

    corpus, queries, qrels = load_beir_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split="test",
    )

    print(f"Loaded {len(corpus)} documents")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(qrels)} qrels")

    results_bm25 = run_bm25(corpus, queries, top_k=100)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_bm25, [1, 3, 5, 10]
    )

    print("\nBM25 Results")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"MAP@10: {_map['MAP@10']:.4f}")
    print(f"P@10: {precision['P@10']:.4f}")


if __name__ == "__main__":
    main()