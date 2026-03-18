from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from beir.retrieval.evaluation import EvaluateRetrieval


def print_metrics(name, qrels, results):
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 3, 5, 10]
    )

    print(f"\n{name} Results")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"MAP@10: {_map['MAP@10']:.4f}")
    print(f"P@10: {precision['P@10']:.4f}")


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
    print_metrics("BM25", qrels, results_bm25)

    results_dense = run_dense_retrieval(
        corpus,
        queries,
        model_name="all-MiniLM-L6-v2",
        top_k=100,
    )
    print_metrics("Dense", qrels, results_dense)


if __name__ == "__main__":
    main()