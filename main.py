from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.retrieval.hybrid_static import fuse_static
from beir.retrieval.evaluation import EvaluateRetrieval
from src.features.query_features import compute_idf_dict, extract_query_features


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

    for alpha in [0.2, 0.5, 0.8]:
        results_hybrid = fuse_static(
            bm25_results=results_bm25,
            dense_results=results_dense,
            alpha=alpha,
            top_k=100,
        )
        print_metrics(f"Static Hybrid (alpha={alpha})", qrels, results_hybrid)
        
        idf_dict = compute_idf_dict(corpus)

    sample_qid = list(queries.keys())[0]
    sample_features = extract_query_features(
        query_text=queries[sample_qid],
        bm25_results_for_query=results_bm25[sample_qid],
        dense_results_for_query=results_dense[sample_qid],
        idf_dict=idf_dict,
    )

    print("\nSample Query Features")
    print(f"Query ID: {sample_qid}")
    print(sample_features)


if __name__ == "__main__":
    main()