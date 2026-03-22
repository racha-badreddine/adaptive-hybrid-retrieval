from collections import Counter


def analyze_oracle_data(oracle_data):
    total = len(oracle_data)
    informative = sum(1 for x in oracle_data.values() if x["is_informative"])
    non_informative = total - informative

    alpha_counter = Counter(
        x["best_alpha"] for x in oracle_data.values() if x["is_informative"]
    )

    print("\nOracle Analysis")
    print(f"Total queries: {total}")
    print(f"Informative queries: {informative}")
    print(f"Non-informative queries: {non_informative}")
    print("Best alpha distribution (informative only):")
    for alpha, count in sorted(alpha_counter.items()):
        print(f"  alpha={alpha:.1f}: {count}")