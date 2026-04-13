import numpy as np
from typing import Dict


def analyze_oracle_data(oracle_data: Dict) -> Dict:
    """
    Print and return summary statistics for oracle alpha data.

    Returns a dict with: total, informative, non_informative, mean_alpha,
    std_alpha, bin_counts.
    """
    total = len(oracle_data)
    informative = sum(1 for x in oracle_data.values() if x["is_informative"])
    non_informative = total - informative

    informative_alphas = [
        x["best_alpha"] for x in oracle_data.values() if x["is_informative"]
    ]

    mean_alpha = float(np.mean(informative_alphas)) if informative_alphas else 0.0
    std_alpha = float(np.std(informative_alphas)) if informative_alphas else 0.0

    # [0.0, 0.2] then (0.2, 0.4] (0.4, 0.6] (0.6, 0.8] (0.8, 1.0]
    bins = [
        ("0.0-0.2", lambda a: 0.0 <= a <= 0.2),
        ("0.2-0.4", lambda a: 0.2 < a <= 0.4),
        ("0.4-0.6", lambda a: 0.4 < a <= 0.6),
        ("0.6-0.8", lambda a: 0.6 < a <= 0.8),
        ("0.8-1.0", lambda a: 0.8 < a <= 1.0),
    ]
    bin_counts = {}
    for label, pred in bins:
        bin_counts[label] = sum(1 for a in informative_alphas if pred(a))

    pct_informative = 100.0 * informative / total if total > 0 else 0.0

    print("\nOracle Analysis")
    print(f"  Total queries:          {total}")
    print(f"  Informative queries:    {informative} ({pct_informative:.1f}%)")
    print(f"  Non-informative:        {non_informative}")
    print(f"  Mean oracle alpha:      {mean_alpha:.4f}")
    print(f"  Std  oracle alpha:      {std_alpha:.4f}")
    print("  Alpha distribution (informative only):")
    for label, count in bin_counts.items():
        pct = 100.0 * count / informative if informative > 0 else 0.0
        print(f"    [{label}]: {count:4d}  ({pct:.1f}%)")

    return {
        "total": total,
        "informative": informative,
        "non_informative": non_informative,
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "bin_counts": bin_counts,
    }
