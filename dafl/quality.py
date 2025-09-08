from typing import List

def compute_data_quality_scores(val_accs: List[float], sample_counts: List[int],
                                weight: float) -> list[float]:
    max_s = max(sample_counts) if sample_counts else 1
    scores = []
    for acc, cnt in zip(val_accs, sample_counts):
        scores.append(acc + weight * (cnt / max_s))
    return scores