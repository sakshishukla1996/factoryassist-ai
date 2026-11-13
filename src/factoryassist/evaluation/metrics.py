"""Pure-Python, document-level retrieval metrics."""
from collections.abc import Iterable


def unique_paths(paths: Iterable[str]) -> list[str]:
    """Keep first occurrence of a source file; retrieval may return multiple chunks."""
    return list(dict.fromkeys(paths))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        raise ValueError("Recall is undefined for no-evidence questions")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        raise ValueError("MRR is undefined for no-evidence questions")
    target = set(relevant)
    for rank, path in enumerate(retrieved[:k], start=1):
        if path in target:
            return 1.0 / rank
    return 0.0
