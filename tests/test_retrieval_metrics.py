from factoryassist.evaluation.metrics import recall_at_k, reciprocal_rank_at_k, unique_paths


def test_recall():
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5


def test_mrr():
    assert reciprocal_rank_at_k(["a", "b"], ["b"], 2) == 0.5


def test_miss():
    assert reciprocal_rank_at_k(["a"], ["b"], 1) == 0


def test_deduplicate_chunk_sources():
    assert unique_paths(["a", "a", "b"]) == ["a", "b"]
