"""Evaluate the existing LangGraph retrieval node against labelled synthetic questions.

Run AFTER rebuilding the FAISS index with the additional documents:
    uv run python -m factoryassist.rag.ingestion
    uv run python -m factoryassist.evaluation.retrieval_eval

This evaluates retrieval only; it does NOT call Qwen or score generation quality.
"""
import argparse
import json
from pathlib import Path

from factoryassist.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank_at_k,
    unique_paths,
)
from factoryassist.orchestration.nodes import retrieve_documents


def main():
    parser = argparse.ArgumentParser(description="FactoryAssist document retrieval evaluation")
    parser.add_argument("--dataset", default="data/evaluation/questions.jsonl")
    args = parser.parse_args()
    examples = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    positives = [r for r in examples if r["expected_status"] == "answered"]
    negatives = [r for r in examples if r["expected_status"] == "no_evidence"]
    scores = {k: {"recall": [], "mrr": []} for k in (1, 3, 4)}
    negative_mistakes = []
    misses = []

    for i, row in enumerate(examples, 1):
        result = retrieve_documents({"question": row["question"]}, config={})
        retrieved = unique_paths(doc["path"] for doc in result.get("documents", []))
        if row["expected_status"] == "no_evidence":
            if retrieved:
                negative_mistakes.append((row["id"], row["question"], retrieved))
        else:
            for k in scores:
                scores[k]["recall"].append(recall_at_k(retrieved, row["relevant_sources"], k))
                scores[k]["mrr"].append(reciprocal_rank_at_k(retrieved, row["relevant_sources"], k))
            if set(retrieved).isdisjoint(row["relevant_sources"]):
                misses.append((row["id"], row["question"], retrieved))
        print(f"[{i:02d}/{len(examples)}] {row['id']} actual={result['status']} expected={row['expected_status']}")

    print("\nRETRIEVAL EVALUATION (document-level; 4 chunks requested from FAISS)")
    print(f"Positive questions: {len(positives)} | Negative questions: {len(negatives)}")
    for k, values in scores.items():
        recall = sum(values["recall"]) / len(values["recall"])
        mrr = sum(values["mrr"]) / len(values["mrr"])
        print(f"Recall@{k}: {recall:.3f}   MRR@{k}: {mrr:.3f}")
    print(f"False positive rate on negatives: {len(negative_mistakes) / len(negatives):.3f}")
    if misses:
        print("\nPOSITIVE QUESTIONS WITH NO RELEVANT DOCUMENTS RETURNED:")
        for qid, q, sources in misses:
            print(f"- {qid}: {q} | retrieved={sources}")
    if negative_mistakes:
        print("\nNEGATIVE QUESTIONS THAT STILL RETURNED DOCUMENTS:")
        for qid, q, sources in negative_mistakes:
            print(f"- {qid}: {q} | retrieved={sources}")


if __name__ == "__main__":
    main()
