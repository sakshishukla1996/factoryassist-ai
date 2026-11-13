
import argparse
import json
import re
import statistics
import time
from pathlib import Path

from factoryassist.orchestration.graph import run_workflow
from factoryassist.observability.langfuse_client import flush_langfuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/evaluation/questions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/generation_results.jsonl",
    )
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]

    if not rows:
        raise ValueError("No evaluation questions selected")

    results = []

    try:
        for i, row in enumerate(rows, start=1):
            start = time.perf_counter()
            result = run_workflow(
                question=row["question"],
                expected_status=row["expected_status"],
                expected_sources=row["relevant_sources"],
)
            latency = time.perf_counter() - start

            answer = result["answer"]
            sources = result.get("sources", [])
            retrieved_paths = {s["path"] for s in sources}
            gold_paths = set(row["relevant_sources"])

            citations = re.findall(r"\[(\d+)\]", answer)
            valid_ids = {str(s["id"]) for s in sources}

            record = {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "generated_answer": answer,
                "expected_status": row["expected_status"],
                "actual_status": result["status"],
                "status_correct": (
                    result["status"] == row["expected_status"]),
                "step_citation_coverage": step_citation_coverage(answer),
                "source_recall": (
                    len(retrieved_paths & gold_paths)
                    / len(gold_paths)
                    if gold_paths else None
                ),
                "citations_present": bool(citations),
                "citation_ids_valid": (
                    bool(citations)
                    and all(c in valid_ids for c in citations)
                ),
                "latency_seconds": round(latency, 3),
                "retrieved_sources": sorted(retrieved_paths),
            }
            results.append(record)

            print(
                f"[{i:02d}/{len(rows)}] {row['id']} "
                f"status_ok={record['status_correct']} "
                f"latency={latency:.2f}s"
            )
    finally:
        flush_langfuse()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for record in results:
            file.write(json.dumps(record) + "\n")

    status_accuracy = statistics.mean(
        r["status_correct"] for r in results
    )
    source_recalls = [
        r["source_recall"] for r in results
        if r["source_recall"] is not None
    ]
    positive_results = [
        r for r in results if r["expected_status"] == "answered"
    ]
    latencies = sorted(
        r["latency_seconds"] for r in results
    )
    p95_index = max(
        0, (95 * len(latencies) + 99) // 100 - 1
    )

    print("\nGENERATION EVALUATION")
    print(f"Questions evaluated: {len(results)}")
    print(f"Status accuracy: {status_accuracy:.3f}")

    if source_recalls:
        print(
            "Mean source recall: "
            f"{statistics.mean(source_recalls):.3f}"
        )

    if positive_results:
        citation_rate = statistics.mean(
            r["citations_present"] for r in positive_results
        )
        print(f"Citation presence: {citation_rate:.3f}")

    print(
        f"Mean latency: "
        f"{statistics.mean(latencies):.2f}s"
    )
    print(f"P95 latency: {latencies[p95_index]:.2f}s")
    print(f"Results saved to: {output}")


def step_citation_coverage(answer: str) -> float | None:
    """Measure citation coverage of numbered and bullet steps."""
    steps = [
        line.strip()
        for line in answer.splitlines()
        if re.match(r"^\s*(?:\d+[.)]|[-*])\s+", line)
    ]

    if not steps:
        return None

    cited_steps = sum(
        bool(re.search(r"\[\d+\]", step))
        for step in steps
    )

    return cited_steps / len(steps)



if __name__ == "__main__":
    main()
