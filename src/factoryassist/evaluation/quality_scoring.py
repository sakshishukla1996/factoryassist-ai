import re

from langfuse import get_client


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
STEP_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+")


def record_quality_scores(
    result: dict,
    trace_id: str,
    expected_status: str | None = None,
    expected_sources: list[str] | None = None,
) -> None:
    """Attach FactoryAssist quality scores to a Langfuse trace."""

    langfuse = get_client()

    def score(name: str, value: float, comment: str) -> None:
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment,
        )
        print(f"Langfuse score: {name} = {value:.3f}")

    answer = result["answer"]
    sources = result.get("sources", [])

    # 1. Citation-related scores for generated answers
    if result["status"] == "answered":
        cited_ids = CITATION_PATTERN.findall(answer)
        valid_ids = {str(source["id"]) for source in sources}

        score(
            "citation_presence",
            float(bool(cited_ids)),
            "Whether the generated answer contains citations.",
        )

        score(
            "citation_id_validity",
            float(
                bool(cited_ids)
                and all(cid in valid_ids for cid in cited_ids)
            ),
            "Whether all cited IDs exist in the retrieved sources.",
        )

        # 2. Citation coverage for numbered or bulleted steps
        steps = [
            line.strip()
            for line in answer.splitlines()
            if STEP_PATTERN.match(line)
        ]

        if steps:
            cited_steps = sum(
                bool(CITATION_PATTERN.search(step))
                for step in steps
            )

            score(
                "step_citation_coverage",
                cited_steps / len(steps),
                "Fraction of numbered or bulleted steps with citations.",
            )

    # 3. Scores that require labeled evaluation data
    if expected_status is not None:
        score(
            "status_correctness",
            float(result["status"] == expected_status),
            "Whether the workflow selected the expected status.",
        )

    if expected_sources:
        actual_sources = {s["path"] for s in sources}
        relevant_sources = set(expected_sources)

        score(
            "source_recall",
            len(actual_sources & relevant_sources)
            / len(relevant_sources),
            "Fraction of expected source documents retrieved.",
        )
