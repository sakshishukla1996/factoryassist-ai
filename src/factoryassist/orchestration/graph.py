from langgraph.graph import StateGraph, START, END

from factoryassist.orchestration.state import FactoryState
from factoryassist.orchestration.nodes import (
    validate_question,
    retrieve_documents,
    generate_answer,
    insufficient_evidence,
)
from factoryassist.observability.langfuse_client import (
    get_langfuse_handler,
    flush_langfuse,
)

from langfuse import observe, get_client

from factoryassist.evaluation.quality_scoring import (
    record_quality_scores,
)


builder = StateGraph(FactoryState)

builder.add_node("validate", validate_question)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)
builder.add_node("fallback", insufficient_evidence)

builder.add_edge(START, "validate")


def after_validation(state: FactoryState) -> str:
    if state["status"] == "valid":
        return "retrieve"
    return "end"


builder.add_conditional_edges(
    "validate",
    after_validation,
    {
        "retrieve": "retrieve",
        "end": END,
    },
)


def after_retrieval(state: FactoryState) -> str:
    if state["status"] == "found":
        return "generate"
    return "fallback"


builder.add_conditional_edges(
    "retrieve",
    after_retrieval,
    {
        "generate": "generate",
        "fallback": "fallback",
    },
)

builder.add_edge("generate", END)
builder.add_edge("fallback", END)

workflow = builder.compile()


@observe(name="factoryassist-orchestrator")
def run_workflow(
    question: str,
    expected_status: str | None = None,
    expected_sources: list[str] | None = None,
) -> dict:

    langfuse = get_client()
    trace_id = langfuse.get_current_trace_id()

    if not trace_id:
        raise RuntimeError("No active Langfuse trace found")

    print(f"FactoryAssist trace ID: {trace_id}")

    handler = get_langfuse_handler()

    result = workflow.invoke(
        {"question": question},
        config={"callbacks": [handler]},
    )

    response = {
        "answer": result["answer"],
        "sources": result.get("documents", []),
        "status": result["status"],
    }

    record_quality_scores(
        result=response,
        trace_id=trace_id,
        expected_status=expected_status,
        expected_sources=expected_sources,
    )

    return response

