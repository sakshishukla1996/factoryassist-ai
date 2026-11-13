
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from factoryassist.models.provider import get_chat_model
from factoryassist.orchestration.state import FactoryState
from factoryassist.rag.pipeline import PROMPT
from factoryassist.rag.retriever import get_retriever

from factoryassist.models.token_budget import (
    validate_question as check_question_budget,
    build_budgeted_context,
    validate_prompt,
)

def validate_question(state: FactoryState) -> dict:
    question = state["question"].strip()

    if not question:
        return {
            "status": "invalid",
            "answer": "Please enter a maintenance question.",
            "documents": [],
        }

    try:
        check_question_budget(question)

    except ValueError as exc:
        return {
            "status": "invalid",
            "answer": str(exc),
            "documents": [],
        }

    return {
        "question": question,
        "status": "valid",
    }

def retrieve_documents(
    state: FactoryState,
    config: RunnableConfig,
) -> dict:
    question = state["question"]

    # Extract machine IDs and error codes.
    identifiers = set(
        re.findall(
            r"\b[A-Z]-\d{3,5}\b",
            question.upper(),
        )
    )

    # Prototype policy:
    # Require an explicit machine ID or error code.
    if not identifiers:
        return {
            "documents": [],
            "status": "no_evidence",
        }

    # Search the full small prototype corpus.
    candidates = get_retriever(k=24).invoke(
        question,
        config=config,
    )

    # Retain passages matching every requested ID.
    matched = [
        doc
        for doc in candidates
        if all(
            identifier in doc.page_content.upper()
            for identifier in identifiers
        )
    ]

    # Keep the four highest-ranked matching passages.
    matched = matched[:4]

    sources = [
        {
            "id": i,
            "path": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content,
        }
        for i, doc in enumerate(matched, start=1)
    ]

    return {
        "documents": sources,
        "status": "found" if sources else "no_evidence",
    }


def generate_answer(
    state: FactoryState,
    config: RunnableConfig,
) -> dict:
    # 1. Fit retrieved documents into the context budget.
    context, selected_docs = build_budgeted_context(
        state["documents"]
    )

    if not selected_docs:
        return {
            "answer": (
                "The retrieved evidence exceeds the "
                "configured context budget."
            ),
            "documents": [],
            "status": "invalid",
        }

    # 2. Select Ollama or Groq through our model factory.
    llm = get_chat_model()

    # 3. Construct the complete prompt.
    messages = PROMPT.format_messages(
        context=context,
        question=state["question"],
    )

    # 4. Validate input size before calling the LLM.
    try:
        validate_prompt(messages)
    except ValueError as exc:
        return {
            "answer": str(exc),
            "documents": [],
            "status": "invalid",
        }

    # 5. Generate the response.
    response = llm.invoke(messages, config=config)
    answer = response.content

    valid_ids = {str(doc["id"]) for doc in selected_docs}

    def citations_valid(text: str) -> bool:
        cited_ids = re.findall(r"\[(\d+)\]", text)
        return (
            bool(cited_ids)
            and all(cid in valid_ids for cid in cited_ids)
        )

    # 6. Retry once if citations are invalid.
    if not citations_valid(answer):
        retry_messages = [
            *messages,
            AIMessage(content=answer),
            HumanMessage(
                content=(
                    "Rewrite your previous answer using only "
                    "the same supported facts. "
                    "Every factual sentence and numbered "
                    "maintenance step must end with a valid "
                    "source citation such as [1] or [2]. "
                    "Use only citation numbers provided "
                    "in the retrieved context. "
                    "Do not introduce new information."
                )
            ),
        ]

        # The retry also consumes input tokens.
        try:
            validate_prompt(retry_messages)
        except ValueError:
            return {
                "answer": (
                    "I couldn't produce a properly cited "
                    "answer within the configured token "
                    "budget. Please try a shorter question."
                ),
                "documents": selected_docs,
                "status": "invalid",
            }

        retry_response = llm.invoke(
            retry_messages,
            config=config,
        )
        answer = retry_response.content

        if not citations_valid(answer):
            return {
                "answer": (
                    "I couldn't produce a reliably cited "
                    "answer from the retrieved documents."
                ),
                "documents": selected_docs,
                "status": "invalid",
            }

    return {
        "answer": answer,
        "documents": selected_docs,
        "status": "answered",
    }


def insufficient_evidence(state: FactoryState) -> dict:
    return {
        "answer": (
            "I couldn't find a matching maintenance document "
            "for this question. Please verify the machine ID "
            "and error code."
        ),
        "status": "no_evidence",
    }
