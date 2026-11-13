import os
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_QUESTION_TOKENS = int(
    os.getenv("MAX_QUESTION_TOKENS", "256")
)

MAX_CONTEXT_TOKENS = int(
    os.getenv("MAX_CONTEXT_TOKENS", "2000")
)

MAX_INPUT_TOKENS = int(
    os.getenv("MAX_INPUT_TOKENS", "3000")
)

# Reserve space for chat formatting and tokenizer differences.
INPUT_SAFETY_MARGIN = 200

def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a string."""
    return len(ENCODING.encode(text))


def validate_question(question: str) -> None:
    """Reject excessively long user questions."""
    used = count_tokens(question)

    if used > MAX_QUESTION_TOKENS:
        raise ValueError(
            f"Question too long: {used} tokens. "
            f"Maximum: {MAX_QUESTION_TOKENS}."
        )


def build_budgeted_context(
    documents: list[dict],
) -> tuple[str, list[dict]]:
    """Keep ranked evidence that fits the context budget."""

    passages = []
    selected_documents = []

    for doc in documents:
        passage = (
            f"[{doc['id']}] Source: {doc['path']}\n"
            f"{doc['content']}"
        )

        candidate = "\n\n".join([*passages, passage])

        if count_tokens(candidate) > MAX_CONTEXT_TOKENS:
            continue

        passages.append(passage)
        selected_documents.append(doc)

    return "\n\n".join(passages), selected_documents


def validate_prompt(messages) -> None:
    """Check the estimated size of the complete chat input."""

    text = "\n".join(
        str(message.content)
        for message in messages
    )

    used = count_tokens(text)
    allowed = MAX_INPUT_TOKENS - INPUT_SAFETY_MARGIN

    if used > allowed:
        raise ValueError(
            f"Prompt too long: {used} estimated tokens. "
            f"Allowed content budget: {allowed}."
        )
