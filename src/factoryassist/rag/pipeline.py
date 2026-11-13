
from langchain_ollama import ChatOllama
from langfuse import observe
from langchain_core.prompts import ChatPromptTemplate

from factoryassist.rag.retriever import get_retriever

from factoryassist.observability.langfuse_client import (
    get_langfuse_handler,
    flush_langfuse,
)

SYSTEM_PROMPT = """
You are FactoryAssist, a read-only maintenance assistant
for a fictional manufacturing environment.

Answer the question using ONLY the retrieved context.

STRICT RULES:
1. State only facts supported by the context.
2. Distinguish:
   - what an error code means,
   - possible causes listed in a manual,
   - confirmed causes from historical incident reports.
3. Do not add explanations, causes, or maintenance actions
   that are not explicitly documented.
4. Every factual sentence MUST end with a source citation
   in square brackets, such as [1] or [2].
5. Use ONLY the citation numbers attached to the documents.
6. If two documents support a statement, cite both:
   [1][2]
7. If evidence is insufficient, say that you cannot
   determine the answer from the available documents.
8. Never invent a source or a maintenance procedure.
9. Only authorised personnel may perform maintenance
   following approved safety procedures.

EXAMPLE OF THE REQUIRED ANSWER STYLE:
"The error indicates an abnormal equipment temperature [1].
The manual lists reduced airflow as a possible cause [1].
A previous incident confirmed a blocked filter [2]."

The example demonstrates formatting only.
Do not copy its claims unless supported by the context.

RETRIEVED DOCUMENTS:
{context}
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

@observe(name="factoryassist-rag")
def answer_question(question: str) -> dict:
    # Start tracing LangChain components
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]}

    # Step 1: Retrieve relevant documents
    retriever = get_retriever(k=2)
    documents = retriever.invoke(
        question,
        config=config,
    )

    # Step 2: Format retrieved evidence
    context = "\n\n".join(
        f"[{i}] Source: {doc.metadata.get('source')}\n"
        f"{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )

    # Step 3: Initialise local Qwen
    llm = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0,
    )

    # Step 4: Build the prompt
    messages = PROMPT.format_messages(
        context=context,
        question=question,
    )

    # Step 5: Generate and trace the response
    response = llm.invoke(
        messages,
        config=config,
    )

    # Step 6: Return the answer and source metadata
    return {
        "answer": response.content,
        "sources": [
            {
                "id": i,
                "path": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
            }
            for i, doc in enumerate(documents, start=1)
        ],
    }



if __name__ == "__main__":
    question = (
        "What causes error E-2047 in machine M-204, "
        "and how should maintenance personnel respond?"
    )

    result = answer_question(question)

    print("\nQUESTION:", question)
    print("\nFACTORYASSIST:\n", result["answer"])

    print("\nRETRIEVED SOURCES:")
    for source in result["sources"]:
        print(f"[{source['id']}] {source['path']}")
    
    flush_langfuse()
