import os
import gradio as gr
import httpx


API_URL = os.getenv(
    "FACTORYASSIST_API_URL",
    "http://127.0.0.1:8000/ask",
)


def chat(message: str, history: list[dict]) -> str:
    try:
        response = httpx.post(
            API_URL,
            json={"question": message},
            timeout=120.0,
        )
        response.raise_for_status()

        result = response.json()

        answer = result["answer"]
        sources = result.get("sources", [])

        if sources:
            answer += "\n\n### Retrieved sources\n"
            for source in sources:
                answer += (
                    f"\n[{source['id']}] "
                    f"`{source['path']}`"
                )

        return answer

    except httpx.TimeoutException:
        return "The request timed out. Please try again."

    except httpx.HTTPStatusError as exc:
        print(f"FactoryAssist API HTTP error: {exc.response.status_code}")
        return (
            "The maintenance service encountered an error. "
            "Please try again."
        )

    except httpx.RequestError as exc:
        print(f"FactoryAssist API connection error: {exc}")
        return "Unable to connect to the maintenance service."


demo = gr.ChatInterface(
    fn=chat,
    title="FactoryAssist AI",
    description=(
        "An AI-powered manufacturing maintenance assistant using "
        "LangGraph, LangChain, FAISS, Qwen and Langfuse."
    ),
    examples=[
        "What causes error E-2047 in machine M-204?",
        "What happened in the previous M-204 incident?",
        "What does error E-9999 mean for machine M-999?",
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=7860,
    )
