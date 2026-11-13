
from fastapi import FastAPI
from pydantic import BaseModel, Field

from factoryassist.orchestration.graph import run_workflow


app = FastAPI(
    title="FactoryAssist AI",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    id: int
    path: str
    content: str


class AskResponse(BaseModel):
    answer: str
    status: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return run_workflow(request.question)
