
from typing import TypedDict


class SourceDocument(TypedDict):
    id: int
    path: str
    content: str


class FactoryState(TypedDict, total=False):
    question: str
    documents: list[SourceDocument]
    answer: str
    status: str
