# FactoryAssist AI — Synthetic Evaluation Add-on

Adds **21 new synthetic documents** and **40 labelled questions** to your existing
`factoryassist-ai` repository, which already has 3 documents:
- `data/manuals/M204_cooling_manual.txt`
- `data/manuals/C105_conveyor_manual.txt`
- `data/incidents/INC-2026-014.txt`

With those existing three, you should have **24 TXT documents** in `data/`.
The evaluation includes **36 answerable questions** and **4 no-evidence questions**.

The add-on intentionally does **not** replace any of your existing application modules,
manuals, incident file, `pyproject.toml`, or Gradio UI.

## 1. Extract the archive into the repository root

Unzip the archive *inside* `factoryassist-ai/`, preserving paths. The new code
lives in `src/factoryassist/evaluation/`; content lives in `data/`.

## 2. Fix identifier recognition in `src/factoryassist/orchestration/nodes.py`

Your current pattern only recognises M-, C- and E- IDs. Replace:

```python
r"\b(?:M|C|E)-\d{3,5}\b"
```

with:

```python
r"\b[A-Z]-\d{3,5}\b"
```

This matches the new equipment ID prefixes (P-, R-, O-, A-, S-, F-, D-).
It is a temporary rule. Production-quality routing should use structured
metadata filters rather than searching raw text for identifiers.

## 3. Avoid reloading embeddings per question

In `src/factoryassist/rag/retriever.py` add:

```python
from functools import lru_cache

@lru_cache(maxsize=4)
def get_retriever(k: int = 3):
    # Your existing function body remains unchanged
    ...
```

Do not duplicate the function; just decorate the one you already have.
This reuses the same FAISS index in a process. Restart after rebuilding.

## 4. Rebuild, evaluate, and test

```bash
uv run python -m factoryassist.rag.ingestion
uv run python -m factoryassist.evaluation.retrieval_eval
uv run pytest -q tests/test_retrieval_metrics.py
```

The evaluation invokes your *existing* `retrieve_documents` LangGraph node.
It reports document-level Recall@1, @3, @4; MRR@1, @3, @4; and a false-positive
rate over no-evidence questions. It **does not** evaluate answer generation yet.

### Important known limitations

- The retriever is configured with `k=4` **chunks**, not 4 unique documents;
  repeated chunks from one manual can reduce document-level recall.
- One negative question intentionally contains no machine/error ID.
  The current graph may wrongly return unrelated documents. This is a useful
  failure case, not a bug in the dataset.
- Some gold questions have **two relevant sources**. A query can have a good
  first relevant hit (high MRR) but still miss one supporting document
  (lower recall).
- The existing three source files must remain at their original paths, with
  the content created earlier in the project.
- Synthetic examples are NOT actual maintenance instructions.
