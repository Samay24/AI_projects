# RAG-Based Document Q&A Assistant

Retrieval-augmented generation pipeline that ingests PDF documents, chunks and embeds the text into a ChromaDB vector store, and answers questions with source-grounded responses via the Mistral API — exposed through a Streamlit interface.

## 🌐 Live Demo: https://aiprojects-m2drvrjerunbnytejdmr7j.streamlit.app/

## Stack

- **Python**, **LangChain** (loaders, splitters, Chroma integration, Mistral LLM)
- **ChromaDB** vector store with local ONNX MiniLM embeddings
- **Mistral API** (`langchain-mistralai`) for grounded answer generation
- **Streamlit** UI

## Pipeline

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → Embeddings → ChromaDB
                                                                   ↓
                    Streamlit UI ← Mistral ← grounded prompt ← top-k retrieval
```

1. **Ingest** — PDFs are loaded page-by-page with source/page metadata.
2. **Chunk** — `RecursiveCharacterTextSplitter` (default `chunk_size=1000`, `overlap=200`, tuned for coherent retrieval windows).
3. **Embed & store** — chunks embedded locally (Chroma's default all-MiniLM-L6-v2 ONNX model) and persisted to `vectorstore/`.
4. **Retrieve** — top-k similarity search (default `k=6`, tunable in **Advanced tuning**).
5. **Generate** — a strict grounding prompt forces Mistral to answer *only* from retrieved context with `[n]` citations and an explicit "I don't have enough information" fallback.
6. **Cite** — the UI shows the answer plus expandable source chunks with file name and page number.


## Tuning

| Parameter        | Default | Where                          |
| ---------------- | ------- | ------------------------------ |
| `CHUNK_SIZE`     | 1000    | `.env` or Advanced tuning slider |
| `CHUNK_OVERLAP`  | 200     | `.env` or Advanced tuning slider |
| `TOP_K`          | 6       | `.env` or Advanced tuning slider |
| `MISTRAL_MODEL`  | `open-mistral-nemo` | `.env` / Cloud Secrets (works on this plan — `mistral-small-latest` returns 429) |
| `TEMPERATURE`    | 0.0     | `.env`                         |

Changing chunk size/overlap requires a re-index (use the **Re-index with new chunk settings** button). `top_k` and model changes apply immediately.


## Project layout

```
rag-document-qa/
├── app.py            # Streamlit UI
├── ingest.py         # PDF → chunks → ChromaDB
├── rag.py            # retrieval + Mistral generation with citations
├── check_api.py      # standalone API key test
├── config.py         # tunable parameters
├── requirements.txt
├── .env.example
├── data/             # uploaded PDFs
└── vectorstore/      # persisted Chroma index
```
