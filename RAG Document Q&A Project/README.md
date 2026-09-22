# RAG-Based Document Q&A Assistant

Retrieval-augmented generation pipeline that ingests PDF documents, chunks and embeds the text into a ChromaDB vector store, and answers questions with source-grounded responses via the Mistral API — exposed through a Streamlit interface.

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

## Setup

```bash
cd rag-document-qa
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then paste your Mistral API key
```

## Check the API key works

Before running the UI, verify the key:

```bash
python check_api.py
# or with a specific model:
python check_api.py --model mistral-large-latest
```

You can also use the **Test API connection** button in the app toolbar.

## Run

```bash
streamlit run app.py
```

Upload one or more PDFs in the left panel → **Index documents** → ask questions in the chat.

## Tuning

| Parameter        | Default | Where                          |
| ---------------- | ------- | ------------------------------ |
| `CHUNK_SIZE`     | 1000    | `.env` or Advanced tuning slider |
| `CHUNK_OVERLAP`  | 200     | `.env` or Advanced tuning slider |
| `TOP_K`          | 6       | `.env` or Advanced tuning slider |
| `MISTRAL_MODEL`  | `open-mistral-nemo` | `.env` / Cloud Secrets (works on this plan — `mistral-small-latest` returns 429) |
| `TEMPERATURE`    | 0.0     | `.env`                         |

Changing chunk size/overlap requires a re-index (use the **Re-index with new chunk settings** button). `top_k` and model changes apply immediately.

## Troubleshooting

- **`429 Rate limit exceeded`** — the key is valid, but your Mistral account quota
  is exhausted (common on the free tier). Wait a minute and retry; the app
  auto-retries and shows a friendly message. Check usage at
  https://console.mistral.ai.
- **`MISTRAL_API_KEY is not set`** — add the key to `.env` locally, or to the
  Streamlit Cloud Secrets when deployed.
- **"No documents have been indexed yet"** — upload a PDF and click **Index documents**.
- Changing chunk settings needs **Re-index with these settings** (re-embeds everything).

## Deploy (Streamlit Community Cloud)

1. Create a repository on GitHub and push this project (`.env`, `data/` and
   `vectorstore/` are gitignored — the secret never leaves your machine).
2. Sign in at https://share.streamlit.io → **New app** → select the repo,
   branch `main`, file `app.py`.
3. Deploy, then open **Settings → Secrets** and add:
   ```toml
   MISTRAL_API_KEY = "your-key"
   MISTRAL_MODEL = "open-mistral-nemo"
   ```
4. The app reads these automatically (`config.py` falls back to `st.secrets`).
   Upload PDFs inside the deployed app — the index is rebuilt on the fly.

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