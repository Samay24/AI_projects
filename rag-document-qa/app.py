"""Streamlit interface for the RAG-Based Document Q&A Assistant.

No sidebar — the API key is read from .env and never shown on screen.
Workflow: upload documents -> index -> ask.
"""

import os

import httpx
import streamlit as st

import config
import ingest
import rag

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    :root {
        --accent: #8b5cf6;
        --accent-2: #38bdf8;
        --card: #141a2c;
        --card-border: rgba(148,163,184,.14);
        --text: #e2e8f0;
        --muted: #94a3b8;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px; }
    [data-testid="stSidebar"] { display: none; }
    .stApp { background:
        radial-gradient(1200px 500px at 15% -10%, rgba(139,92,246,.14), transparent 60%),
        radial-gradient(1000px 420px at 90% -10%, rgba(56,189,248,.10), transparent 55%),
        #0b0f1a; }

    /* ---- brand header ---- */
    .brand {
        display: flex; align-items: center; gap: 1rem; padding: .9rem 1.2rem;
        border: 1px solid var(--card-border); border-radius: 18px;
        background: linear-gradient(120deg, rgba(139,92,246,.12), rgba(56,189,248,.08));
        margin-bottom: 1.1rem;
    }
    .brand-logo {
        width: 46px; height: 46px; border-radius: 13px; font-size: 1.5rem;
        display: grid; place-items: center; flex: 0 0 auto;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        box-shadow: 0 6px 18px rgba(124,58,237,.4);
    }
    .brand h1 { margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); }
    .brand p  { margin: 0; color: var(--muted); font-size: .85rem; }
    .brand-right { margin-left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: .35rem; }

    /* ---- status pills / badges ---- */
    .pill { display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .75rem;
            border-radius: 999px; font-size: .74rem; font-weight: 600; white-space: nowrap; }
    .pill-ok   { background: rgba(46,204,113,.13); color: #4ade80; border: 1px solid rgba(46,204,113,.38); }
    .pill-err  { background: rgba(239,68,68,.12); color: #f87171; border: 1px solid rgba(239,68,68,.38); }
    .model-badge { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .72rem;
                   color: #c7cbe1; padding: .25rem .7rem; border-radius: 10px;
                   background: rgba(139,92,246,.12); border: 1px solid rgba(139,92,246,.3); }

    /* ---- stat cards ---- */
    .stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin-bottom: 1.2rem; }
    .stat-card { border: 1px solid var(--card-border); border-radius: 16px; padding: .75rem 1rem;
                 background: rgba(255,255,255,.02); }
    .stat-card .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
    .stat-card .v { font-size: 1.5rem; font-weight: 800; color: var(--text); margin-top: .1rem; }
    .stat-card .v small { font-size: .8rem; color: var(--muted); font-weight: 600; }

    /* ---- step headers ---- */
    .step { display: flex; align-items: center; gap: .5rem; color: #a5b4fc; font-weight: 700;
            font-size: .85rem; letter-spacing: .02em; margin-bottom: .6rem; }
    .step-num { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 7px;
                background: linear-gradient(135deg, #7c3aed, #2563eb); color: #fff; font-size: .72rem; }

    /* ---- doc cards ---- */
    .doc-card { display: flex; align-items: center; gap: .6rem; padding: .55rem .8rem;
                border: 1px solid var(--card-border); border-radius: 12px;
                background: rgba(255,255,255,.02); margin-bottom: .45rem; }
    .doc-card .doc-icon { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
                          background: rgba(239,68,68,.14); font-size: .95rem; flex: 0 0 auto; }
    .doc-card .doc-name { color: var(--text); font-size: .84rem; font-weight: 600; word-break: break-all; }

    /* ---- source chunks inside chat ---- */
    .src-card { border: 1px solid var(--card-border); border-radius: 12px; padding: .55rem .8rem;
                margin: .35rem 0; background: rgba(139,92,246,.06); }
    .src-top { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
    .src-num { width: 20px; height: 20px; border-radius: 6px; display: grid; place-items: center;
               font-size: .68rem; font-weight: 800; color: #fff;
               background: linear-gradient(135deg, #7c3aed, #2563eb); }
    .src-file { font-size: .74rem; color: #c4b5fd; font-weight: 600; word-break: break-all; }
    .src-page { font-size: .68rem; color: var(--muted); border: 1px solid var(--card-border);
                border-radius: 6px; padding: .05rem .4rem; }
    .src-excerpt { font-size: .78rem; color: #cbd5e1; margin-top: .4rem; line-height: 1.55; }

    /* ---- chips (pills) for suggestions ---- */
    [data-testid="stPills"] > div > div { gap: .45rem; }
    [data-testid="stPills"] label {
        border: 1px solid var(--card-border); border-radius: 999px;
        background: rgba(255,255,255,.03); color: #cbd5e1; font-size: .8rem;
        padding: .3rem .85rem; transition: all .15s ease;
    }
    [data-testid="stPills"] label:hover { border-color: rgba(139,92,246,.6); color: #fff; }
    [data-testid="stPills"] label[aria-checked="true"] {
        background: rgba(139,92,246,.22) !important; border-color: var(--accent) !important; color: #e9d5ff;
    }

    /* ---- chat ---- */
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,.03); border: 1px solid var(--card-border);
        border-radius: 16px; padding: .6rem .95rem .35rem .95rem; margin-bottom: .65rem;
    }
    [data-testid="stChatInput"] textarea { border-radius: 14px; }

    /* ---- misc ---- */
    .tip { color: var(--muted); font-size: .78rem; line-height: 1.5; margin-top: .5rem; }
    .footer { color: #64748b; font-size: .72rem; text-align: center; margin-top: 1.4rem; }
    .empty-hero { text-align: center; padding: 2rem 1rem 1rem; }
    .empty-hero .icon { font-size: 2.6rem; }
    .empty-hero h3 { color: var(--text); margin: .6rem 0 .25rem; font-weight: 700; }
    .empty-hero p { color: var(--muted); margin: 0 auto; max-width: 460px; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- session state
defaults = {
    "store": None,
    "chunk_size": config.CHUNK_SIZE,
    "chunk_overlap": config.CHUNK_OVERLAP,
    "top_k": config.TOP_K,
    "indexed": 0,
    "indexed_files": [],
    "messages": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

API_KEY = config.MISTRAL_API_KEY

SUGGESTIONS = [
    "Summarize the key points of this document.",
    "What are the main topics covered?",
    "List the projects or skills mentioned.",
    "Which companies or people appear in the document?",
]


# ---------------------------------------------------------------------- actions
def reindex() -> None:
    with st.spinner("Indexing documents into ChromaDB..."):
        paths = ingest.list_pdfs()
        if not paths:
            st.session_state.store = None
            st.session_state.indexed = 0
            st.session_state.indexed_files = []
            return
        store, n = ingest.ingest(
            paths,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
        st.session_state.store = store
        st.session_state.indexed = n
        st.session_state.indexed_files = sorted(os.path.basename(p) for p in paths)


def index_uploads(up_files) -> None:
    if not up_files:
        st.warning("Choose at least one PDF first.")
        return
    for up in up_files:
        ingest.save_upload(up)
    reindex()


def clear_documents() -> None:
    ingest.clear_vectorstore()
    for path in ingest.list_pdfs():
        try:
            os.remove(path)
        except OSError:
            pass
    st.session_state.store = None
    st.session_state.indexed = 0
    st.session_state.indexed_files = []


def test_api() -> None:
    if not API_KEY:
        st.error("No API key found. Add MISTRAL_API_KEY to your .env file first.")
        return
    with st.spinner("Contacting Mistral API..."):
        try:
            reply = rag.test_connection(model=config.MISTRAL_MODEL, api_key=API_KEY)
            st.success(f"Connected — {config.MISTRAL_MODEL} replied: **{reply}**")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                st.warning(
                    "Mistral is rate-limiting this account (429). Wait a bit and retry, "
                    "or check your quota at console.mistral.ai."
                )
            else:
                st.error(f"API test failed (HTTP {exc.response.status_code}).")
        except Exception as exc:
            st.error(f"API test failed: {exc}")


def clear_chat() -> None:
    st.session_state.messages = []


# ---------------------------------------------------------------- auto-attach
# Restore a persisted index ONLY if it is backed by actual PDFs in data/.
if (
    st.session_state.store is None
    and ingest.list_pdfs()
    and os.path.isdir(config.VECTORSTORE_DIR)
):
    try:
        store = ingest.get_vectorstore()
        if store._collection.count() > 0:
            st.session_state.store = store
            st.session_state.indexed = store._collection.count()
            stored = store.get(include=["metadatas"])
            st.session_state.indexed_files = sorted(
                {
                    os.path.basename(m.get("source", ""))
                    for m in stored.get("metadatas", [])
                    if m.get("source")
                }
            )
    except Exception:
        pass

# ----------------------------------------------------------------------- header
api_state = "ok" if API_KEY else "err"
api_label = "API connected" if API_KEY else "API key missing"

st.markdown(
    f"""
    <div class="brand">
        <div class="brand-logo">📄</div>
        <div>
            <h1>RAG Document Q&amp;A Assistant</h1>
            <p>Ask questions about your PDFs — every answer is grounded in your documents with citations.</p>
        </div>
        <div class="brand-right">
            <span class="pill pill-{api_state}">{"●" if API_KEY else "○"} {api_label}</span>
            <span class="model-badge">⚙ {config.MISTRAL_MODEL}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_KEY:
    st.error(
        "**Add your Mistral API key** to `.env` as `MISTRAL_API_KEY=...` and restart the app. "
        "It is loaded from there and never displayed on screen."
    )

# ------------------------------------------------------------------- stat cards
n_pdfs = len(ingest.list_pdfs())
st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-card"><div class="k">Documents</div><div class="v">{n_pdfs}</div></div>
        <div class="stat-card"><div class="k">Chunks indexed</div><div class="v">{st.session_state.indexed}</div></div>
        <div class="stat-card"><div class="k">Retrieval (top-k)</div><div class="v">{st.session_state.top_k}</div></div>
        <div class="stat-card"><div class="k">Chat turns</div><div class="v">{len(st.session_state.messages) // 2}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- toolbar actions
tool = st.columns([1, 1, 1, 3])
if tool[0].button("🔌 Test API", use_container_width=True):
    test_api()
if tool[1].button("💬 New chat", use_container_width=True):
    clear_chat()
    st.rerun()
if tool[2].button("🗑 Clear documents", use_container_width=True):
    clear_documents()
    st.rerun()

# ------------------------------------------------------------------ layout panels
left, right = st.columns([1.35, 2.65], gap="large")

with left:
    # --------------------------------------------------------------- documents
    st.markdown(
        '<div class="step"><span class="step-num">1</span>Upload documents</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        up_files = st.file_uploader(
            "Drop PDFs here", type=["pdf"], accept_multiple_files=True
        )
        if up_files:
            st.caption(f"{len(up_files)} file(s) selected — click below to index.")
        st.button(
            "Index documents",
            type="primary",
            use_container_width=True,
            on_click=index_uploads,
            args=(up_files,),
        )
        st.markdown(
            '<div class="tip">Indexing reads each PDF, splits it into chunks, embeds them '
            "with a local ONNX miniLM model, and stores the vectors in ChromaDB.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------- your index
    st.markdown(
        '<div class="step" style="margin-top:1rem;"><span class="step-num">2</span>Your knowledge base</div>',
        unsafe_allow_html=True,
    )
    pdfs = ingest.list_pdfs()
    if pdfs:
        for p in pdfs:
            st.markdown(
                f'<div class="doc-card"><span class="doc-icon">📕</span>'
                f'<span class="doc-name">{os.path.basename(p)}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(f"{st.session_state.indexed} chunks · {len(pdfs)} PDF(s)")
        if st.button("Remove all", use_container_width=True):
            clear_documents()
            st.rerun()
    else:
        st.info("No PDFs yet — upload one in Step 1.")

    # ------------------------------------------------------- advanced tuning
    with st.expander("Advanced tuning"):
        st.session_state.chunk_size = st.slider(
            "Chunk size (chars)", 200, 2000, st.session_state.chunk_size, 50
        )
        st.session_state.chunk_overlap = st.slider(
            "Overlap (chars)", 0, 500, st.session_state.chunk_overlap, 25
        )
        st.session_state.top_k = st.slider(
            "Top-k retrieval", 1, 10, st.session_state.top_k, 1
        )
        if st.button("Re-index with these settings", use_container_width=True):
            reindex()
            st.rerun()

# ----------------------------------------------------------------------- chat
with right:
    st.markdown(
        '<div class="step"><span class="step-num">3</span>Ask a question</div>',
        unsafe_allow_html=True,
    )

    def render_sources(result) -> None:
        srcs = result.unique_sources
        if not srcs:
            return
        with st.expander(f"Sources · {len(srcs)} chunk(s)"):
            for s in srcs:
                page = f"page {s['page']}" if s["page"] is not None else "page —"
                st.markdown(
                    f'<div class="src-card">'
                    f'<div class="src-top">'
                    f'<span class="src-num">{s["number"]}</span>'
                    f'<span class="src-file">📄 {s["source"]}</span>'
                    f'<span class="src-page">{page}</span>'
                    f'</div>'
                    f'<div class="src-excerpt">{s["excerpt"]}{"…" if len(s["excerpt"]) == 400 else ""}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------- history
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"])
    else:
        st.markdown(
            '<div class="empty-hero"><div class="icon">🔍</div>'
            "<h3>Ask about your documents</h3>"
            "<p>Upload a PDF on the left, then ask anything — the assistant retrieves the "
            "most relevant sections and answers with numbered citations.</p></div>",
            unsafe_allow_html=True,
        )
        pick = st.pills("Try asking", SUGGESTIONS, key="sugg")
        if pick:
            st.session_state.prompt0 = pick

    # ------------------------------------------------------- question handler
    def run_question(question: str) -> None:
        question = (question or "").strip()
        if not question:
            return

        pdfs = ingest.list_pdfs()
        if not pdfs:
            st.chat_message("user", avatar="🧑").markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("Upload at least one PDF in **Step 1** before asking.")
            return
        if not API_KEY:
            st.chat_message("user", avatar="🧑").markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("Add MISTRAL_API_KEY to your `.env`, then restart the app.")
            return

        expected = sorted(os.path.basename(p) for p in pdfs)
        if expected != st.session_state.indexed_files:
            reindex()

        if not st.session_state.store:
            st.chat_message("user", avatar="🧑").markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("Index at least one PDF (Step 1) before asking.")
            return

        st.chat_message("user", avatar="🧑").markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            try:
                k = min(st.session_state.top_k, max(1, st.session_state.indexed))
                docs = rag.retrieve(st.session_state.store, question, k)
                if not docs:
                    st.markdown("No matching content found in your documents.")
                    st.session_state.messages.append({"role": "user", "content": question})
                    return
                with st.status("Retrieving context…", expanded=False) as status:
                    st.write(f"Matched **{len(docs)}** chunks (top-k = {k}).")
                    status.update(label="Context retrieved", state="complete")
                full = st.write_stream(
                    rag.generate_stream(
                        question,
                        docs,
                        model=config.MISTRAL_MODEL,
                        api_key=API_KEY,
                    )
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    st.warning(
                        "Mistral is rate-limiting this account (429). Wait a minute and retry, "
                        "or check your quota at console.mistral.ai."
                    )
                else:
                    st.error(f"HTTP error {exc.response.status_code} from Mistral API.")
                return
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
                return

        from rag import RAGResult

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append(
            {"role": "assistant", "content": full, "sources": RAGResult(full, docs)}
        )

    def on_chat() -> None:
        run_question(st.session_state.prompt)

    st.chat_input("Ask anything about your documents…", key="prompt", on_submit=on_chat)

    if st.session_state.get("prompt0"):
        pending = st.session_state.pop("prompt0")
        run_question(pending)