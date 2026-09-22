"""Retrieval-augmented generation: top-k retrieval + source-grounded Mistral answers."""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

import config

SYSTEM_PROMPT = """\
You are a document Q&A assistant. Answer questions STRICTLY using only the \
CONTEXT provided below. Follow these rules:

1. Use only facts found in the CONTEXT. Do not use prior knowledge.
2. Cite sources inline using bracketed markers like [1], [2] matching the \
context chunk numbers.
3. If the CONTEXT does not contain the answer, reply exactly: \
"I don't have enough information in the provided documents to answer that."
4. Be concise and direct. Quote key phrases when helpful.
"""

USER_PROMPT = """\
CONTEXT:
{context}

QUESTION: {question}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]

    @property
    def unique_sources(self) -> list[dict]:
        """One entry per distinct chunk (source + page + chunk_id)."""
        seen: dict[tuple, dict] = {}
        for i, doc in enumerate(self.sources, start=1):
            meta = doc.metadata
            key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
            if key not in seen:
                seen[key] = {
                    "number": i,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "excerpt": doc.page_content[:400],
                }
        return list(seen.values())


def get_llm(
    model: str = config.MISTRAL_MODEL,
    api_key: str | None = None,
) -> ChatMistralAI:
    key = api_key or config.MISTRAL_API_KEY
    if not key:
        raise ValueError("MISTRAL_API_KEY is not set. Add it to .env or the sidebar.")
    return ChatMistralAI(
        model=model,
        mistral_api_key=key,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        timeout=120,
        max_retries=5,
    )


def _invoke_with_retry(llm, messages):
    """Call the model, waiting and retrying on mistral free-tier rate limits."""
    for attempt in range(config.RATE_LIMIT_RETRIES):
        try:
            return llm.invoke(messages)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < config.RATE_LIMIT_RETRIES - 1:
                time.sleep(config.RATE_LIMIT_WAIT_SECONDS * (attempt + 1))
                continue
            raise
    raise RuntimeError("Mistral API rate limit reached.")


def test_connection(
    model: str = config.MISTRAL_MODEL,
    api_key: str | None = None,
) -> str:
    """Verify the Mistral API key works with a trivial completion."""
    llm = get_llm(model=model, api_key=api_key)
    reply = _invoke_with_retry(llm, "Reply with the single word: ok")
    return reply.content.strip()


def retrieve(
    store: Chroma,
    question: str,
    top_k: int = config.TOP_K,
) -> list[Document]:
    """Similarity search returning the top-k most relevant chunks."""
    return store.similarity_search(question, k=top_k)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        locator = f"{source}, p. {page}" if page is not None else source
        blocks.append(f"[{i}] ({locator})\n{doc.page_content}")
    return "\n\n".join(blocks)


def generate(
    question: str,
    docs: list[Document],
    model: str = config.MISTRAL_MODEL,
    api_key: str | None = None,
) -> RAGResult:
    """Generate a grounded answer from retrieved chunks (non-streaming)."""
    if not docs:
        return RAGResult(
            answer="No documents have been indexed yet. Upload a PDF first.",
            sources=[],
        )
    llm = get_llm(model=model, api_key=api_key)
    response = _invoke_with_retry(
        llm, PROMPT.format_messages(context=format_context(docs), question=question)
    )
    return RAGResult(answer=response.content, sources=docs)


def generate_stream(
    question: str,
    docs: list[Document],
    model: str = config.MISTRAL_MODEL,
    api_key: str | None = None,
) -> Iterator[str]:
    """Stream token-by-token answers grounded in retrieved chunks."""
    if not docs:
        yield "No documents have been indexed yet. Upload a PDF first."
        return
    llm = get_llm(model=model, api_key=api_key)
    for attempt in range(config.RATE_LIMIT_RETRIES):
        try:
            for chunk in llm.stream(
                PROMPT.format_messages(context=format_context(docs), question=question)
            ):
                if chunk.content:
                    yield chunk.content
            return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < config.RATE_LIMIT_RETRIES - 1:
                time.sleep(config.RATE_LIMIT_WAIT_SECONDS * (attempt + 1))
                continue
            raise
    raise RuntimeError("Mistral API rate limit reached.")


def answer(
    question: str,
    store: Chroma,
    top_k: int = config.TOP_K,
    model: str = config.MISTRAL_MODEL,
    api_key: str | None = None,
) -> RAGResult:
    """One-shot end-to-end call: retrieve then generate."""
    docs = retrieve(store, question, top_k)
    return generate(question, docs, model, api_key)