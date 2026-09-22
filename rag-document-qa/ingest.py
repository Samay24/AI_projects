"""PDF ingestion pipeline: load -> chunk -> embed -> persist in ChromaDB."""

import os
import shutil
import uuid
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class _ChromaLocalEmbeddings:
    """Adapts Chroma's built-in ONNX MiniLM embedding function to the
    LangChain interface (embed_documents / embed_query)."""

    def __init__(self) -> None:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        except Exception:
            from chromadb.lib.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _default_embeddings():
    """Lightweight ONNX MiniLM embeddings bundled with Chroma (no torch needed)."""
    return _ChromaLocalEmbeddings()


def save_upload(uploaded_file, data_dir: str = config.DATA_DIR) -> str:
    """Persist an uploaded PDF into the data directory; return its path."""
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, os.path.basename(uploaded_file.name))
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def list_pdfs(data_dir: str = config.DATA_DIR) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if os.path.splitext(name)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def load_pdfs(paths: Iterable[str]) -> list[Document]:
    docs: list[Document] = []
    for path in paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = os.path.basename(path)
        docs.extend(pages)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def get_vectorstore(persist_dir: str = config.VECTORSTORE_DIR) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_default_embeddings(),
        persist_directory=persist_dir,
    )


def clear_vectorstore(persist_dir: str = config.VECTORSTORE_DIR) -> None:
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def ingest(
    paths: Iterable[str] | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    persist_dir: str = config.VECTORSTORE_DIR,
) -> tuple[Chroma, int]:
    """Full pipeline: load PDFs, chunk, embed, and return a populated vector store."""
    paths = list(paths if paths is not None else list_pdfs())
    if not paths:
        raise ValueError("No PDF files found to ingest.")

    docs = load_pdfs(paths)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("Documents produced no text chunks (scanned PDF?).")

    store = get_vectorstore(persist_dir)
    store.delete_collection()
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_default_embeddings(),
        persist_directory=persist_dir,
    )
    store.add_documents(chunks, ids=[str(uuid.uuid4()) for _ in chunks])
    return store, len(chunks)
