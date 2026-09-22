"""Central configuration and tunable RAG parameters."""

import os

from dotenv import load_dotenv


def _streamlit_secrets() -> dict:
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


load_dotenv()

# Deployed on Streamlit Cloud there is no .env — keys go in dashboard Secrets.
_SECRETS = _streamlit_secrets() if not os.getenv("MISTRAL_API_KEY") else {}

# --- Mistral API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "") or _SECRETS.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "") or _SECRETS.get("MISTRAL_MODEL", "open-mistral-nemo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# --- Chunking (tuned) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval (tuned) ---
TOP_K = int(os.getenv("TOP_K", "6"))

# --- Rate-limit handling (free-tier Mistral) ---
RATE_LIMIT_RETRIES = int(os.getenv("RATE_LIMIT_RETRIES", "4"))
RATE_LIMIT_WAIT_SECONDS = int(os.getenv("RATE_LIMIT_WAIT_SECONDS", "15"))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))
COLLECTION_NAME = "documents"

ALLOWED_EXTENSIONS = {".pdf"}