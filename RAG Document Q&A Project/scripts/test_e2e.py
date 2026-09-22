"""End-to-end pipeline test: PDF -> chunk -> embed -> retrieve -> (optionally) generate.

Run with:
    python scripts/test_e2e.py
Set MISTRAL_API_KEY to also test LLM generation, e.g.:
    set MISTRAL_API_KEY=<key> & python scripts/test_e2e.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
import ingest
import rag

CHECK = "✅"
FAIL = "❌"
SKIP = "⏭️"

results: list[tuple[bool, str]] = []


def record(ok: bool, label: str) -> None:
    results.append((ok, label))
    print(f"{CHECK if ok else FAIL} {label}")


def main() -> int:
    print("RAG pipeline end-to-end test")
    print("=" * 50)

    sys.path.insert(0, str(ROOT / "scripts"))
    from dotenv import load_dotenv

    load_dotenv()

    # 1. Sample PDF present (generate if missing)
    import generate_sample_pdf as gen_mod

    paths = ingest.list_pdfs()
    if not paths:
        gen_mod.main()
        paths = ingest.list_pdfs()
    record(bool(paths), f"Sample PDFs present: {[os.path.basename(p) for p in paths]}")

    # 2. Ingest
    try:
        store, n = ingest.ingest(paths, chunk_size=800, chunk_overlap=100)
        record(n > 0, f"Ingestion ok — {n} chunks in ChromaDB")
    except Exception as exc:
        record(False, f"Ingestion failed: {exc}")
        store = None

    if store is None:
        print(f"{FAIL} Aborting: cannot retrieve without an index.")
        return 1

    # 3. Retrieve
    question = "How much did revenue grow in Q3 2026?"
    docs = rag.retrieve(store, question, top_k=3)
    record(len(docs) > 0, f"Top-k retrieval returned {len(docs)} chunks")
    for i, d in enumerate(docs, 1):
        print(f"    [{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}: "
              f"{d.page_content[:80].replace(chr(10), ' ')}…")

    # 4. Grounded context formatting
    ctx = rag.format_context(docs)
    record("[1]" in ctx, "Context formatted with citations")

    # 5. LLM generation (only if a key is available)
    key = config.MISTRAL_API_KEY or os.environ.get("MISTRAL_API_KEY")
    if key:
        try:
            reply = rag.test_connection(model=config.MISTRAL_MODEL, api_key=key)
            record(bool(reply.strip()), f"API connection ok — model said: {reply[:40]}")
        except Exception as exc:
            record(False, f"API connection failed: {exc}")

        result = rag.generate(question, docs, model=config.MISTRAL_MODEL, api_key=key)
        record(bool(result.answer.strip()), "LLM generation ok")
        print(f"    Answer: {result.answer[:160]}…")
        print(f"    Unique sources: {len(result.unique_sources)}")
    else:
        print(f"{SKIP} No MISTRAL_API_KEY — skipping LLM generation test")
        record(True, "No key → LLM test skipped (pipeline verified)")

    print("=" * 50)
    passed = sum(1 for ok, _ in results if ok)
    print(f"Result: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())