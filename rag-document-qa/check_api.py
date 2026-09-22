"""Standalone Mistral API connection check.

Usage:
    python check_api.py
    python check_api.py --model mistral-large-latest
    set MISTRAL_API_KEY=... then: python check_api.py
"""

import argparse

import config
import rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Mistral API connectivity.")
    parser.add_argument(
        "--model", default=config.MISTRAL_MODEL, help="Mistral model to test against"
    )
    args = parser.parse_args()

    if not config.MISTRAL_API_KEY:
        print("No MISTRAL_API_KEY found in .env or environment.")
        return

    print(f"Testing model '{args.model}'...")
    try:
        reply = rag.test_connection(model=args.model, api_key=config.MISTRAL_API_KEY)
        print(f"SUCCESS: model replied -> {reply}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()