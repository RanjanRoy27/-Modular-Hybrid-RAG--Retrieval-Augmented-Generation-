#!/usr/bin/env python
"""
ingest_runner.py — CLI Entry Point for Offline Ingestion
Usage: python ingest_runner.py [--data-dir path/to/data]
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from modules.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG V3 — Document Ingestion Runner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory (default: ./data)",
    )
    args = parser.parse_args()

    print("=" * 56)
    print("  RAG V3 — Ingestion Pipeline")
    print("=" * 56)

    result = run_ingestion(data_dir=args.data_dir)

    print("\n── Ingestion Report ────────────────────────────────")
    print(f"  Files processed : {result['files_processed']}")
    print(f"  Chunks indexed  : {result['chunks_indexed']}")
    print(f"  Chunks skipped  : {result['chunks_skipped']} (duplicates)")
    print(f"  Duration        : {result['duration_ms']} ms")

    if result.get("error"):
        print(f"  ERROR           : {result['error']}")
        sys.exit(1)

    print("  Status          : ✓ Complete")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
