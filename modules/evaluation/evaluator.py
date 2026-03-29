"""
evaluator.py — RAGAS Evaluation Pipeline
Responsibility: Measure RAG system quality against a golden dataset.

Guard: exits cleanly if dataset has < 3 populated entries.
Run as: python -m modules.evaluation.evaluator

Metrics (targets):
  faithfulness      > 0.85  (hallucination rate)
  answer_relevancy  > 0.80  (answer quality)
  context_precision > 0.70  (retrieval efficiency)
  context_recall    > 0.75  (retrieval completeness)
"""
from __future__ import annotations

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("RAG-V3-EVAL")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.75,
}


def _load_dataset() -> List[Dict[str, Any]]:
    if not GOLDEN_DATASET_PATH.exists():
        print(f"⚠  golden_dataset.json not found at: {GOLDEN_DATASET_PATH}")
        sys.exit(1)

    raw = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    dataset = raw.get("dataset", [])

    # Filter out empty/placeholder entries
    populated = [
        entry for entry in dataset
        if entry.get("question") and entry.get("ground_truth_answer")
    ]
    return populated


def run_eval() -> None:
    # ── Guard: minimum dataset size ───────────────────────────────────────────
    dataset = _load_dataset()
    if len(dataset) < 3:
        print(
            f"\n⚠  Golden dataset has {len(dataset)} populated entry/entries — "
            f"minimum 3 required.\n"
            f"   Populate modules/evaluation/golden_dataset.json with real Q&A pairs\n"
            f"   from your actual documents before running evaluation.\n"
            f"   Target: 20+ entries for a production baseline.\n"
        )
        sys.exit(0)

    print(f"\n── RAG V3 Evaluation ({'─' * 38})")
    print(f"   Dataset  : {len(dataset)} entries")
    print(f"   Targets  : {TARGETS}\n")

    # ── Run each question through the pipeline ────────────────────────────────
    try:
        import pipeline
    except ImportError:
        print("⚠  Could not import pipeline.py — run from the rag-v3/ root directory")
        sys.exit(1)

    ragas_data: List[Dict] = []

    for i, entry in enumerate(dataset, 1):
        q = entry["question"]
        gt = entry["ground_truth_answer"]
        print(f"  [{i}/{len(dataset)}] Running: {q[:70]}...")

        try:
            result = pipeline.run(question=q)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            contexts = [s.get("excerpt", "") for s in sources if s.get("excerpt")]

            ragas_data.append({
                "question": q,
                "answer": answer,
                "contexts": contexts if contexts else [""],
                "ground_truth": gt,
            })
            print(f"       → answer: {answer[:80]}...")
        except Exception as e:
            logger.error(f"Pipeline failed for question {i}: {e}")
            ragas_data.append({
                "question": q,
                "answer": "",
                "contexts": [""],
                "ground_truth": gt,
            })

    # ── Run RAGAS ─────────────────────────────────────────────────────────────
    print("\n── Computing RAGAS Metrics... ──────────────────────────────────────")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        hf_dataset = Dataset.from_list(ragas_data)
        results = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        print("\n── Results ─────────────────────────────────────────────────────────")
        print(f"{'Metric':<25} {'Score':>8} {'Target':>8} {'Status':>8}")
        print("─" * 55)
        for metric, target in TARGETS.items():
            score = results.get(metric, 0.0)
            status = "✓ PASS" if score >= target else "✗ FAIL"
            print(f"{metric:<25} {score:>8.3f} {target:>8.2f} {status:>8}")
        print("─" * 55)

        # Per-question breakdown
        if hasattr(results, "to_pandas"):
            df = results.to_pandas()
            print(f"\n── Per-Question Detail ─────────────────────────────────────────────")
            print(df[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))

    except ImportError as e:
        print(f"⚠  RAGAS not installed: {e}")
        print("   Run: pip install ragas datasets")
        print(f"\n   Raw answers collected ({len(ragas_data)} entries):")
        for item in ragas_data:
            print(f"   Q: {item['question'][:60]}")
            print(f"   A: {item['answer'][:80]}\n")


if __name__ == "__main__":
    run_eval()
