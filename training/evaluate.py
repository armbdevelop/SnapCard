"""Evaluate BLIP captioning quality on the test set.

Computes BLEU-4, ROUGE-L, average caption length, and inference time.
Supports comparing base BLIP vs LoRA-adapted model.

Usage:
    # Base model
    python training/evaluate.py --test training/data/test.jsonl --captions training/data/captions.jsonl --limit 50

    # With LoRA adapter
    python training/evaluate.py --test training/data/test.jsonl --captions training/data/captions.jsonl \
        --lora-path backend/model_cache/snapcard_lora --limit 50
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLIP captioning on test set")
    parser.add_argument("--test", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--captions", type=str, required=True, help="Path to captions.jsonl with reference caption_ru")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--output", type=str, default="training/data/eval_results.json", help="Path to save results")
    args = parser.parse_args()

    # Load test data
    test_path = Path(args.test)
    captions_path = Path(args.captions)

    if not test_path.exists():
        logger.info("Test file not found: %s — no test data, skipping", test_path)
        sys.exit(0)

    if not captions_path.exists():
        logger.info("Captions file not found: %s — no test data, skipping", captions_path)
        sys.exit(0)

    test_records = load_jsonl(str(test_path))
    caption_records = load_jsonl(str(captions_path))

    if not test_records:
        logger.info("No test data, skipping")
        sys.exit(0)

    # Build caption lookup
    caption_map = {c["id"]: c["caption_ru"] for c in caption_records if "caption_ru" in c}

    # Merge test records with reference captions
    test_data = []
    # Resolve image paths relative to test file's parent directory
    data_dir = test_path.parent
    for r in test_records:
        ref = caption_map.get(r["id"])
        if ref:
            img_path = data_dir / r["image_path"]
            if img_path.exists():
                test_data.append({"image_path": str(img_path), "reference": ref})

    if not test_data:
        logger.info("No matching test data with captions and images, skipping")
        sys.exit(0)

    if args.limit:
        test_data = test_data[: args.limit]

    logger.info("Evaluating on %d samples", len(test_data))

    # Import ML components — add backend to path
    project_root = Path(__file__).resolve().parent.parent
    backend_path = project_root / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from app.ml.image_captioner import ImageCaptioner

    # Load captioner
    model_type = "lora" if args.lora_path else "base"
    logger.info("Loading BLIP model (type: %s)...", model_type)
    captioner = ImageCaptioner(lora_path=args.lora_path)
    logger.info("Model loaded")

    # Run inference and collect predictions
    predictions = []
    references = []
    total_inference_ms = 0.0

    for i, sample in enumerate(test_data):
        t0 = time.perf_counter()
        caption, _ = captioner.caption(sample["image_path"])
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        total_inference_ms += elapsed_ms

        predictions.append(caption)
        references.append(sample["reference"])

        if (i + 1) % 10 == 0:
            logger.info("  processed %d/%d", i + 1, len(test_data))

    # Compute metrics
    import sacrebleu
    from rouge_score import rouge_scorer

    # BLEU-4
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    bleu4 = bleu.score / 100.0  # normalize to 0-1

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    # Average length in words
    avg_length = sum(len(p.split()) for p in predictions) / len(predictions)

    # Average inference time
    avg_inference_ms = total_inference_ms / len(test_data)

    # Print results table
    print()
    print(f"{'Metric':<20}| {'Value':>10}")
    print(f"{'-' * 20}|{'-' * 11}")
    print(f"{'BLEU-4':<20}| {bleu4:>10.3f}")
    print(f"{'ROUGE-L':<20}| {rouge_l:>10.3f}")
    print(f"{'Avg length':<20}| {avg_length:>7.1f} words")
    print(f"{'Avg inference':<20}| {avg_inference_ms:>7.0f} ms")
    print()

    # Save results
    result_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "lora_path": args.lora_path,
        "num_samples": len(test_data),
        "bleu4": round(bleu4, 4),
        "rougeL": round(rouge_l, 4),
        "avg_length_words": round(avg_length, 1),
        "avg_inference_ms": round(avg_inference_ms, 1),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results or start fresh
    existing = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(result_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
