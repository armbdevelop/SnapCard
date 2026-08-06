import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "google/mt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model(adapter_path: Path) -> PeftModel:
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(DEVICE)
    model.eval()
    return model


def generate(model: PeftModel, tokenizer: AutoTokenizer, prompt: str, max_length: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate(adapter_path: Path, val_path: Path, max_length: int = 128) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_model(adapter_path)
    records = load_jsonl(val_path)

    predictions, references = [], []
    times = []

    for rec in records:
        start = time.time()
        pred = generate(model, tokenizer, rec["input"], max_length=max_length)
        times.append(time.time() - start)
        predictions.append(pred)
        references.append(rec["target"])

    bleu = corpus_bleu(predictions, [references])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(references, predictions)
    ]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    return {
        "adapter": adapter_path,
        "samples": len(records),
        "bleu4": bleu.score / 100,
        "rouge_l": avg_rouge,
        "avg_latency_ms": sum(times) / len(times) * 1000,
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python evaluate_mt5.py <title_adapter_path> <description_adapter_path>")
        sys.exit(1)

    title_adapter = Path(sys.argv[1])
    desc_adapter = Path(sys.argv[2])

    print("=== Title adapter ===")
    title_metrics = evaluate(title_adapter, DATA_DIR / "mt5_title_val.jsonl", max_length=64)
    print(f"  Samples:    {title_metrics['samples']}")
    print(f"  BLEU-4:     {title_metrics['bleu4']:.4f}")
    print(f"  ROUGE-L:    {title_metrics['rouge_l']:.4f}")
    print(f"  Avg latency: {title_metrics['avg_latency_ms']:.1f} ms")

    print("\n=== Description adapter ===")
    desc_metrics = evaluate(desc_adapter, DATA_DIR / "mt5_description_val.jsonl", max_length=200)
    print(f"  Samples:    {desc_metrics['samples']}")
    print(f"  BLEU-4:     {desc_metrics['bleu4']:.4f}")
    print(f"  ROUGE-L:    {desc_metrics['rouge_l']:.4f}")
    print(f"  Avg latency: {desc_metrics['avg_latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
