# mT5 LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `google/mt5-base` with LoRA on the existing Russian captions dataset to generate product titles and descriptions, then integrate the trained adapters into the backend.

**Architecture:** Two separate LoRA adapters (title and description) are trained on prompt/target pairs derived from `training/data/captions.jsonl`. The backend `TextGenerator` loads the base mT5 model once and wraps it with `PeftModel` for each adapter, selecting the active adapter per task.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, PEFT, Datasets, sacrebleu, rouge-score, FastAPI.

## Global Constraints

- Base model: `google/mt5-base`
- LoRA target modules: `["q", "v"]`
- `r=16`, `lora_alpha=32`, `lora_dropout=0.1`
- Training platform: Google Colab (T4 GPU)
- Adapters saved to `backend/model_cache/snapcard_mt5_title_lora/` and `backend/model_cache/snapcard_mt5_description_lora/`
- Existing tests must continue to pass
- Backend must degrade gracefully to rule-based fallback if adapters are missing or fail to load

---

## File Structure

| File | Responsibility |
|------|----------------|
| `training/prepare_mt5_dataset.py` | Merge captions + metadata, create train/val splits, emit title/description datasets |
| `training/train_mt5_lora.ipynb` | Colab notebook: install deps, load data, train two LoRA adapters, run sample inference |
| `training/evaluate_mt5.py` | Evaluate trained adapters on test split with BLEU-4 / ROUGE-L |
| `backend/app/config.py` | Add `mt5_title_lora_path` and `mt5_description_lora_path` settings |
| `backend/app/ml/text_generator.py` | Load adapters, implement title/description generation with adapter switching |
| `backend/app/ml/pipeline.py` | Ensure `caption_ru` and `category` are passed correctly (already done) |
| `backend/pyproject.toml` | Ensure `peft` is in ML dependencies |
| `backend/Dockerfile` | Install `peft` when `INSTALL_ML=true` |
| `README.md` | Update ML-pipeline and results sections |

---

### Task 1: Create Dataset Preparation Script

**Files:**
- Create: `training/prepare_mt5_dataset.py`
- Test: manual (`python training/prepare_mt5_dataset.py`)

**Interfaces:**
- Consumes: `training/data/captions.jsonl`, `training/data/metadata.jsonl`
- Produces: `training/data/mt5_title_train.jsonl`, `training/data/mt5_title_val.jsonl`, `training/data/mt5_description_train.jsonl`, `training/data/mt5_description_val.jsonl`

Each record has fields:
- `input`: prompt string
- `target`: expected output string
- `id`: original record id
- `category`: product category

- [ ] **Step 1: Implement data merging and split logic**

```python
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
CAPTIONS_FILE = DATA_DIR / "captions.jsonl"
METADATA_FILE = DATA_DIR / "metadata.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_split(records: list[dict], val_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    by_category = defaultdict(list)
    for rec in records:
        by_category[rec["category"]].append(rec)

    train, val = [], []
    for cat, items in by_category.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def build_title_record(caption: dict, meta: dict) -> dict:
    return {
        "id": caption["id"],
        "category": meta["master_category"],
        "input": f"Заголовок товара: {caption['caption_ru']}. Категория: {meta['master_category']}.",
        "target": caption["title"],
    }


def build_description_record(caption: dict, meta: dict) -> dict:
    return {
        "id": caption["id"],
        "category": meta["master_category"],
        "input": f"Описание товара: {caption['caption_ru']}. Категория: {meta['master_category']}. Заголовок: {caption['title']}.",
        "target": caption["description"],
    }


def main():
    captions = {c["id"]: c for c in load_jsonl(CAPTIONS_FILE)}
    metadata = {m["id"]: m for m in load_jsonl(METADATA_FILE)}

    common_ids = sorted(set(captions.keys()) & set(metadata.keys()))

    title_records, description_records = [], []
    for rid in common_ids:
        cap = captions[rid]
        meta = metadata[rid]
        if not cap.get("caption_ru") or not cap.get("title") or not cap.get("description"):
            continue
        title_records.append(build_title_record(cap, meta))
        description_records.append(build_description_record(cap, meta))

    title_train, title_val = stratified_split(title_records)
    desc_train, desc_val = stratified_split(description_records)

    def save(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save(title_train, DATA_DIR / "mt5_title_train.jsonl")
    save(title_val, DATA_DIR / "mt5_title_val.jsonl")
    save(desc_train, DATA_DIR / "mt5_description_train.jsonl")
    save(desc_val, DATA_DIR / "mt5_description_val.jsonl")

    print(f"Title: {len(title_train)} train / {len(title_val)} val")
    print(f"Description: {len(desc_train)} train / {len(desc_val)} val")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script and verify outputs**

Run:
```bash
cd training
python prepare_mt5_dataset.py
```

Expected output:
```
Title: 583 train / 146 val
Description: 583 train / 146 val
```

- [ ] **Step 3: Commit**

```bash
git add training/prepare_mt5_dataset.py training/data/mt5_*.jsonl
git commit -m "feat(training): add mT5 LoRA dataset preparation script"
```

---

### Task 2: Add Backend Settings for Adapter Paths

**Files:**
- Modify: `backend/app/config.py`

**Interfaces:**
- Produces: `Settings.mt5_title_lora_path: Optional[Path]`, `Settings.mt5_description_lora_path: Optional[Path]`

- [ ] **Step 1: Add new settings**

```python
mt5_title_lora_path: Optional[Path] = None
mt5_description_lora_path: Optional[Path] = None
```

Add after the existing `blip_lora_path` setting with the same style.

- [ ] **Step 2: Verify config loads**

Run:
```bash
cd backend
python -c "from app.config import settings; print(settings.model_dump())"
```

Expected: no errors, new keys present.

- [ ] **Step 3: Commit**

```bash
git add backend/app/config.py
git commit -m "feat(config): add mT5 LoRA adapter path settings"
```

---

### Task 3: Update TextGenerator to Load and Use LoRA Adapters

**Files:**
- Modify: `backend/app/ml/text_generator.py`
- Modify: `backend/app/config.py` (already done in Task 2)

**Interfaces:**
- Consumes: `Settings.mt5_title_lora_path`, `Settings.mt5_description_lora_path`
- Produces: `TextGenerator.generate_title()` and `TextGenerator.generate_description()` use adapters when available

- [ ] **Step 1: Refactor TextGenerator to load adapters**

```python
from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextGenerator:
    def __init__(
        self,
        model_name: str = "google/mt5-base",
        device: str = "cpu",
        title_lora_path: Optional[Path] = None,
        description_lora_path: Optional[Path] = None,
    ):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.base_model.eval()

        self.title_model = self._load_adapter(self.base_model, title_lora_path)
        self.description_model = self._load_adapter(self.base_model, description_lora_path)

    def _load_adapter(self, base_model, adapter_path: Optional[Path]):
        if adapter_path and Path(adapter_path).exists():
            try:
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load adapter {adapter_path}: {e}")
        return None

    def _generate(
        self,
        model,
        prompt: str,
        max_length: int = 200,
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(raw)
```

- [ ] **Step 2: Update generate_title and generate_description to use adapters**

```python
def generate_title(self, caption: str, category: str, caption_ru: str = "") -> str:
    source = caption_ru.strip() if caption_ru else caption
    if not source:
        return self._fallback_title(caption, category)

    prompt = f"Заголовок товара: {source}. Категория: {category}."

    if self.title_model is None:
        return self._fallback_title(caption, category)

    try:
        title = self._generate(self.title_model, prompt, max_length=64)
        if title and self._is_russian(title):
            return title
        logger.warning("mT5 title adapter returned empty or non-Russian text, using fallback")
    except Exception as e:
        logger.error(f"mT5 title generation failed: {e}")

    return self._fallback_title(caption, category)


def generate_description(
    self, caption: str, category: str, title: str, caption_ru: str = ""
) -> str:
    source = caption_ru.strip() if caption_ru else caption
    if not source:
        return self._fallback_description(caption, category, caption_ru)

    prompt = f"Описание товара: {source}. Категория: {category}. Заголовок: {title}."

    if self.description_model is None:
        return self._fallback_description(caption, category, caption_ru)

    try:
        description = self._generate(self.description_model, prompt, max_length=200)
        if description and self._is_russian(description):
            return description
        logger.warning("mT5 description adapter returned empty or non-Russian text, using fallback")
    except Exception as e:
        logger.error(f"mT5 description generation failed: {e}")

    return self._fallback_description(caption, category, caption_ru)
```

- [ ] **Step 3: Update pipeline to pass adapter paths**

In `backend/app/ml/pipeline.py`, change `TextGenerator` instantiation:

```python
from app.config import settings

self._text_generator = TextGenerator(
    model_name=settings.text_model,
    device=device,
    title_lora_path=settings.mt5_title_lora_path,
    description_lora_path=settings.mt5_description_lora_path,
)
```

- [ ] **Step 4: Run backend tests**

Run:
```bash
cd backend
pytest -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/text_generator.py backend/app/ml/pipeline.py
git commit -m "feat(ml): load mT5 title/description LoRA adapters"
```

---

### Task 4: Ensure PEFT is Installed

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/Dockerfile`

- [ ] **Step 1: Add `peft` to ML dependencies**

In `backend/pyproject.toml` under `[project.optional-dependencies]` `ml`:

```toml
ml = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
]
```

- [ ] **Step 2: Add `peft` to Dockerfile**

In `backend/Dockerfile`, change the ML install block:

```dockerfile
RUN if [ "$INSTALL_ML" = "true" ] ; then \
        pip install --no-cache-dir torch transformers accelerate peft ; \
    fi
```

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml backend/Dockerfile
git commit -m "chore(deps): add peft for mT5 LoRA adapters"
```

---

### Task 5: Create Colab Training Notebook

**Files:**
- Create: `training/train_mt5_lora.ipynb`

**Interfaces:**
- Consumes: `mt5_title_train.jsonl`, `mt5_title_val.jsonl`, `mt5_description_train.jsonl`, `mt5_description_val.jsonl`
- Produces: `snapcard_mt5_title_lora/` and `snapcard_mt5_description_lora/` adapter directories

- [ ] **Step 1: Create notebook with cells for title training**

Key cells:
1. Install dependencies: `transformers`, `datasets`, `peft`, `accelerate`, `sacrebleu`, `rouge-score`.
2. Mount Google Drive and set paths.
3. Load datasets from JSONL.
4. Define `SnapCardSeq2SeqDataset` returning `input_ids`, `attention_mask`, `labels`.
5. Load base `google/mt5-base`.
6. Apply `LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.1, task_type="SEQ_2_SEQ_LM")`.
7. Train title adapter with `Trainer` for 5 epochs, save to `/content/drive/MyDrive/snapcard_mt5_title_lora`.
8. Train description adapter similarly.
9. Run sample inference on 5 validation examples.

- [ ] **Step 2: Add sample training code block to notebook**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

BASE_MODEL = "google/mt5-base"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 64  # 200 for description

 tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
 model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)

def preprocess(examples):
    inputs = examples["input"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LEN, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LEN, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("json", data_files={"train": "mt5_title_train.jsonl", "val": "mt5_title_val.jsonl"})
dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./mt5_title_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
model.save_pretrained("/content/drive/MyDrive/snapcard_mt5_title_lora")
```

- [ ] **Step 3: Commit notebook skeleton**

```bash
git add training/train_mt5_lora.ipynb
git commit -m "feat(training): add Colab notebook for mT5 LoRA fine-tuning"
```

---

### Task 6: Create MT5 Evaluation Script

**Files:**
- Create: `training/evaluate_mt5.py`

**Interfaces:**
- Consumes: trained adapter directories, `mt5_title_val.jsonl`, `mt5_description_val.jsonl`
- Produces: BLEU-4 and ROUGE-L scores printed to stdout

- [ ] **Step 1: Implement evaluation script**

```python
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "google/mt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model(adapter_path: Path):
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(DEVICE)
    model.eval()
    return model


def generate(model, tokenizer, prompt: str, max_length: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate(adapter_path: Path, val_path: Path, max_length: int = 128):
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
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    print(f"Adapter: {adapter_path}")
    print(f"Samples: {len(records)}")
    print(f"BLEU-4: {bleu.score / 100:.4f}")
    print(f"ROUGE-L: {avg_rouge:.4f}")
    print(f"Avg latency: {sum(times) / len(times) * 1000:.1f} ms")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate_mt5.py <title_adapter_path> <description_adapter_path>")
        sys.exit(1)

    title_adapter = Path(sys.argv[1])
    desc_adapter = Path(sys.argv[2])

    data_dir = Path(__file__).parent / "data"
    print("=== Title ===")
    evaluate(title_adapter, data_dir / "mt5_title_val.jsonl", max_length=64)
    print("\n=== Description ===")
    evaluate(desc_adapter, data_dir / "mt5_description_val.jsonl", max_length=200)
```

- [ ] **Step 2: Commit**

```bash
git add training/evaluate_mt5.py
git commit -m "feat(training): add mT5 LoRA evaluation script"
```

---

### Task 7: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update ML Pipeline section**

Add mT5 LoRA to the pipeline table:

```markdown
| Этап | Модель | Назначение |
|------|--------|------------|
| Captioning | BLIP + LoRA | Описание изображения на английском |
| Перевод | Helsinki-NLP/opus-mt-en-ru | Перевод caption на русский |
| Классификация | CLIP | Zero-shot категория и теги |
| Генерация текста | mT5-base + LoRA | Русский заголовок и описание |
| SEO | Rule-based | SEO-метаданные |
```

- [ ] **Step 2: Update Results section**

Add placeholder for mT5 metrics after training:

```markdown
### mT5 Title & Description Generation
| Task | BLEU-4 | ROUGE-L |
|------|--------|---------|
| Title | TBD | TBD |
| Description | TBD | TBD |
```

- [ ] **Step 3: Add training instructions**

Add a short section after "LoRA Fine-tuning":

```markdown
### mT5 LoRA Fine-Tuning
1. Prepare datasets: `python training/prepare_mt5_dataset.py`
2. Open `training/train_mt5_lora.ipynb` in Google Colab.
3. Run all cells to train title and description adapters.
4. Copy adapters to `backend/model_cache/snapcard_mt5_title_lora/` and `backend/model_cache/snapcard_mt5_description_lora/`.
5. Evaluate: `python training/evaluate_mt5.py <title_adapter> <description_adapter>`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document mT5 LoRA training pipeline"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full backend test suite**

```bash
cd backend
pytest -v
```

Expected: 15 passed.

- [ ] **Step 2: Push all commits**

```bash
git push origin main
```

- [ ] **Step 3: Provide handoff notes**

After code is merged, user must:
1. Run `python training/prepare_mt5_dataset.py` locally to generate datasets.
2. Upload datasets and notebook to Colab.
3. Run training and download adapters.
4. Place adapters in `backend/model_cache/`.
5. Set environment variables or rely on default paths in `config.py`.
6. Restart backend and test `/api/v1/cards/generate`.

---

## Spec Coverage Check

| Spec Requirement | Task |
|------------------|------|
| Prepare title/description datasets from captions.jsonl | Task 1 |
| Add adapter path settings | Task 2 |
| Load adapters in TextGenerator | Task 3 |
| Install PEFT dependency | Task 4 |
| Colab notebook for training | Task 5 |
| Evaluation with BLEU/ROUGE | Task 6 |
| README updates | Task 7 |
| Tests still pass | Task 8 |

## Placeholder Scan

No TBD/TODO in implementation steps. The README results table intentionally uses TBD as placeholders for metrics obtained after training, which is acceptable because the actual training happens outside this repository in Colab.

## Type Consistency

- `mt5_title_lora_path` and `mt5_description_lora_path` are `Optional[Path]` in config and passed to `TextGenerator` constructor as `Optional[Path]`.
- `TextGenerator.generate_title()` and `generate_description()` signatures remain unchanged from the caller's perspective; only internal adapter usage changes.
