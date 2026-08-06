# SnapCard: Fine-tuning mT5 with LoRA for Title & Description Generation

## Goal
Fine-tune `google/mt5-base` using LoRA on the existing Russian captions dataset (`training/data/captions.jsonl`) so that the backend can generate meaningful Russian product titles and descriptions from `caption_ru + category` instead of using rule-based fallbacks.

## Context
The current `TextGenerator` loads `google/mt5-base` but always returns fallback templates because zero-shot generation produces low-quality or non-Russian output. A synthetic dataset of 729 Russian product cards (title, description, caption_ru) already exists, generated via Gemini Flash through OpenRouter.

## Scope
- Create a Colab-ready training notebook for mT5 LoRA fine-tuning.
- Train **two separate LoRA adapters**:
  1. `snapcard_mt5_title_lora` — generates title from `caption_ru + category`.
  2. `snapcard_mt5_description_lora` — generates description from `caption_ru + category + title`.
- Save adapters to `backend/model_cache/`.
- Update `backend/app/ml/text_generator.py` to load and use the adapters.
- Add evaluation script/notebook for BLEU-4 / ROUGE-L on a held-out test set.

## Out of Scope
- Full fine-tuning of mT5 (we use LoRA).
- Fine-tuning on incomplete records (only records with valid captions are used).
- Replacing CLIP or BLIP.

## Data Preparation

### Source Files
- `training/data/captions.jsonl` — contains `id`, `title`, `description`, `caption_ru`.
- `training/data/metadata.jsonl` — contains `id`, `master_category`.

### Merging
Join captions and metadata by `id`. Keep only records where `caption_ru`, `title`, and `description` are non-empty.

### Datasets

**Title dataset:**
- Input prompt: `Заголовок товара: {caption_ru}. Категория: {category}.`
- Target: `{title}`

**Description dataset:**
- Input prompt: `Описание товара: {caption_ru}. Категория: {category}. Заголовок: {title}.`
- Target: `{description}`

### Split
- 80% train / 20% validation.
- Stratify by `category` where possible.

## Model & Training

### Base Model
`google/mt5-base` (~1 GB).

### LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch 16) |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Max source length | 128 |
| Max target length | 64 (title), 200 (description) |
| Beam width at inference | 4 |

### Training Procedure
1. Load base mT5.
2. Train title adapter.
3. Save title adapter.
4. Re-load base mT5 (or reset adapter).
5. Train description adapter.
6. Save description adapter.
7. Run inference on 5–10 validation examples.

## Output Artifacts

```
backend/model_cache/
├── snapcard_mt5_title_lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── snapcard_description_lora/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Backend Integration

### TextGenerator Changes
- Accept two adapter paths via settings or constructor.
- Load base mT5 once.
- Wrap with `PeftModel` for each adapter or switch adapters via `set_adapter`.
- `generate_title(caption_ru, category)` loads title adapter, runs generation.
- `generate_description(caption_ru, category, title)` loads description adapter, runs generation.
- Keep rule-based fallback if generation fails or returns empty/non-Russian text.

### Settings
Add to `backend/app/config.py`:
```python
mt5_title_lora_path: Optional[Path] = None
mt5_description_lora_path: Optional[Path] = None
```

## Evaluation

### Metrics
- BLEU-4 (sacrebleu)
- ROUGE-L (rouge-score)

### Process
1. Load test split.
2. Generate title/description for each record.
3. Compare predictions with references.
4. Report per-task metrics and average inference latency.

## Files to Create/Modify

### New
- `training/train_mt5_lora.ipynb`
- `training/evaluate_mt5.py` (optional)
- `backend/model_cache/snapcard_mt5_title_lora/`
- `backend/model_cache/snapcard_mt5_description_lora/`

### Modified
- `backend/app/ml/text_generator.py`
- `backend/app/config.py`
- `backend/pyproject.toml` (ensure `peft` is in deps)
- `backend/Dockerfile` (ensure `peft` is installed)
- `README.md` (update ML-pipeline and results sections)

## Success Criteria
- Generated titles contain Russian text and relate to the product.
- Generated descriptions are longer than fallback templates and marketing-appropriate.
- BLEU-4 > 0.25 and ROUGE-L > 0.35 on validation set.
- Backend loads adapters successfully and uses them in `/cards/generate`.
- Existing tests still pass.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| 729 records is small | Use LoRA (low trainable params), dropout, small number of epochs |
| mT5 overfits | Monitor validation loss, early stopping |
| Adapter loading fails in backend | Keep fallback templates, log warning |
| Colab session expires | Save checkpoints after each epoch to Drive |
