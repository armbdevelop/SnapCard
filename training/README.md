# SnapCard Training

Scripts for preparing data, generating captions, fine-tuning BLIP with LoRA, and evaluating results.

## Prerequisites

Install training dependencies from the project root:

```bash
cd backend
pip install -e ".[ml,training]"
```

## 1. Preparing the dataset

Downloads the [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) dataset from HuggingFace, selects a stratified subset, resizes images to 384×384, and splits into train/val/test.

```bash
python training/prepare_dataset.py --size 1000 --output training/data
```

Options:
- `--size N` — number of samples to select (default: 1000)
- `--output DIR` — output directory (default: `training/data`)
- `--seed N` — random seed (default: 42)

### Output structure

```
training/data/
├── images/          # Resized product images (384×384 JPEG)
│   ├── 12345.jpg
│   └── ...
├── metadata.jsonl   # Full metadata for all samples
├── train.jsonl      # Training split (~70%)
├── val.jsonl        # Validation split (~15%)
└── test.jsonl       # Test split (~15%)
```

Each JSONL line contains: `id`, `image_path`, `master_category`, `sub_category`, `article_type`, `base_colour`, `season`, `usage`, `product_display_name`.

## 2. Generating Russian captions

Uses an LLM via [OpenRouter](https://openrouter.ai) to generate Russian product cards (title, description, caption) for each record. These serve as ground truth for fine-tuning.

### Getting an API key

1. Go to [openrouter.ai](https://openrouter.ai) and sign up
2. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
3. Set the environment variable:
   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-...
   ```

### Running

```bash
python training/generate_captions.py \
    --input training/data/metadata.jsonl \
    --output training/data/captions.jsonl \
    --model google/gemini-2.0-flash-001
```

Options:
- `--input` — path to metadata JSONL (default: `training/data/metadata.jsonl`)
- `--output` — path to output captions JSONL (default: `training/data/captions.jsonl`)
- `--model` — OpenRouter model ID (default: `google/gemini-2.0-flash-001`). Alternatives: `anthropic/claude-3.5-haiku`, `openai/gpt-4o-mini`

The script is **idempotent** — if interrupted, re-run the same command and it will skip already processed records. Each result is appended to the output file immediately after generation, so nothing is lost on crash.

Typical run for 1000 records takes ~15-20 minutes and costs ~$0.30-0.50 with gemini-flash.

## 3. Training in Colab

The notebook `train_blip_lora.ipynb` fine-tunes a LoRA adapter for BLIP on the prepared dataset using Google Colab's free GPU.

### Setup

1. Upload the `training/data/` folder to Google Drive, e.g. to `My Drive/snapcard/data/`
2. Open `training/train_blip_lora.ipynb` in Google Colab (File → Upload notebook, or drag-and-drop)
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. In **Cell 4**, update the paths:
   - `DATASET_PATH` — path to your data folder on Drive (e.g. `/content/drive/MyDrive/snapcard/data`)
   - `OUTPUT_PATH` — where to save checkpoints (e.g. `/content/drive/MyDrive/snapcard/output`)
5. Run all cells (Runtime → Run all)

### What to expect

- Training on 700 samples with T4 GPU takes approximately 30-60 minutes (3 epochs)
- Checkpoints are saved after each epoch to Google Drive
- Final adapter is saved to `{OUTPUT_PATH}/snapcard_lora_final/`
- The last cell compares base vs fine-tuned captions on 3 test images

### After training

Download the `snapcard_lora_final/` folder from Google Drive and place it in `backend/model_cache/snapcard_lora/`. Then set the environment variable:

```bash
export SNAPCARD_BLIP_LORA_PATH=./model_cache/snapcard_lora
```

## 4. Evaluation

Compare captioning quality before and after fine-tuning using BLEU-4, ROUGE-L, average caption length, and inference time.

```bash
# Base BLIP
python training/evaluate.py --test training/data/test.jsonl --captions training/data/captions.jsonl --limit 50

# After fine-tuning
python training/evaluate.py --test training/data/test.jsonl --captions training/data/captions.jsonl --lora-path backend/model_cache/snapcard_lora --limit 50
```

Options:
- `--test` — path to test.jsonl
- `--captions` — path to captions.jsonl with reference `caption_ru`
- `--lora-path` — path to LoRA adapter folder (omit for base model)
- `--limit N` — limit number of test samples
- `--output` — path to save results JSON (default: `training/data/eval_results.json`)

Results are accumulated in `eval_results.json` so you can compare multiple runs.
