"""Generate the BLIP LoRA fine-tuning notebook via nbformat."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
    "accelerator": "GPU",
    "colab": {
        "provenance": [],
        "gpuType": "T4",
    },
})

# Cell 1: Title
nb.cells.append(nbf.v4.new_markdown_cell("""\
# SnapCard — BLIP LoRA Fine-tuning

Fine-tune a LoRA adapter for [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) \
on a fashion product images dataset to generate better Russian captions.

**Requirements:** GPU runtime (T4 or higher). Go to *Runtime → Change runtime type → T4 GPU*.

**Steps:**
1. Install dependencies
2. Mount Google Drive with prepared dataset
3. Configure training parameters
4. Load data, train, evaluate
5. Download the trained adapter\
"""))

# Cell 2: Install dependencies
nb.cells.append(nbf.v4.new_code_cell("""\
!pip install -q torch transformers peft datasets accelerate bitsandbytes tqdm pandas pillow\
"""))

# Cell 3: Mount Google Drive
nb.cells.append(nbf.v4.new_code_cell("""\
from google.colab import drive
drive.mount('/content/drive')\
"""))

# Cell 4: Parameters
nb.cells.append(nbf.v4.new_code_cell("""\
# === Training Parameters ===
# Change these paths to match your Google Drive structure
DATASET_PATH = "/content/drive/MyDrive/snapcard/data"  # folder with train.jsonl, val.jsonl, captions.jsonl, images/
OUTPUT_PATH = "/content/drive/MyDrive/snapcard/output"  # where to save checkpoints and final adapter

BASE_MODEL = "Salesforce/blip-image-captioning-large"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 500
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1\
"""))

# Cell 5: Load data
nb.cells.append(nbf.v4.new_code_cell("""\
import json
import os
from pathlib import Path

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# Load splits and captions
train_meta = load_jsonl(os.path.join(DATASET_PATH, "train.jsonl"))
val_meta = load_jsonl(os.path.join(DATASET_PATH, "val.jsonl"))
captions = load_jsonl(os.path.join(DATASET_PATH, "captions.jsonl"))

# Build caption lookup by id
caption_map = {c["id"]: c["caption_ru"] for c in captions}

# Merge: attach caption_ru to each record
def merge_with_captions(meta_list, caption_map):
    merged = []
    for m in meta_list:
        cap = caption_map.get(m["id"])
        if cap:
            m["caption_ru"] = cap
            m["full_image_path"] = os.path.join(DATASET_PATH, m["image_path"])
            merged.append(m)
    return merged

train_data = merge_with_captions(train_meta, caption_map)
val_data = merge_with_captions(val_meta, caption_map)

print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
if train_data:
    print(f"Example caption: {train_data[0]['caption_ru']}")\
"""))

# Cell 6: PyTorch Dataset
nb.cells.append(nbf.v4.new_code_cell("""\
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor

class SnapCardDataset(Dataset):
    \"\"\"Dataset for BLIP fine-tuning with Russian captions.\"\"\"

    def __init__(self, data, processor, max_length=128):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["full_image_path"]).convert("RGB")
        caption = item["caption_ru"]

        # Process image
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Squeeze batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": input_ids,
        }

print("SnapCardDataset class defined.")\
"""))

# Cell 7: Load BLIP + LoRA
nb.cells.append(nbf.v4.new_code_cell("""\
from transformers import BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

# Load base model and processor
processor = BlipProcessor.from_pretrained(BASE_MODEL)
model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL)

# Configure LoRA
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "value"],
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")\
"""))

# Cell 8: Training loop
nb.cells.append(nbf.v4.new_code_cell("""\
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

# Create datasets and dataloaders
train_dataset = SnapCardDataset(train_data, processor)
val_dataset = SnapCardDataset(val_data, processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - WARMUP_STEPS))
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])

# Training
os.makedirs(OUTPUT_PATH, exist_ok=True)
train_losses = []  # (step, loss) pairs for plotting
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        epoch_loss += outputs.loss.item()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                train_losses.append((global_step, avg_loss))
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Handle remaining gradients
    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / max(1, len(val_loader))

    print(f"Epoch {epoch + 1}/{EPOCHS} — Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(OUTPUT_PATH, f"checkpoint_epoch_{epoch + 1}")
    model.save_pretrained(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

print("Training complete!")\
"""))

# Cell 9: Loss plot
nb.cells.append(nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt

if train_losses:
    steps, losses = zip(*train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("SnapCard BLIP LoRA — Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "training_loss.png"), dpi=150)
    plt.show()
else:
    print("No loss data recorded.")\
"""))

# Cell 10: Save final adapter
nb.cells.append(nbf.v4.new_code_cell("""\
final_path = os.path.join(OUTPUT_PATH, "snapcard_lora_final")
model.save_pretrained(final_path)
processor.save_pretrained(final_path)
print(f"Final adapter saved to: {final_path}")
print("Download this folder and place it in backend/model_cache/snapcard_lora/")\
"""))

# Cell 11: Inference comparison
nb.cells.append(nbf.v4.new_code_cell("""\
from peft import PeftModel

# Load test data
test_meta = load_jsonl(os.path.join(DATASET_PATH, "test.jsonl"))
test_data = merge_with_captions(test_meta, caption_map)

# Take 3 samples for comparison
samples = test_data[:3]

# Load base model (without LoRA)
base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
base_model.eval()

# Load fine-tuned model (with LoRA)
finetuned_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL)
finetuned_model = PeftModel.from_pretrained(finetuned_model, final_path).to(device)
finetuned_model.eval()

print("=" * 80)
print("INFERENCE COMPARISON: Base vs Fine-tuned vs Reference")
print("=" * 80)

for i, sample in enumerate(samples):
    image = Image.open(sample["full_image_path"]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Base model caption
    with torch.no_grad():
        base_out = base_model.generate(**inputs, max_new_tokens=50, num_beams=5)
    base_caption = processor.decode(base_out[0], skip_special_tokens=True)

    # Fine-tuned model caption
    with torch.no_grad():
        ft_out = finetuned_model.generate(**inputs, max_new_tokens=50, num_beams=5)
    ft_caption = processor.decode(ft_out[0], skip_special_tokens=True)

    reference = sample["caption_ru"]

    print(f"\\nSample {i + 1}: {sample.get('product_display_name', 'N/A')}")
    print(f"  Base model:    {base_caption}")
    print(f"  Fine-tuned:    {ft_caption}")
    print(f"  Reference:     {reference}")
    print("-" * 80)\
"""))

# Write notebook
nbf.write(nb, "training/train_blip_lora.ipynb")
print("Notebook saved to training/train_blip_lora.ipynb")
