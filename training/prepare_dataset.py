"""Download and prepare a fashion product images dataset for BLIP fine-tuning.

Usage:
    python training/prepare_dataset.py --size 1000 --output training/data
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METADATA_FIELDS = [
    "id",
    "image_path",
    "master_category",
    "sub_category",
    "article_type",
    "base_colour",
    "season",
    "usage",
    "product_display_name",
]


def resize_with_padding(image: Image.Image, target_size: int = 384) -> Image.Image:
    """Resize image preserving aspect ratio with white padding (letterboxing)."""
    image = image.convert("RGB")
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


def stratified_sample(dataset, size: int, key: str = "masterCategory") -> list[int]:
    """Return indices for a stratified sample by the given key."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset):
        buckets[row.get(key, "Unknown")].append(idx)

    selected: list[int] = []
    total = len(dataset)
    for cat, indices in buckets.items():
        n = max(1, round(len(indices) / total * size))
        n = min(n, len(indices))
        selected.extend(random.sample(indices, n))

    # Adjust to exact size
    if len(selected) > size:
        selected = random.sample(selected, size)
    elif len(selected) < size:
        remaining = set(range(total)) - set(selected)
        extra = random.sample(list(remaining), min(size - len(selected), len(remaining)))
        selected.extend(extra)

    return selected


def stratified_split(
    records: list[dict], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), key: str = "master_category"
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test with stratification."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get(key, "Unknown")].append(r)

    train, val, test = [], [], []
    for cat, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * ratios[0]))
        n_val = max(0, round(n * ratios[1]))
        # Ensure at least 1 in train
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare fashion dataset for BLIP fine-tuning")
    parser.add_argument("--size", type=int, default=1000, help="Number of samples to select")
    parser.add_argument("--output", type=str, default="training/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset ashraq/fashion-product-images-small from HuggingFace...")
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    logger.info("Dataset loaded: %d records", len(ds))

    # Stratified sampling
    indices = stratified_sample(ds, args.size)
    logger.info("Selected %d samples (stratified by masterCategory)", len(indices))

    # Process samples
    records = []
    for idx in tqdm(indices, desc="Processing images"):
        row = ds[idx]
        sample_id = row.get("id", idx)
        image: Image.Image | None = row.get("image")
        if image is None:
            continue

        # Save image
        img_filename = f"{sample_id}.jpg"
        img_path = images_dir / img_filename
        resized = resize_with_padding(image, 384)
        resized.save(img_path, "JPEG", quality=95)

        record = {
            "id": sample_id,
            "image_path": f"images/{img_filename}",
            "master_category": row.get("masterCategory", ""),
            "sub_category": row.get("subCategory", ""),
            "article_type": row.get("articleType", ""),
            "base_colour": row.get("baseColour", ""),
            "season": row.get("season", ""),
            "usage": row.get("usage", ""),
            "product_display_name": row.get("productDisplayName", ""),
        }
        records.append(record)

    # Save full metadata
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved metadata: %s (%d records)", metadata_path, len(records))

    # Stratified split
    train, val, test = stratified_split(records)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_path = output_dir / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in split_data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Saved %s: %s (%d records)", split_name, split_path, len(split_data))

    logger.info("Done! Dataset prepared in %s", output_dir)


if __name__ == "__main__":
    main()
