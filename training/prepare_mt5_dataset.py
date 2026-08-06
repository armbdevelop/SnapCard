import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
CAPTIONS_FILE = DATA_DIR / "captions.jsonl"
METADATA_FILE = DATA_DIR / "metadata.jsonl"

# Map English masterCategory from the fashion dataset to Russian categories
# used by the CLIP classifier in the backend pipeline.
CATEGORY_MAP = {
    "Apparel": "Одежда",
    "Footwear": "Обувь",
    "Accessories": "Аксессуары",
    "Personal Care": "Косметика",
    "Home": "Мебель",
    "Sporting Goods": "Спорт и отдых",
    "Free Items": "Другое",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_russian_category(master_category: str) -> str:
    return CATEGORY_MAP.get(master_category, "Другое")


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


def build_title_record(caption: dict, category: str) -> dict:
    return {
        "id": caption["id"],
        "category": category,
        "input": f"Заголовок товара: {caption['caption_ru']}. Категория: {category}.",
        "target": caption["title"],
    }


def build_description_record(caption: dict, category: str) -> dict:
    return {
        "id": caption["id"],
        "category": category,
        "input": f"Описание товара: {caption['caption_ru']}. Категория: {category}. Заголовок: {caption['title']}.",
        "target": caption["description"],
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError(f"Captions file not found: {CAPTIONS_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    captions = {c["id"]: c for c in load_jsonl(CAPTIONS_FILE)}
    metadata = {m["id"]: m for m in load_jsonl(METADATA_FILE)}

    common_ids = sorted(set(captions.keys()) & set(metadata.keys()))

    title_records, description_records = [], []
    skipped = 0
    for rid in common_ids:
        cap = captions[rid]
        meta = metadata[rid]

        if not cap.get("caption_ru") or not cap.get("title") or not cap.get("description"):
            skipped += 1
            continue

        category = to_russian_category(meta.get("master_category", "Другое"))
        title_records.append(build_title_record(cap, category))
        description_records.append(build_description_record(cap, category))

    title_train, title_val = stratified_split(title_records)
    desc_train, desc_val = stratified_split(description_records)

    save_jsonl(title_train, DATA_DIR / "mt5_title_train.jsonl")
    save_jsonl(title_val, DATA_DIR / "mt5_title_val.jsonl")
    save_jsonl(desc_train, DATA_DIR / "mt5_description_train.jsonl")
    save_jsonl(desc_val, DATA_DIR / "mt5_description_val.jsonl")

    print(f"Records with both captions and metadata: {len(common_ids)}")
    print(f"Skipped (missing fields): {skipped}")
    print(f"Title: {len(title_train)} train / {len(title_val)} val")
    print(f"Description: {len(desc_train)} train / {len(desc_val)} val")
    print(f"Categories: {sorted({r['category'] for r in title_records})}")


if __name__ == "__main__":
    main()
