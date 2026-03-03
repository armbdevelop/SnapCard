from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


PRODUCT_CATEGORIES = [
    "Электроника",
    "Одежда",
    "Обувь",
    "Аксессуары",
    "Мебель",
    "Продукты питания",
    "Косметика",
    "Спорт и отдых",
    "Игрушки",
    "Книги",
    "Бытовая техника",
    "Инструменты",
    "Автотовары",
    "Другое",
]

# English labels for CLIP (it works better with English)
CATEGORY_LABELS_EN = [
    "electronics, gadgets, phones, computers",
    "clothing, shirts, pants, dresses",
    "shoes, boots, sneakers, footwear",
    "accessories, bags, watches, jewelry",
    "furniture, chairs, tables, sofas",
    "food, groceries, snacks, beverages",
    "cosmetics, makeup, skincare, beauty",
    "sports equipment, fitness, outdoor recreation",
    "toys, games, children's playthings",
    "books, magazines, publications",
    "home appliances, kitchen appliances",
    "tools, hardware, construction equipment",
    "car accessories, auto parts, vehicle",
    "miscellaneous product, other item",
]


class CategoryClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.categories = PRODUCT_CATEGORIES
        self.labels_en = CATEGORY_LABELS_EN

    def classify(self, image_path: str) -> tuple[str, list[str], float]:
        """Classify image into a product category and generate tags.

        Returns:
            Tuple of (category, tags, confidence)
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.labels_en,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)

        # Best category
        best_idx = probs.argmax().item()
        category = self.categories[best_idx]
        confidence = float(probs[best_idx].item())

        # Tags: top-3 categories with prob > 0.1
        top_indices = probs.argsort(descending=True)[:5]
        tags = []
        for idx in top_indices:
            idx = idx.item()
            if probs[idx].item() > 0.05:
                tags.append(self.categories[idx])

        return category, tags, confidence
