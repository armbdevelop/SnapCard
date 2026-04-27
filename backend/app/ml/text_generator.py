import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Sentinel tokens that mT5 sometimes emits when confused by English prompts
_EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")


class TextGenerator:
    def __init__(self, model_name: str = "google/mt5-base", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def _clean_output(self, text: str) -> str:
        """Remove mT5 sentinel tokens and normalize whitespace."""
        text = _EXTRA_ID_RE.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _generate(self, prompt: str, max_length: int = 200) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(raw)

    def generate_title(self, caption: str, category: str) -> str:
        # Временно всегда используем fallback
        return self._fallback_title(caption, category)

    def generate_description(self, caption: str, category: str, title: str) -> str:
        # Временно всегда используем fallback
        return self._fallback_description(caption, category)

    def generate_characteristics(self, caption: str, category: str) -> dict[str, str]:
        """Generate product characteristics based on caption and category."""
        return self._infer_characteristics(caption, category)

    def _fallback_title(self, caption: str, category: str) -> str:
        """Rule-based fallback title in Russian."""
        category_titles = {
            "Электроника": "Электронное устройство",
            "Одежда": "Предмет одежды",
            "Обувь": "Обувь",
            "Аксессуары": "Аксессуар",
            "Мебель": "Предмет мебели",
            "Продукты питания": "Продукт питания",
            "Косметика": "Косметическое средство",
            "Спорт и отдых": "Спортивный товар",
            "Игрушки": "Игрушка",
            "Книги": "Книга",
            "Бытовая техника": "Бытовая техника",
            "Инструменты": "Инструмент",
            "Автотовары": "Автотовар",
        }
        base = category_titles.get(category, "Товар")

        return base

    def _fallback_description(self, caption: str, category: str) -> str:
        """Rule-based fallback description in Russian."""
        parts = [f"Товар из категории «{category}»."]
        if caption:
            parts.append(f"На изображении: {caption}.")
        parts.append("Высокое качество, доступная цена.")
        return " ".join(parts)

    def _infer_characteristics(self, caption: str, category: str) -> dict[str, str]:
        """Infer characteristics from caption and category."""
        chars: dict[str, str] = {"Категория": category}

        caption_lower = caption.lower() if caption else ""

        # Color detection
        color_map = {
            "red": "Красный", "blue": "Синий", "green": "Зелёный",
            "black": "Чёрный", "white": "Белый", "yellow": "Жёлтый",
            "pink": "Розовый", "purple": "Фиолетовый", "orange": "Оранжевый",
            "brown": "Коричневый", "gray": "Серый", "grey": "Серый",
            "silver": "Серебристый", "gold": "Золотистый",
        }
        for en, ru in color_map.items():
            if en in caption_lower:
                chars["Цвет"] = ru
                break

        # Material detection
        material_map = {
            "leather": "Кожа", "wood": "Дерево", "wooden": "Дерево",
            "metal": "Металл", "plastic": "Пластик", "glass": "Стекло",
            "cotton": "Хлопок", "silk": "Шёлк", "rubber": "Резина",
            "ceramic": "Керамика", "fabric": "Ткань", "steel": "Сталь",
        }
        for en, ru in material_map.items():
            if en in caption_lower:
                chars["Материал"] = ru
                break

        # Category-specific defaults
        if category in ("Электроника", "Бытовая техника"):
            chars["Гарантия"] = "12 месяцев"
        elif category in ("Одежда", "Обувь"):
            chars["Сезон"] = "Всесезонный"
        elif category == "Продукты питания":
            chars["Срок хранения"] = "Смотрите на упаковке"

        return chars
