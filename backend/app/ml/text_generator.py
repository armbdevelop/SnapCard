import logging
import re
from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Sentinel tokens that mT5 sometimes emits when confused
_EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")

# Heuristic: if output contains mostly ASCII letters, it's likely English.
_CYRILLIC_RE = re.compile(r"[А-ЯЁа-яё]")


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

        # Each adapter needs its own base model instance: wrapping the same
        # base in two PeftModels makes the second adapter overwrite the first
        # (both register as adapter "default" on the shared modules).
        self.title_model = self._load_adapter(title_lora_path)
        self.description_model = self._load_adapter(description_lora_path)

    def _load_adapter(self, adapter_path: Optional[Path]) -> Optional[PeftModel]:
        if not adapter_path or not Path(adapter_path).exists():
            logger.warning(f"mT5 adapter not found at {adapter_path}, generation will use fallback")
            return None

        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            base_model.eval()
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model.eval()
            logger.info(f"Loaded mT5 adapter from {adapter_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load mT5 adapter {adapter_path}: {e}")
            return None

    def _clean_output(self, text: str) -> str:
        """Remove mT5 sentinel tokens and normalize whitespace."""
        text = _EXTRA_ID_RE.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_russian(self, text: str) -> bool:
        """Heuristic check that generated text contains Cyrillic characters."""
        return bool(_CYRILLIC_RE.search(text))

    def _generate(self, model, prompt: str, max_length: int = 200) -> str:
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

    def generate_title(self, caption: str, category: str, caption_ru: str = "") -> str:
        """Generate a Russian product title using mT5 LoRA, fallback to rule-based if needed."""
        source = caption_ru.strip() if caption_ru else caption
        if not source:
            return self._fallback_title(caption, category)

        if self.title_model is None:
            return self._fallback_title(caption, category)

        # Prompt must exactly match the training format (prepare_mt5_dataset.py),
        # otherwise the LoRA adapter degenerates to echoing prompt fragments.
        prompt = f"Заголовок товара: {source}. Категория: {category}."

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
        """Generate a Russian product description using mT5 LoRA, fallback to rule-based if needed."""
        source = caption_ru.strip() if caption_ru else caption
        if not source:
            return self._fallback_description(caption, category, caption_ru)

        if self.description_model is None:
            return self._fallback_description(caption, category, caption_ru)

        # Prompt must exactly match the training format (prepare_mt5_dataset.py).
        prompt = (
            f"Описание товара: {source}. Категория: {category}. "
            f"Заголовок: {title}."
        )

        try:
            description = self._generate(self.description_model, prompt, max_length=200)
            if description and self._is_russian(description):
                return description
            logger.warning("mT5 description adapter returned empty or non-Russian text, using fallback")
        except Exception as e:
            logger.error(f"mT5 description generation failed: {e}")

        return self._fallback_description(caption, category, caption_ru)

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

    def _fallback_description(
        self, caption: str, category: str, caption_ru: str = ""
    ) -> str:
        """Marketing-style product description for e-commerce."""
        if caption_ru:
            return (
                f"Представляем вашему вниманию товар из категории «{category}». "
                f"{caption_ru} "
                f"Превосходное качество по доступной цене. "
                f"Быстрая доставка и удобная оплата. Закажите прямо сейчас!"
            )
        return f"Отличный товар из категории «{category}». Высокое качество, доступная цена. Закажите прямо сейчас!"

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

        # Brand detection (naive keyword matching from caption)
        brand_map = {
            "nike": "Nike", "adidas": "Adidas", "puma": "Puma",
            "apple": "Apple", "samsung": "Samsung", "sony": "Sony",
            "xiaomi": "Xiaomi", "lg": "LG", "philips": "Philips",
            "bosch": "Bosch", "levi": "Levi's", "zara": "Zara",
            "gucci": "Gucci", "prada": "Prada", "hugo": "Hugo Boss",
        }
        for en, brand in brand_map.items():
            if en in caption_lower:
                chars["Бренд"] = brand
                break
        if "Бренд" not in chars:
            chars["Бренд"] = "Не указан"

        # Country of origin (placeholder)
        chars["Страна производства"] = "Уточняйте у продавца"

        # Category-specific fields
        if category in ("Электроника", "Бытовая техника"):
            chars["Гарантия"] = "12 месяцев"
            chars["Напряжение"] = "220 В"
        elif category in ("Одежда", "Обувь"):
            chars["Сезон"] = "Всесезонный"
            chars["Размер"] = "Универсальный"
        elif category == "Продукты питания":
            chars["Срок хранения"] = "Смотрите на упаковке"

        return chars
