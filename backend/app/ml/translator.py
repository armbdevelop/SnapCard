import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class EnRuTranslator:
    """Lightweight English-to-Russian translator (Helsinki-NLP/opus-mt-en-ru)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_name = "Helsinki-NLP/opus-mt-en-ru"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
            self.model.eval()
            self.available = True
            logger.info("EN→RU translator loaded")
        except Exception as e:
            logger.warning(f"Failed to load translator: {e}")
            self.available = False

    def translate(self, text: str) -> str:
        if not self.available or not text:
            return text
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model.generate(**inputs, max_length=128)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
