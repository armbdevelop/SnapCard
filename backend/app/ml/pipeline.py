import logging

from app.config import settings

logger = logging.getLogger(__name__)


class MLPipeline:
    def __init__(self):
        self.is_loaded = False
        self._captioner = None
        self._classifier = None
        self._text_generator = None
        self._seo_generator = None

    def load_models(self):
        """Load all ML models. Called once at startup."""
        device = settings.device

        # SEO generator is always available (rule-based)
        from app.ml.seo_generator import SEOGenerator
        self._seo_generator = SEOGenerator()
        logger.info("SEO generator loaded (rule-based)")

        try:
            from app.ml.image_captioner import ImageCaptioner
            self._captioner = ImageCaptioner(model_name=settings.blip_model, device=device)
            logger.info("BLIP captioner loaded")
        except Exception as e:
            logger.warning(f"Failed to load captioner: {e}")

        try:
            from app.ml.category_classifier import CategoryClassifier
            self._classifier = CategoryClassifier(model_name=settings.clip_model, device=device)
            logger.info("CLIP classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")

        try:
            from app.ml.text_generator import TextGenerator
            self._text_generator = TextGenerator(model_name=settings.text_model, device=device)
            logger.info("Text generator loaded")
        except Exception as e:
            logger.warning(f"Failed to load text generator: {e}")
            # Try fallback model
            try:
                from app.ml.text_generator import TextGenerator
                self._text_generator = TextGenerator(
                    model_name=settings.text_model_fallback, device=device
                )
                logger.info("Fallback text generator loaded")
            except Exception as e2:
                logger.warning(f"Failed to load fallback text generator: {e2}")

        self.is_loaded = self._captioner is not None or self._classifier is not None
        logger.info(f"ML Pipeline loaded. Status: {self.get_status()}")

    def get_status(self) -> dict:
        return {
            "captioner": self._captioner is not None,
            "classifier": self._classifier is not None,
            "text_generator": self._text_generator is not None,
            "seo_generator": self._seo_generator is not None,
        }

    def process(self, image_path: str) -> dict:
        """Run the full ML pipeline on an image.

        Pipeline: Image -> BLIP (caption) -> CLIP (category+tags) -> mT5 (title+description) -> Rules (SEO)
        """
        caption = ""
        confidence = 0.0
        category = "Другое"
        tags: list[str] = []
        title = "Товар"
        description = ""
        characteristics: dict[str, str] = {}

        # Stage 1: Image captioning (BLIP)
        if self._captioner:
            try:
                caption, confidence = self._captioner.caption(image_path)
                logger.info(f"Caption: {caption} (confidence: {confidence:.2f})")
            except Exception as e:
                logger.error(f"Captioning failed: {e}")

        # Stage 2: Category classification (CLIP)
        if self._classifier:
            try:
                category, tags, cls_confidence = self._classifier.classify(image_path)
                confidence = (confidence + cls_confidence) / 2 if confidence > 0 else cls_confidence
                logger.info(f"Category: {category}, Tags: {tags}")
            except Exception as e:
                logger.error(f"Classification failed: {e}")

        # Stage 3: Text generation (mT5 / rugpt3)
        if self._text_generator:
            try:
                title = self._text_generator.generate_title(caption, category)
                description = self._text_generator.generate_description(caption, category, title)
                characteristics = self._text_generator.generate_characteristics(caption, category)
                logger.info(f"Generated title: {title}")
            except Exception as e:
                logger.error(f"Text generation failed: {e}")
                # Use fallback
                title = self._text_generator._fallback_title(caption, category)
                description = self._text_generator._fallback_description(caption, category)
                characteristics = self._text_generator._infer_characteristics(caption, category)
        else:
            # No text generator — use rule-based fallbacks
            characteristics = {"Категория": category}
            if caption:
                title = f"Товар — {caption[:50]}"
                description = f"Товар из категории «{category}». На изображении: {caption}."
            else:
                title = f"Товар из категории «{category}»"
                description = f"Товар из категории «{category}». Высокое качество."

        # Stage 4: SEO generation (rule-based)
        seo = {"seo_title": "", "seo_description": "", "seo_keywords": ""}
        if self._seo_generator:
            try:
                seo = self._seo_generator.generate(title, description, category, tags)
            except Exception as e:
                logger.error(f"SEO generation failed: {e}")

        return {
            "title": title,
            "description": description,
            "category": category,
            "characteristics": characteristics,
            "tags": tags,
            "caption": caption,
            "confidence_score": round(confidence, 3),
            **seo,
        }
