import logging
from pathlib import Path

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """BLIP-based image captioner with optional LoRA adapter support."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cpu",
        lora_path: str | None = None,
    ):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # Apply LoRA adapter if path is provided and exists
        if lora_path and Path(lora_path).is_dir():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_path).to(device)
            logger.info("LoRA adapter loaded from %s", lora_path)

        self.model.eval()

    def caption(self, image_path: str) -> tuple[str, float]:
        """Generate a caption for the image.

        Returns:
            Tuple of (caption_text, confidence_score)
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            output_scores=True,
            return_dict_in_generate=True,
        )

        caption = self.processor.decode(output.sequences[0], skip_special_tokens=True)

        # Approximate confidence from sequence scores
        if hasattr(output, "sequences_scores") and output.sequences_scores is not None:
            confidence = float(output.sequences_scores[0].exp().item())
        else:
            confidence = 0.8  # default when scores unavailable

        confidence = min(max(confidence, 0.0), 1.0)
        return caption.strip(), confidence
