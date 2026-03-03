from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = "cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
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
