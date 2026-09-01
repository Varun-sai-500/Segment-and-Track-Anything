import numpy as np
import torch
from contextlib import nullcontext
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

def _autocast_context(device):
    if device.startswith("cuda"):
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

    elif device.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.amp.autocast(
                device_type="mps",
                dtype=torch.bfloat16,
            )

    return nullcontext()

class Detector:
    def __init__(self, dino_args):
        self.device = dino_args["device"]
        self.model_id = dino_args["model_id"]

        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading GroundingDINO model: {self.model_id}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_id
            )

            self.model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(self.model_id)
                .to(self.device)
            )

            self.model.eval()

    @staticmethod
    def normalize_caption(text: str):
        parts = [
            part.strip().lower()
            for part in text.split(".")
            if part.strip()
        ]
        return ". ".join(parts) + "."
    
    @torch.inference_mode()
    def detect(
        self,
        origin_frame,
        grounding_caption,
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        frame = np.asarray(origin_frame)

        self._load_model()

        grounding_caption = self.normalize_caption(
            grounding_caption
        )

        inputs = self.processor(
            images=frame,
            text=grounding_caption,
            return_tensors="pt",
        ).to(self.device)

        with _autocast_context(self.device):
            outputs = self.model(**inputs)

        results = (
            self.processor
            .post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[frame.shape[:2]],
            )[0]
        )

        boxes = (
            results["boxes"]
            .to(dtype=torch.int32, device="cpu")
            .numpy()
        )

        return boxes