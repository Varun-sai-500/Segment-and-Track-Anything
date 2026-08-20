import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class ASTPredictor:
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available. Falling back to MPS.")
                device = "mps"
            else:
                print("CUDA not available. Falling back to CPU.")
                device = "cpu"

        self.device = device
        self.feature_extractor = None
        self.model = None

    def _load_model(self):
        """Lazy-loads HF feature extractor and model on first inference call."""
        if self.model is None:
            print(f"Loading AST model: {self.MODEL_ID}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.MODEL_ID
            ).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(self, wav_path="./audio.flac"):
        import torchaudio

        # Trigger model loading only when inference is actually requested
        self._load_model()

        # 1. Load and resample audio using torchaudio
        waveform, sampling_rate = torchaudio.load(wav_path)

        # AST models expect 1-channel mono at 16kHz
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sampling_rate != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.feature_extractor.sampling_rate,
            )
            waveform = resampler(waveform)
            sampling_rate = self.feature_extractor.sampling_rate

        # Convert tensor to 1D numpy array for HF feature extractor
        speech_array = waveform.squeeze().numpy()

        # 2. Extract features
        inputs = self.feature_extractor(
            speech_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        # 3. Inference
        device_type = torch.device(self.device).type
        with torch.amp.autocast(
            device_type=device_type,
            enabled=(device_type == "cuda"),
        ):
            logits = self.model(input_values).logits
            probabilities = torch.sigmoid(logits).squeeze(0)

        # 4. Extract predictions and top indices
        probs_np = probabilities.cpu().numpy()
        sorted_indexes = np.argsort(probs_np)[::-1]

        top_labels = []
        top_labels_probs = []

        for idx in sorted_indexes:
            label = self.model.config.id2label[idx]
            prob = float(probs_np[idx])

            # Sanitize string formatting
            label = label.replace('"', "").replace("\\", "")

            top_labels.append(label)
            top_labels_probs.append(prob)

        # 5. Maintain strict output contract guarantees
        if len(top_labels) < 10:
            pad = 10 - len(top_labels)
            top_labels += ["unknown"] * pad
            top_labels_probs += [0.0] * pad

        if max(top_labels_probs[:10]) == 0.0:
            top_labels[0] = "silence"
            top_labels_probs[0] = 1.0

        return top_labels[:10], top_labels_probs[:10]
