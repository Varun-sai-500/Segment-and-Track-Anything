import numpy as np
import torch
from contextlib import nullcontext
from transformers import SamModel, SamProcessor

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


class Segmentor:

    def __init__(self, sam_args):
        self.device = sam_args["device"]
        self.model_id = sam_args["model_id"]

        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading SAM model: {self.model_id}")

            self.processor = SamProcessor.from_pretrained(
                self.model_id
            )

            self.model = (
                SamModel.from_pretrained(
                    self.model_id
                )
                .to(self.device)
            )

            self.model.eval()

    def _move_inputs(self, inputs):
        return {
            key: value.to(self.device)
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

    def _post_process(self, outputs, inputs):
        """
        Move SAM outputs/metadata to CPU once and restore masks
        to their original image dimensions.
        """
        pred_masks = outputs.pred_masks.cpu()
        iou_scores = outputs.iou_scores.cpu()

        original_sizes = inputs[
            "original_sizes"
        ].cpu()

        reshaped_input_sizes = inputs[
            "reshaped_input_sizes"
        ].cpu()

        masks = (
            self.processor
            .image_processor
            .post_process_masks(
                pred_masks,
                original_sizes,
                reshaped_input_sizes,
            )
        )

        return masks, iou_scores

    @torch.inference_mode()
    def segment_points(
        self,
        origin_frame,
        coords,
        modes,
        multimask=False,
    ):
        """
        Segment an object using point prompts.

        coords:
            [[x, y], [x, y], ...]

        modes:
            [1, 1, 0, ...]
            1 = foreground
            0 = background

        Returns:
            mask: (H, W) uint8
        """
        self._load_model()

        coords = np.asarray(coords)
        modes = np.asarray(modes)

        inputs = self.processor(
            images=origin_frame,
            input_points=[[coords.tolist()]],
            input_labels=[[modes.tolist()]],
            return_tensors="pt",
        )

        inputs = self._move_inputs(inputs)

        with _autocast_context(self.device):
            outputs = self.model(
                **inputs,
                multimask_output=multimask,
            )

        masks, scores = self._post_process(
            outputs,
            inputs,
        )

        masks = masks[0][0]
        scores = scores[0, 0]

        best_idx = torch.argmax(scores).item()

        return (
            masks[best_idx]
            .numpy()
            .astype(np.uint8)
        )

    @torch.inference_mode()
    def segment_points_multi(
        self,
        origin_frame,
        coords_groups,
        modes_groups,
        multimask=False,
    ):
        """
        Segment multiple independently prompted objects in one SAM
        forward pass.

        coords_groups:
            [
                [[x, y], [x, y], ...],
                [[x, y], [x, y], ...],
                ...
            ]

        modes_groups:
            [
                [1, 1, 0, ...],
                [1, 1, ...],
                ...
            ]

        Returns:
            list of masks, one mask per object group.
        """
        self._load_model()

        if not coords_groups:
            return []

        if len(coords_groups) != len(modes_groups):
            raise ValueError(
                "coords_groups and modes_groups must have "
                "the same length."
            )

        max_points = max(
            len(coords)
            for coords in coords_groups
        )

        input_points = []
        input_labels = []

        for coords, modes in zip(
            coords_groups,
            modes_groups,
        ):
            if len(coords) != len(modes):
                raise ValueError(
                    "Each coords group must have the same number "
                    "of points as its modes group."
                )

            padded_coords = [
                [float(x), float(y)]
                for x, y in coords
            ]

            padded_labels = [
                int(label)
                for label in modes
            ]

            padding = max_points - len(padded_coords)

            if padding:
                padded_coords.extend(
                    [[0.0, 0.0]] * padding
                )

                padded_labels.extend(
                    [-1] * padding
                )

            input_points.append(padded_coords)
            input_labels.append(padded_labels)

        inputs = self.processor(
            images=origin_frame,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        )

        inputs = self._move_inputs(inputs)

        with _autocast_context(self.device):
            outputs = self.model(
                **inputs,
                multimask_output=multimask,
            )

        masks, scores = self._post_process(
            outputs,
            inputs,
        )

        masks = masks[0]
        scores = scores[0]

        interactive_masks = []

        for i in range(len(coords_groups)):
            best_idx = torch.argmax(
                scores[i]
            ).item()

            interactive_masks.append(
                masks[i, best_idx]
                .numpy()
                .astype(np.uint8)
            )

        return interactive_masks

    @torch.inference_mode()
    def segment_box(
        self,
        origin_frame,
        bbox,
        multimask=False,
    ):
        """
        Segment an object using a bounding-box prompt.

        bbox:
            [[x0, y0], [x1, y1]]

        Returns:
            mask: (H, W) uint8
        """
        self._load_model()

        x0, y0 = bbox[0]
        x1, y1 = bbox[1]

        inputs = self.processor(
            images=origin_frame,
            input_boxes=[[
                [
                    float(x0),
                    float(y0),
                    float(x1),
                    float(y1),
                ]
            ]],
            return_tensors="pt",
        )

        inputs = self._move_inputs(inputs)

        with _autocast_context(self.device):
            outputs = self.model(
                **inputs,
                multimask_output=multimask,
            )

        masks, scores = self._post_process(
            outputs,
            inputs,
        )

        masks = masks[0][0]
        scores = scores[0, 0]

        best_idx = torch.argmax(scores).item()

        return (
            masks[best_idx]
            .numpy()
            .astype(np.uint8)
        )