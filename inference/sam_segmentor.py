import torch
from contextlib import nullcontext
from transformers import SamModel, SamProcessor

def _autocast_context(device):
    if torch.cuda.is_available():
        if (torch.cuda.is_bf16_supported()):
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    elif device.startswith("mps") and torch.backends.mps.is_available():
        return torch.amp.autocast(
            device_type="mps",
            dtype=torch.float16,
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

            self.processor = SamProcessor.from_pretrained(self.model_id)
            self.model = (SamModel.from_pretrained(self.model_id).to(self.device))
            self.model.eval()
            
    def _move_inputs(self, inputs):
        metadata_keys = {
            "original_sizes",
            "reshaped_input_sizes",
        }

        for key, value in inputs.items():
            if torch.is_tensor(value) and key not in metadata_keys:
                inputs[key] = value.to(self.device)

        return inputs


    def _post_process(self, outputs, inputs):
        return self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
    @torch.inference_mode()
    def segment_points_multi(self, origin_frame, coords_groups, modes_groups):
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
                multimask_output=False,
            )

        masks = self._post_process(outputs, inputs)[0]
        interactive_masks = [
            masks[i, 0]
            for i in range(len(coords_groups))
        ]

        return interactive_masks

    @torch.inference_mode()
    def segment_box(self, origin_frame, bbox):
        """
        Segment an object using a bounding-box prompt.

        bbox:
            [x0, y0, x1, y1]

        Returns:
            mask: (H, W) uint8
        """
        self._load_model()

        x0, y0, x1, y1 = bbox

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
                multimask_output=False,
            )

        masks = self._post_process(
            outputs,
            inputs,
        )[0]

        return masks[0, 0]