import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from deaot.deaot_engine import DeAOTEngine
from deaot.deaot_model import DeAOT
import deaot.video_transforms as tr


def load_network(net, repo_id, model_filename, device):
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint.get("model", checkpoint),
    )

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    net.load_state_dict(state_dict, strict=True)
    return net


class Tracker:
    def __init__(self, deaot_args):
        self.device = deaot_args["device"]

        self.repo_id = deaot_args["repo_id"]
        self.model_filename = deaot_args["model_filename"]

        self.long_term_mem_gap = (
            deaot_args["long_term_mem_gap"]
        )
        self.short_term_mem_skip = (
            deaot_args["short_term_mem_skip"]
        )
        self.max_len_long_term = (
            deaot_args["max_len_long_term"]
        )

        self.model = None
        self.engine = None

        self.transforms = (
            tr.MultiRestrictSize(),
            tr.MultiToTensor(),
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.model is not None:
            return

        print("Loading DeAOT model")

        self.model = DeAOT().to(self.device)

        load_network(
            self.model,
            self.repo_id,
            self.model_filename,
            self.device,
        )

        self.model.eval()

        self.engine = DeAOTEngine(
            deaot_model=self.model,
            long_term_mem_gap=self.long_term_mem_gap,
            short_term_mem_skip=self.short_term_mem_skip,
            max_len_long_term=self.max_len_long_term,
        )

    def _require_initialized(self):
        self._load_model()

        if self.engine.obj_nums is None:
            raise RuntimeError(
                "Tracker has not been initialized. "
                "Call initialize() first."
            )

    # ------------------------------------------------------------------
    # Object validation
    # ------------------------------------------------------------------

    def _validate_obj_nums(self, obj_nums):
        if isinstance(obj_nums, (list, tuple)):
            if len(obj_nums) != 1:
                raise ValueError(
                    "obj_nums must contain exactly one object count."
                )

            obj_nums = obj_nums[0]

        try:
            obj_nums = int(obj_nums)
        except (TypeError, ValueError):
            raise ValueError(
                "obj_nums must be an integer."
            )

        max_obj_num = self.model.max_obj_num

        if obj_nums < 1:
            raise ValueError(
                "obj_nums must be at least 1."
            )

        if obj_nums > max_obj_num:
            raise ValueError(
                f"DeAOT supports at most {max_obj_num} "
                f"foreground objects, got {obj_nums}."
            )

        return obj_nums

    def _validate_mask_object_ids(self, mask):
        if mask is None:
            return

        mask_tensor = torch.as_tensor(mask)

        if mask_tensor.numel() == 0:
            raise ValueError(
                "Mask is empty."
            )

        max_label = int(mask_tensor.max().item())
    
        if max_label > self.model.max_obj_num:
            raise ValueError(
                f"Mask contains object ID {max_label}, "
                f"but DeAOT supports object IDs up to "
                f"{self.model.max_obj_num}."
            )

        if int(mask_tensor.min().item()) < 0:
            raise ValueError(
                "Mask contains negative object IDs."
            )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _transform(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def _prepare_reference(self, frame, mask):
        self._validate_mask_object_ids(mask)

        sample = self._transform({
            "current_img": frame,
            "current_label": mask,
        })

        frame = (
            sample[0]["current_img"]
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        mask = (
            sample[0]["current_label"]
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        mask = F.interpolate(
            mask,
            size=frame.shape[-2:],
            mode="nearest",
        )

        return frame, mask

    def _prepare_mask(self, mask):
        self._validate_mask_object_ids(mask)

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32,
            device=self.device,
        )

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)

        elif mask.ndim != 4:
            raise ValueError(
                f"Expected mask with 2, 3, or 4 dimensions, "
                f"got shape {tuple(mask.shape)}"
            )

        target_size = self.engine.input_size_2d

        if tuple(mask.shape[-2:]) != tuple(target_size):
            mask = F.interpolate(
                mask,
                size=target_size,
                mode="nearest",
            )

        return mask

    # ------------------------------------------------------------------
    # Tracker initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize(self, frame, mask, obj_nums, frame_step=0):
        
        self._load_model()
        obj_nums = self._validate_obj_nums(obj_nums)
        frame, mask = self._prepare_reference(frame,mask)

        self.engine.add_reference_frame(
            frame,
            mask,
            obj_nums=obj_nums,
            frame_step=frame_step,
        )

    # ------------------------------------------------------------------
    # Incremental object addition
    # ------------------------------------------------------------------
        
    @torch.no_grad()
    def add_objects(self, mask, obj_nums, frame_step):
        self._require_initialized()

        obj_nums = self._validate_obj_nums(
            obj_nums
        )

        if obj_nums <= self.engine.obj_nums:
            raise ValueError(
                f"obj_nums must increase during "
                f"incremental tracking: current="
                f"{self.engine.obj_nums}, new={obj_nums}."
            )

        mask = self._prepare_mask(mask)

        self.engine.add_reference_frame_incremental(
            mask=mask,
            obj_nums=obj_nums,
            frame_step=frame_step,
        )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _track(self, image, update_memory=False):
        self._require_initialized()

        output_size = image.shape[:2]

        sample = self._transform({
            "current_img": image,
        })

        image_tensor = (
            sample[0]["current_img"]
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        self.engine.match_propogate_one_frame(image_tensor)

        logits = self.engine.decode_current_logits(
            output_size=image_tensor.shape[-2:]
        )

        internal_mask = torch.argmax(logits, dim=1, keepdim=True).float()

        if update_memory:
            self.engine.update_memory(internal_mask)

        if tuple(image_tensor.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )

        return torch.argmax(logits, dim=1, keepdim=True).float()

    def track(self, image):
        return self._track(
            image,
            update_memory=False,
        )

    def track_and_update(self, image):
        return self._track(
            image,
            update_memory=True,
        )

    # ------------------------------------------------------------------
    # Explicit memory update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_memory(self, mask, skip_long_term_update=False):
        self._require_initialized()

        mask = self._prepare_mask(mask)

        self.engine.update_memory(
            mask,
            skip_long_term_update=skip_long_term_update,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def restart(self):
        if self.engine is not None:
            self.engine.restart_engine()