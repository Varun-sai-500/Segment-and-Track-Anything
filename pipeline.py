import os
import tempfile
import zipfile
import cv2
import numpy as np
import threading
import torch
from contextlib import contextmanager

from inference.sam_segmentor import Segmentor
from inference.dino_detector import Detector
from inference.deaot_tracker import Tracker
from mask_utils import draw_mask


class Pipeline:
    def __init__(self, sam_args, dino_args, deaot_args):
        self.device = sam_args["device"]

        self.segmentor = Segmentor(sam_args)
        self.tracker = Tracker(deaot_args)
        self.detector = Detector(dino_args)

        self.current_mask = None
        self.curr_idx = 1

        self._stop_event = threading.Event()

        print(f"Pipeline initialized successfully on {self.device}.")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def stop_tracking(self):
        self._stop_event.set()

    def get_current_mask(self):
        if self.current_mask is None:
            return None

        return self.current_mask.clone()

    def set_current_mask(self, mask):
        if mask is None:
            self.current_mask = None
            return

        mask = self._mask_to_device(mask)

        if mask.ndim != 2:
            raise ValueError("Current mask must be a 2D label mask.")

        self.current_mask = mask.clone()

        obj_num = self.get_obj_num()
        self.curr_idx = max(self.curr_idx, obj_num + 1)

    # ------------------------------------------------------------------
    # Device boundaries
    # ------------------------------------------------------------------

    def _frame_to_device(self, frame):
        if isinstance(frame, torch.Tensor):
            if frame.device.type != self.device:
                return frame.to(self.device, non_blocking=True)
            return frame

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array or Torch tensor.")

        return torch.from_numpy(frame).to(self.device, non_blocking=True)

    def _mask_to_device(self, mask):
        if isinstance(mask, torch.Tensor):
            return mask.to(self.device, dtype=torch.uint8, non_blocking=True)

        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array or Torch tensor.")

        return torch.as_tensor(mask, dtype=torch.uint8, device=self.device)

    def _tensor_to_numpy(self, tensor):
        return tensor.cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------
    # Object bookkeeping
    # ------------------------------------------------------------------

    def get_tracking_objs(self):
        if self.current_mask is None:
            return []

        objs = torch.unique(self.current_mask)
        objs = objs[objs != 0]

        return objs.tolist()

    def get_obj_num(self):
        if self.current_mask is None:
            return 0

        return int(torch.max(self.current_mask).item())

    # ------------------------------------------------------------------
    # DeAOT reference / tracking
    # ------------------------------------------------------------------

    def initialize_reference(self, frame, mask, frame_step=0):
        if mask is None:
            raise ValueError("Cannot initialize tracker without a mask.")

        frame = self._frame_to_device(frame)
        mask = self._mask_to_device(mask)

        if mask.ndim != 2:
            raise ValueError("Reference mask must be a 2D label mask.")

        obj_num = int(torch.max(mask).item())

        self.tracker.initialize(frame, mask, obj_num, frame_step=frame_step)

        self.current_mask = mask.clone()
        self.curr_idx = max(self.curr_idx, obj_num + 1)

    def add_objects(self, mask, frame_step=0):
        if mask is None:
            raise ValueError("Cannot add objects without a mask.")

        if self.current_mask is None:
            raise RuntimeError("Cannot add objects before the tracker is initialized.")

        mask = self._mask_to_device(mask)

        new_obj_num = int(max(torch.max(self.current_mask).item(), torch.max(mask).item()))

        merged_mask = self.current_mask.clone()

        new_pixels = mask > 0
        merged_mask[new_pixels] = mask[new_pixels]

        self.tracker.add_objects(merged_mask, new_obj_num, frame_step)

        self.current_mask = merged_mask
        self.curr_idx = max(self.curr_idx, new_obj_num + 1)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _format_pred_mask(self, mask):
        return mask.squeeze(0).squeeze(0).to(torch.uint8)

    def track(self, frame):
        pred_mask = self.tracker.track(frame)

        self.current_mask = self._format_pred_mask(pred_mask)

        return self.current_mask

    def track_and_update(self, frame):
        pred_mask = self.tracker.track_and_update(frame)

        self.current_mask = self._format_pred_mask(pred_mask)

        return self.current_mask

    def update_memory(self, mask, skip_long_term_update=False):
        if mask is None:
            raise ValueError("Cannot update memory without a mask.")

        mask = self._mask_to_device(mask)

        if mask.ndim != 2:
            raise ValueError("Memory mask must be a 2D label mask.")

        self.tracker.update_memory(mask, skip_long_term_update=skip_long_term_update)

        self.current_mask = mask.clone()

        obj_num = self.get_obj_num()
        self.curr_idx = max(self.curr_idx, obj_num + 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def restart_tracker(self):
        self.stop_tracking()

        self.tracker.restart()

        self.current_mask = None
        self.curr_idx = 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, frame, mask):
        if not isinstance(frame, torch.Tensor):
            raise TypeError("frame must be a torch.Tensor")

        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")

        if frame.device.type != self.device:
            raise RuntimeError(f"frame is on {frame.device}, expected {self.device}")

        if mask.device != frame.device:
            raise RuntimeError(f"mask is on {mask.device}, expected {frame.device}")

        # GPU-native.
        # Do NOT convert to NumPy here.
        return draw_mask(frame, mask)

    # ------------------------------------------------------------------
    # Interactive segmentation
    # ------------------------------------------------------------------

    def seg_acc_click(self, origin_frame, coords_groups, modes_groups):
        interactive_masks = self.segmentor.segment_points_multi(origin_frame, coords_groups, modes_groups)

        if not interactive_masks:
            return (self.get_current_mask(), origin_frame)

        if self.current_mask is None:
            mask_shape = interactive_masks[0].shape

            self.current_mask = torch.zeros(mask_shape, dtype=torch.uint8, device=self.device)

        refined_mask = self.current_mask.clone()

        for interactive_mask in interactive_masks:
            interactive_mask = self._mask_to_device(interactive_mask)

            if not torch.any(interactive_mask):
                continue

            refined_mask[interactive_mask > 0] = self.curr_idx
            self.curr_idx += 1

        self.current_mask = refined_mask

        frame_tensor = self._frame_to_device(origin_frame)

        masked_frame = self.render(frame_tensor, refined_mask)

        # Gradio boundary: CUDA -> CPU NumPy.
        masked_frame_np = self._tensor_to_numpy(masked_frame)

        return refined_mask, masked_frame_np

    # ------------------------------------------------------------------
    # Detection + segmentation
    # ------------------------------------------------------------------

    def detect_and_seg(self, origin_frame, grounding_caption, box_threshold, text_threshold, box_size_threshold=1.0):
        boxes = self.detector.detect(origin_frame, grounding_caption, box_threshold, text_threshold)

        if self.current_mask is None:
            self.current_mask = torch.zeros(origin_frame.shape[:2], dtype=torch.uint8, device=self.device)

        refined_mask = self.current_mask.clone()

        frame_area = origin_frame.shape[0] * origin_frame.shape[1]

        for bbox in boxes:
            x0, y0, x1, y1 = bbox

            bbox_area = (x1 - x0) * (y1 - y0)

            if bbox_area > (frame_area * box_size_threshold):
                continue

            interactive_mask = self.segmentor.segment_box(origin_frame, bbox)

            interactive_mask = self._mask_to_device(interactive_mask)

            if not torch.any(interactive_mask):
                continue

            refined_mask[interactive_mask > 0] = self.curr_idx
            self.curr_idx += 1

        self.current_mask = refined_mask

        frame_tensor = self._frame_to_device(origin_frame)

        masked_frame = self.render(frame_tensor, refined_mask)

        # Gradio boundary: CUDA -> CPU NumPy.
        masked_frame_np = self._tensor_to_numpy(masked_frame)

        return refined_mask, masked_frame_np

    # ------------------------------------------------------------------
    # Video tracking generator
    # ------------------------------------------------------------------

    def track_video_sequence(self, input_video=None, input_img_seq=None, fps=30.0, frame_num=0):
        self._stop_event.clear()

        if self.current_mask is None:
            raise RuntimeError("No initial mask found.")

        with self._get_frame_source(input_video, input_img_seq, fps, frame_num) as (frames, width, height, source_fps):
            if frames is None:
                return

            first_frame = True

            for idx, frame_bgr in enumerate(frames):
                curr_frame_idx = frame_num + idx

                if self._stop_event.is_set():
                    break

                # CPU input boundary.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Single CPU -> GPU transfer.
                frame_tensor = self._frame_to_device(frame_rgb)

                if first_frame:
                    pred_mask_tensor = self.get_current_mask()
                    first_frame = False
                else:
                    pred_mask_tensor = self.track_and_update(frame_tensor)

                # GPU-native rendering.
                rendered = self.render(frame_tensor, pred_mask_tensor)

                # Output boundary.
                masked_frame_np = self._tensor_to_numpy(rendered)

                yield (masked_frame_np, curr_frame_idx)

    # ------------------------------------------------------------------
    # Frame source
    # ------------------------------------------------------------------

    @contextmanager
    def _get_frame_source(self, input_video, input_img_seq, default_fps, frame_num):
        if input_video is not None:
            cap = cv2.VideoCapture(input_video)

            if not cap.isOpened():
                cap.release()
                yield (None, None, None, None)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if frame_num > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            def frames():
                while not self._stop_event.is_set():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    yield frame

            try:
                yield (frames(), width, height, fps)
            finally:
                cap.release()

            return

        if input_img_seq is None:
            yield (None, None, None, None)
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(input_img_seq) as zf:
                zf.extractall(temp_dir)

            valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

            image_paths = sorted(
                [
                    os.path.join(root, filename)
                    for root, _, files in os.walk(temp_dir)
                    for filename in files
                    if os.path.splitext(filename)[1].lower() in valid_extensions
                ]
            )

            image_paths = image_paths[frame_num:]

            if not image_paths:
                yield (None, None, None, None)
                return

            first_frame = cv2.imread(image_paths[0])

            if first_frame is None:
                yield (None, None, None, None)
                return

            height, width = first_frame.shape[:2]

            def frames():
                for image_path in image_paths:
                    if self._stop_event.is_set():
                        break

                    frame = cv2.imread(image_path)

                    if frame is not None:
                        yield frame

            yield (frames(), width, height, default_fps)
