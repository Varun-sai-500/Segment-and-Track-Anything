import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")

if not os.path.isdir(CKPT_DIR):
    raise RuntimeError(f"Checkpoint folder not found: {CKPT_DIR}")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

sam_args = {
    "model_id": "facebook/sam-vit-base",
    "device": device,
    "generator_args": {
        "points_per_crop": 16,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
        "crops_n_layers": 0,
    },
}

dino_args = {
    "model_id": "IDEA-Research/grounding-dino-base",
    "device": device,
}

ast_args = {
    "model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "device": device,
}

deaot_args = {
    "model_path": os.path.join(CKPT_DIR, "R50_DeAOTL_PRE_YTB_DAV.pth"),
    "device": device,
}

segtracker_args = {
    "sam_gap": 10,
    "min_area": 200,
    "max_obj_num": 255,
    "min_new_obj_iou": 0.8,
}