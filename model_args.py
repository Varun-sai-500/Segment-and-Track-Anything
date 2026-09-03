import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

sam_args = {
<<<<<<< HEAD
    "model_id": "facebook/sam-vit-base",
    "device": device,
}

dino_args = {
    "model_id": "IDEA-Research/grounding-dino-base",
    "device": device,
}

deaot_args = {
    "repo_id": "Varun-Sai-500/DeAOT",
    "model_filename": "R50_DeAOT-L_inference.pth",
    "device": device,
    "long_term_mem_gap": 9999,
    "short_term_mem_skip": 1,
    "max_len_long_term": 9999,
=======
    'sam_checkpoint': sam_ckpt,
    'model_type': infer_sam_type(sam_ckpt),
    'generator_args':{
        'points_per_side': 16,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    "device": "cuda",
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    "model": infer_deaot_model(deaot_ckpt),
    "model_path": deaot_ckpt,
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    "device": "cuda",
}
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80%
>>>>>>> c4992261d0d1c6e0a6e3f2c0eec9e65c78474987
}