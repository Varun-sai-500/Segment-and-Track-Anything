# Explanation of generator_args is in sam/segment_anything/automatic_mask_generator.py: SamAutomaticMaskGenerator
import glob 
import os

def find_sam_checkpoint(folder="ckpt"):
    files = glob.glob(os.path.join(folder, "sam_*.pth"))
    if not files:
        raise FileNotFoundError("No SAM checkpoint found inside ./ckpt")
    return files[0]  

def infer_sam_type(path):
    if "vit_h" in path:
        return "vit_h"
    if "vit_l" in path:
        return "vit_l"
    if "vit_b" in path:
        return "vit_b"
    raise ValueError("Unknown SAM model type in filename")

def find_aot_checkpoint(folder="ckpt"):
    files = glob.glob(os.path.join(folder, "*.pth"))

    valid = []
    for f in files:
        name = os.path.basename(f).lower()

        if "pre_ytb_dav" not in name:
            continue

        if "aot" in name:  # covers aot + deaot
            valid.append(f)

    if not valid:
        raise FileNotFoundError(
            "No valid PRE_YTB_DAV AOT/DeAOT checkpoint found in ./ckpt"
        )

    return valid[0]

def infer_aot_model(path):
    name = os.path.basename(path).lower()
    prefix = "deaot" if "deaot" in name else "aot"
    # detect size tier
    if "aott" in name:
        size = "t"
    elif "aots" in name:
        size = "s"
    elif "aotb" in name:
        size = "b"
    elif "aotl" in name:
        size = "l"
    else:
        size = "l" 

    # detect backbone
    if "r50" in name:
        backbone = "r50"
    elif "swin" in name:
        backbone = "swinb"
    else:
        backbone = "r50" 
    return f"{backbone}_{prefix}{size}"

sam_ckpt = find_sam_checkpoint()
aot_ckpt = find_aot_checkpoint()

sam_args = {
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
    'gpu_id': 0,
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    "model": infer_aot_model(aot_ckpt),
    "model_path": aot_ckpt,
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    'gpu_id': 0,
}
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
}