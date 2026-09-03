import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

sam_args = {
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
}