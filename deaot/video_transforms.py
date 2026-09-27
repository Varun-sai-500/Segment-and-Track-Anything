import torch
import torch.nn.functional as F


class MultiRestrictSize:
    def __init__(self, max_long_edge: int = 1040, antialias: bool = False):
        self.max_long_edge = max_long_edge
        self.antialias = antialias

    def __call__(self, sample):
        image = sample["current_img"]  # (H, W, 3)
        h, w = image.shape[:2]
        new_h, new_w = h, w

        long_edge = max(h, w)
        if long_edge > self.max_long_edge:
            scale = self.max_long_edge / long_edge
            new_h = int(h * scale)
            new_w = int(w * scale)

        if (new_h - 1) % 16 != 0:
            new_h = int(round((new_h - 1) / 16) * 16 + 1)
        if (new_w - 1) % 16 != 0:
            new_w = int(round((new_w - 1) / 16) * 16 + 1)

        if new_h != h or new_w != w:
            img_t = image.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
            resized = F.interpolate(img_t, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=self.antialias)
            resized = resized.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
            resized = resized.clamp(0, 255).to(torch.uint8)

            sample = {"current_img": resized, "current_label": sample.get("current_label")}

        return [sample]


class MultiToTensor:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, samples):
        for sample in samples:
            image = sample["current_img"].float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
            image = (image - self.mean) / self.std
            sample["current_img"] = image

            label = sample.get("current_label")
            if label is not None:
                sample["current_label"] = label.unsqueeze(0).int()  # (H,W) -> (1,H,W)

        return samples
