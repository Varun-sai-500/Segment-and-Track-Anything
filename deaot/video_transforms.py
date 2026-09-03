import cv2
import numpy as np
import torch

class MultiRestrictSize:
    def __init__(self):
        self.max_long_edge = 1040

    def __call__(self, sample):
        image = sample["current_img"]

        h, w = image.shape[:2]
        new_h, new_w = h, w

        long_edge = max(h, w)

        if long_edge > self.max_long_edge:
            scale = self.max_long_edge / long_edge
            new_h = int(h * scale)
            new_w = int(w * scale)

        if (new_h - 1) % 16 != 0:
            new_h = int(
                np.around((new_h - 1) / 16) * 16 + 1
            )

        if (new_w - 1) % 16 != 0:
            new_w = int(
                np.around((new_w - 1) / 16) * 16 + 1
            )

        if new_h != h or new_w != w:
            sample = {
                "current_img": cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC,
                ),
                "current_label": sample.get(
                    "current_label"
                ),
            }

        return [sample]


class MultiToTensor:
    def __call__(self, samples):
        for sample in samples:
            image = sample["current_img"]
            image = image.astype(np.float32) / 255.0
            image -= (0.485, 0.456, 0.406)
            image /= (0.229, 0.224, 0.225)

            image = image.transpose(2, 0, 1)
            sample["current_img"] = torch.from_numpy(image)

            if sample.get("current_label") is not None:
                label = sample["current_label"]

                label = label[:, :, np.newaxis]
                label = label.transpose(2, 0, 1)

                sample["current_label"] = (
                    torch.from_numpy(label).int()
                )

        return samples