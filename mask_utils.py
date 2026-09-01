import cv2
import numpy as np

np.random.seed(200)

_palette = [0, 0, 0] + (
    (np.random.random(3 * 255) * 0.7 + 0.3) * 255
).astype(np.uint8).tolist()

_PALETTE = np.asarray(
    _palette,
    dtype=np.uint8,
).reshape(-1, 3)


def draw_mask(img, mask, alpha=0.5):
    binary_mask = mask != 0

    if not np.any(binary_mask):
        return img

    colors = _PALETTE[mask]

    inv_alpha = 1.0 - alpha

    pixels = img[binary_mask].astype(np.float32)
    colors = colors[binary_mask].astype(np.float32)

    img[binary_mask] = (
        pixels * inv_alpha + colors * alpha
    ).astype(img.dtype)

    return img


def draw_outline(mask, frame):
    binary_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        frame,
        contours,
        -1,
        (255, 0, 0),
        2,
    )

    return frame