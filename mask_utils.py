import torch
import torch.nn.functional as F

_palette = None

def _get_palette(device, dtype):
    global _palette
    if _palette is None or _palette.device != device or _palette.dtype != dtype:
        g = torch.Generator().manual_seed(200)
        colors = (torch.rand(255, 3, generator=g) * 0.7 + 0.3) * 255
        colors = colors.to(dtype)
        black = torch.zeros(1, 3, dtype=dtype)
        _palette = torch.cat([black, colors], dim=0).to(device)
    return _palette


@torch.no_grad()
def draw_mask(frame: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5,
              outline_color=(255, 0, 0), outline_thickness: int = 2) -> torch.Tensor:
    """
    Mutates `frame` in place: fills segmented regions with a per-id color
    and draws a colored outline around each region's boundary.

    frame: (H, W, 3) uint8 tensor
    mask:  (H, W) integer tensor, 0 = background, >0 = object id
    """
    device = frame.device
    mask = mask.to(device).long()
    binary_mask = mask != 0

    palette = _get_palette(device, frame.dtype)
    colors = palette[mask].float()
    frame_f = frame.float()

    blended = torch.where(binary_mask.unsqueeze(-1),
                           frame_f * (1 - alpha) + colors * alpha,
                           frame_f)
    frame.copy_(blended.to(frame.dtype))

    # boundary = pixel differs from any 4-neighbor
    m = mask.float().unsqueeze(0).unsqueeze(0)
    padded = F.pad(m, (1, 1, 1, 1), mode='replicate')
    up, down = padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]
    left, right = padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]
    boundary = (m != up) | (m != down) | (m != left) | (m != right)
    boundary = boundary.squeeze(0).squeeze(0) & binary_mask

    if outline_thickness > 1:
        radius = outline_thickness // 2
        kernel = 2 * radius + 1
        b = boundary.float().unsqueeze(0).unsqueeze(0)
        b = F.max_pool2d(b, kernel_size=kernel, stride=1, padding=radius)
        boundary = b.squeeze(0).squeeze(0) > 0

    color_t = torch.tensor(outline_color, dtype=frame.dtype, device=device)
    frame[boundary] = color_t

    return frame