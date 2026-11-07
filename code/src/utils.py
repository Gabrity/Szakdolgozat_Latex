"""Image / tensor conversion helpers used by the GUI."""
from __future__ import annotations

import re
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms


MODEL_INPUT_SIZE = (224, 224)
_TO_TENSOR = transforms.ToTensor()
_RESIZE_CROP = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224)]
)


def load_image_as_tensor(path: str, device: torch.device) -> Tuple[torch.Tensor, Image.Image]:
    """Load an image from disk, resize/crop to 224x224 and return a pixel-space
    tensor (1, 3, 224, 224) in [0, 1], plus the cropped PIL image.

    Images that are already 224x224 are used as-is, so that a previously
    downloaded original/adversarial image round-trips pixel-exactly."""
    pil = Image.open(path).convert("RGB")
    if pil.size != MODEL_INPUT_SIZE:
        pil = _RESIZE_CROP(pil)
    tensor = _TO_TENSOR(pil).unsqueeze(0).to(device)
    return tensor, pil


def build_adversarial_filename(
    model: str,
    method: str,
    epsilon: float,
    iterations: int,
    target_idx: Optional[int] = None,
    target_name: Optional[str] = None,
) -> str:
    """Build a descriptive file name (without extension) for an adversarial
    image, with the parts separated by underscores. FGSM is single-step, so
    the iteration count is only included for the other (iterative) methods."""
    parts = [model, method, f"eps{epsilon:.3f}"]
    if method.upper() != "FGSM":
        parts.append(f"iter{iterations}")
    if target_idx is None:
        parts.append("untargeted")
    else:
        parts.append("targeted")
        parts.append(str(target_idx))
        if target_name:
            parts.append(re.sub(r"[^A-Za-z0-9]+", "-", target_name).strip("-"))
    return "_".join(parts)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert (1,3,H,W) pixel-space tensor in [0,1] into a PIL image."""
    x = x.detach().clamp(0.0, 1.0).cpu().squeeze(0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def perturbation_to_pil(delta: torch.Tensor, magnify: float = 10.0) -> Image.Image:
    """Visualize a perturbation tensor by centering at 0.5 and scaling."""
    d = delta.detach().cpu().squeeze(0)
    vis = (d * magnify) + 0.5
    vis = vis.clamp(0.0, 1.0)
    arr = (vis.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")
