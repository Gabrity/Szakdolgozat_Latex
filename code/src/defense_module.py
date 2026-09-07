"""Simple input-sanitization defenses against adversarial perturbations."""
from __future__ import annotations

import io

import torch
from PIL import Image, ImageFilter
from torchvision import transforms


class DefenseModule:
    """Apply low-pass / lossy transforms that tend to remove high-frequency
    adversarial noise."""

    def __init__(self) -> None:
        self._to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------ #
    # Pixel-space tensor in/out helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
        """Convert (1,3,H,W) [0,1] tensor to a PIL RGB image."""
        x = x.detach().clamp(0.0, 1.0).cpu().squeeze(0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    def _pil_to_tensor(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        return self._to_tensor(img).unsqueeze(0).to(device)

    # ------------------------------------------------------------------ #
    # Defenses
    # ------------------------------------------------------------------ #
    def gaussian_blur(self, x: torch.Tensor, radius: float = 1.5) -> torch.Tensor:
        img = self._tensor_to_pil(x)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self._pil_to_tensor(blurred, x.device)

    def jpeg_compression(self, x: torch.Tensor, quality: int = 75) -> torch.Tensor:
        img = self._tensor_to_pil(x)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        decoded = Image.open(buf).convert("RGB")
        return self._pil_to_tensor(decoded, x.device)

    def apply(self, x: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
        method = method.lower()
        if method in ("gaussian", "blur", "gaussian_blur"):
            return self.gaussian_blur(x, **kwargs)
        if method in ("jpeg", "jpeg_compression"):
            return self.jpeg_compression(x, **kwargs)
        raise ValueError(f"Unknown defense '{method}'.")
