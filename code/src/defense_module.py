from __future__ import annotations

import io

import torch
from PIL import Image, ImageFilter
from torchvision import transforms


class DefenseModule:

    def __init__(self) -> None:
        self._to_tensor = transforms.ToTensor()

    @staticmethod
    def _tensor_to_pil(x: torch.Tensor) -> Image.Image: # RGB-t csinál
        x = x.detach().clamp(0.0, 1.0).cpu().squeeze(0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    def _pil_to_tensor(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        return self._to_tensor(img).unsqueeze(0).to(device)

    def gaussian(self, x: torch.Tensor) -> torch.Tensor:
        img = self._tensor_to_pil(x)
        radius = 1.5
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self._pil_to_tensor(blurred, x.device)

    def jpeg(self, x: torch.Tensor) -> torch.Tensor:
        img = self._tensor_to_pil(x)
        buf = io.BytesIO()
        quality = 75
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        decoded = Image.open(buf).convert("RGB")
        return self._pil_to_tensor(decoded, x.device)

    def apply(self, x: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
        if method == "gaussian":
            return self.gaussian(x, **kwargs)
        if method == "jpeg":
            return self.jpeg(x, **kwargs)
        raise ValueError(f"Unknown defense '{method}'.")
