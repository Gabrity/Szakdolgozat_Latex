from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn
from torchvision import models, transforms


# ImageNet normalizaciós konstansok
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DEFAULT_CACHE_DIR = "./models"


@dataclass
class ModelBundle:

    name: str
    model: nn.Module
    preprocess: Callable
    mean: torch.Tensor
    std: torch.Tensor


class ModelManager:

    SUPPORTED = ("ResNet50", "MobileNetV2") # modell bővítés

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR, device: str | None = None) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", self.cache_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._cache: Dict[str, ModelBundle] = {}

    def get(self, name: str) -> ModelBundle:
        if name not in self._cache:
            self._cache[name] = self._build(name)
        return self._cache[name]

    @property
    def mean_std(self) -> tuple[list[float], list[float]]:
        return _IMAGENET_MEAN, _IMAGENET_STD

    def _build(self, name: str) -> ModelBundle:
        if name == "ResNet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
        elif name == "MobileNetV2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            net = models.mobilenet_v2(weights=weights)
        else:
            raise ValueError(
                f"Unsupported model '{name}'. Supported: {self.SUPPORTED}"
            )

        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        return ModelBundle(
            name=name, model=net, preprocess=preprocess, mean=mean, std=std
        )
