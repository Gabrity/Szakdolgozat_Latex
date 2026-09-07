"""Lookup table of human-readable ImageNet class labels.

We rely on the metadata bundled with the torchvision pretrained weights so
no extra files have to be shipped with the project.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def get_imagenet_classes() -> List[str]:
    """Return a list of 1000 human-readable ImageNet class names."""
    # Importing lazily so the GUI can start even if torchvision is slow to load.
    from torchvision.models import ResNet50_Weights

    categories = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    if len(categories) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet categories, got {len(categories)}."
        )
    return categories
