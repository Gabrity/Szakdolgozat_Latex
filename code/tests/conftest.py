"""Shared fixtures for all test levels (unit / integration / gui).

Design notes
------------
- ``tiny_model`` stands in for the real ResNet50/MobileNetV2 so attack-engine
  tests run in milliseconds and never touch the network or the filesystem
  cache. Swapping the real backbone for a tiny ``nn.Linear`` is the
  Python/PyTorch equivalent of injecting a test double.
- Fixtures that need randomness call ``torch.manual_seed`` themselves right
  before generating data, so fixture *order* never affects reproducibility.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


class _TinyLogitsModel(nn.Module):
    """Minimal classifier: flattens a small image and applies one Linear layer."""

    def __init__(self, num_classes: int = 5, input_hw: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * input_hw * input_hw, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def tiny_model(device: torch.device) -> nn.Module:
    torch.manual_seed(0)
    model = _TinyLogitsModel().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture()
def attack_ctx(tiny_model: nn.Module, device: torch.device):
    from src.attack_engine import AttackContext

    mean = torch.zeros(1, 3, 1, 1, device=device)
    std = torch.ones(1, 3, 1, 1, device=device)
    return AttackContext(model=tiny_model, mean=mean, std=std)


@pytest.fixture()
def sample_image_tensor(device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, 8, 8, device=device)


@pytest.fixture()
def sample_png(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 200, 50)).save(path)
    return path
