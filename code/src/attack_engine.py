"""FGSM and PGD adversarial attacks operating in pixel space [0, 1].

The model expects normalized input, so the attack normalizes internally and
keeps the adversarial example in raw pixel space — this makes the
``epsilon`` budget intuitive (e.g. ``8/255``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AttackResult:
    """Container for the output of an attack."""

    adversarial: torch.Tensor
    perturbation: torch.Tensor
    iterations_run: int


@dataclass
class AttackContext:
    """Model and normalization data needed to run an attack. Pure data, no behavior."""

    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.mean.device


def _forward(context: AttackContext, x: torch.Tensor) -> torch.Tensor:
    """Normalize and run a forward pass returning unnormalized_scores."""
    x_norm = (x - context.mean) / context.std
    return context.model(x_norm)


@torch.no_grad()
def predict(context: AttackContext, x_pixels: torch.Tensor, topk: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k probabilities and indices."""
    unnormalized_scores = _forward(context, x_pixels)
    probs = F.softmax(unnormalized_scores, dim=1)
    return probs.topk(topk, dim=1)


def attack(
    context: AttackContext,
    image_tensor: torch.Tensor,
    label: int,
    method: str,
    epsilon: float,
    iterations: int,
    target_label: Optional[int] = None,
    step_size: Optional[float] = None,
) -> AttackResult:
    """Run the requested attack.

    Parameters
    ----------
    ctx : AttackContext
        Model and normalization data to attack against.
    image_tensor : torch.Tensor
        Image in pixel space, shape (1, 3, H, W), values in [0, 1].
    label : int
        Ground truth (or current top-1) class index.
    method : {"FGSM", "PGD"}
        Case-sensitive; must match exactly.
    epsilon : float
        L_inf perturbation budget in pixel space.
    iterations : int
        Number of PGD steps. Ignored for FGSM.
    target_label : int, optional
        If given, performs a targeted attack (minimize loss w.r.t. target).
    step_size : float, optional
        PGD step size. Defaults to ``epsilon / max(1, iterations) * 2.5``.
    """

    if method not in ("FGSM", "PGD"):
        raise ValueError(f"Unknown attack method '{method}'.")

    x = image_tensor.detach().to(context.device).clamp(0.0, 1.0)
    targeted = target_label is not None
    y = torch.tensor(
        [target_label if targeted else label],
        dtype=torch.long,
        device=context.device,
    )

    if method == "FGSM":
        adversarial = _fgsm(context, x, y, epsilon, targeted)
        iters_run = 1
    else:
        if step_size is None:
            step_size = (epsilon / max(1, iterations)) * 2.5
        adversarial = _pgd(context, x, y, epsilon, iterations, step_size, targeted)
        iters_run = iterations

    perturbation = (adversarial - x).detach()
    return AttackResult(
        adversarial=adversarial.detach(),
        perturbation=perturbation,
        iterations_run=iters_run,
    )


def _fgsm(
    ctx: AttackContext,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    targeted: bool,
) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    unnormalized_scores = _forward(ctx, x_adv)
    loss = F.cross_entropy(unnormalized_scores , y) # softmax majd -log(p_k)

    grad = torch.autograd.grad(loss, x_adv)[0] # minden pixelre, hogy mennyivel változna a loss
    sign = grad.sign()
    if targeted:
        x_adv = x_adv - epsilon * sign
    else:
        x_adv = x_adv + epsilon * sign

    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _pgd(
    ctx: AttackContext,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    iterations: int,
    step_size: float,
    targeted: bool,
) -> torch.Tensor:
    x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(max(1, iterations)):
        x_adv.requires_grad_(True)
        unnormalized_scores = _forward(ctx, x_adv)
        loss = F.cross_entropy(unnormalized_scores, y)
        grad = torch.autograd.grad(loss, x_adv)[0] 

        with torch.no_grad():
            if targeted:
                x_adv = x_adv - step_size * grad.sign()
            else:
                x_adv = x_adv + step_size * grad.sign()

            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv
