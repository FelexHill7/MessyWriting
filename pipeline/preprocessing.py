"""
preprocessing.py — Image transforms and GPU batch augmentation.

Single source of truth for all image preprocessing:
  - base_transform    — deterministic transform applied at dataset init & inference
  - tta_transforms    — test-time augmentation variants for inference
  - gpu_augment()     — batch-parallel GPU augmentation for training
"""

import math
import torch
from torchvision import transforms

from config import IMG_HEIGHT, IMG_WIDTH


# ── Base transform (used by all datasets and inference) ───────────────────────
base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# ── Test-time augmentation transforms ────────────────────────────────────────
# Light augmentations: slightly different perspectives of the same image
# to reduce prediction variance when averaged.
tta_transforms = [
    # Original (identity)
    base_transform,
    # Slight rotation left
    transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=(-3, -3)),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight rotation right
    transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=(3, 3)),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight scale up
    transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((int(IMG_HEIGHT * 1.1), int(IMG_WIDTH * 1.1))),
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight scale down
    transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((int(IMG_HEIGHT * 0.9), int(IMG_WIDTH * 0.9))),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
]


# ── GPU batch augmentation ────────────────────────────────────────────────────
@torch.no_grad()
def gpu_augment(images):
    """Apply random augmentation to an entire batch on GPU in one shot.

    Replaces per-sample CPU transforms with batch-parallel GPU operations:
    - Random affine (rotation, translation, shear)
    - Elastic distortion (simulates handwriting deformation)
    - Random brightness / contrast
    - Gaussian blur
    - Morphological erosion / dilation (pen stroke thickness variation)

    Input/output: (B, 1, 32, 128) tensors in [-1, 1] range.
    """
    B, C, H, W = images.shape
    device = images.device

    # ── Random affine: rotation (±5°) + translation (±5%) + shear (±5°) ───
    angles = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    shear  = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    tx     = (torch.rand(B, device=device) * 2 - 1) * 0.05
    ty     = (torch.rand(B, device=device) * 2 - 1) * 0.05

    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_a + shear * sin_a
    theta[:, 0, 1] = -sin_a + shear * cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=False)
    images = torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")

    # ── Elastic distortion (50% chance per batch) ─────────────────────────
    if torch.rand(1).item() > 0.5:
        alpha = 3.0
        sigma = 4.0
        dx = torch.rand(B, 1, H, W, device=device) * 2 - 1
        dy = torch.rand(B, 1, H, W, device=device) * 2 - 1
        ek = 7
        eax = torch.arange(ek, dtype=torch.float32, device=device) - ek // 2
        ekernel = torch.exp(-0.5 * (eax / sigma) ** 2)
        ekernel = ekernel / ekernel.sum()
        ek2d = (ekernel[:, None] * ekernel[None, :]).view(1, 1, ek, ek)
        dx = torch.nn.functional.conv2d(dx, ek2d, padding=ek // 2) * alpha
        dy = torch.nn.functional.conv2d(dy, ek2d, padding=ek // 2) * alpha
        dx = dx.squeeze(1) / (W / 2)
        dy = dy.squeeze(1) / (H / 2)
        base_grid = torch.nn.functional.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
            images.shape, align_corners=False
        )
        base_grid[..., 0] += dx
        base_grid[..., 1] += dy
        images = torch.nn.functional.grid_sample(images, base_grid, align_corners=False, padding_mode="border")

    # ── Random brightness (±30%) and contrast (±30%) ─────────────────────
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
    contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
    mean = images.mean(dim=(2, 3), keepdim=True)
    images = contrast * (images - mean) + mean + (brightness - 1.0)

    # ── Gaussian blur (kernel_size=3, random sigma 0.1–1.0) ──────────────
    sigma = 0.1 + torch.rand(1, device=device).item() * 0.9
    k = 3
    ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)
    images = torch.nn.functional.conv2d(images, kernel_2d, padding=k // 2, groups=1)

    # ── Morphological erosion / dilation (30% chance each) ────────────────
    morph_roll = torch.rand(1).item()
    if morph_roll < 0.3:
        images = -torch.nn.functional.max_pool2d(-images, kernel_size=3, stride=1, padding=1)
    elif morph_roll < 0.6:
        images = torch.nn.functional.max_pool2d(images, kernel_size=3, stride=1, padding=1)

    return images.clamp(-1, 1)
