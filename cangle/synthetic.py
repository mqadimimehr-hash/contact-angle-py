"""Synthetic sessile-drop images with a known contact angle.

The drop is a spherical cap (2-D: circular segment) resting on a flat,
horizontal substrate. Drop and substrate are dark, background is light,
mimicking a back-lit goniometer photo. Rendering is supersampled so the
edges are anti-aliased like a real camera image.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import cv2

__all__ = ["cap_geometry", "make_drop"]


def cap_geometry(angle_deg: float, radius: float) -> dict:
    """Geometry of a circular cap with sphere radius ``radius``.

    Returns ``base_half_width`` and ``height`` (pixels) plus the vertical
    offset of the circle centre *above* the baseline (``center_height``).
    """
    t = math.radians(angle_deg)
    return {
        "base_half_width": radius * math.sin(t),
        "height": radius * (1.0 - math.cos(t)),
        "center_height": -radius * math.cos(t),
    }


def make_drop(
    angle_deg: float,
    width: int = 640,
    height: int = 480,
    baseline_y: float = 360.0,
    radius: Optional[float] = None,
    center_x: Optional[float] = None,
    drop_level: int = 40,
    background_level: int = 225,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.8,
    supersample: int = 4,
    seed: Optional[int] = 0,
    substrate: str = "dark",
) -> np.ndarray:
    """Render a grayscale (uint8) side view of a sessile drop.

    Parameters
    ----------
    angle_deg : true contact angle, 0 < angle < 180.
    radius : sphere radius in px. If ``None`` it is chosen so the drop
        fills roughly 80 % of the width / 85 % of the space above the baseline.
    substrate : ``"dark"`` (opaque dark band below the baseline), ``"none"``
        (background colour below the baseline) or ``"reflective"`` (light
        substrate showing a mirror image of the drop).
    noise_sigma : std-dev of additive Gaussian noise (grey levels).
    blur_sigma : Gaussian optical blur (px) applied after downsampling.
    """
    if not 0 < angle_deg < 180:
        raise ValueError("angle_deg must be in (0, 180)")
    t = math.radians(angle_deg)
    if radius is None:
        rel_w = 2 * math.sin(t) if angle_deg <= 90 else 2.0
        rel_h = 1 - math.cos(t)
        radius = min(0.8 * width / rel_w, 0.85 * baseline_y / rel_h)
    cx = width / 2.0 if center_x is None else center_x
    cy = baseline_y + radius * math.cos(t)  # image coords, y points down

    s = supersample
    # pixel-centre coordinates of the supersampled grid, in output pixel units
    ys = (np.arange(height * s) + 0.5) / s - 0.5
    xs = (np.arange(width * s) + 0.5) / s - 0.5
    X, Y = np.meshgrid(xs, ys)
    inside = ((X - cx) ** 2 + (Y - cy) ** 2 <= radius**2) & (Y <= baseline_y)
    below = Y > baseline_y
    if substrate == "dark":
        substrate_px = below
    elif substrate == "none":
        substrate_px = np.zeros_like(below)
    elif substrate == "reflective":
        Ym = 2 * baseline_y - Y
        substrate_px = below & ((X - cx) ** 2 + (Ym - cy) ** 2 <= radius**2)
    else:
        raise ValueError("substrate must be 'dark', 'none' or 'reflective'")
    hi = np.where(inside | substrate_px, drop_level, background_level).astype(np.float32)
    img = cv2.resize(hi, (width, height), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
