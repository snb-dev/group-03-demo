"""Synthetic dent surfaces with exact height maps (used for testing and for training data)."""
import numpy as np


def grid(n=256, size_mm=200.0):
    """Square measurement patch of size_mm x size_mm sampled on n x n points."""
    x = np.linspace(-size_mm / 2, size_mm / 2, n)
    X, Y = np.meshgrid(x, x)
    return X, Y, x[1] - x[0]


def dent(X, Y, depth=6.0, radius=28.0, sharpness=1.4, cx=0.0, cy=0.0, elong=1.0, angle_deg=0.0):
    """Generalised Gaussian dent. Negative values are below the panel surface (mm).

    sharpness > 1 gives a flatter floor and steeper walls (closer to a crease as it grows).
    elong > 1 stretches the dent along the rotated x axis.
    """
    a = np.deg2rad(angle_deg)
    xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
    r = np.sqrt((xr / elong) ** 2 + yr ** 2)
    return -depth * np.exp(-(r / radius) ** (2 * sharpness))


def panel_waviness(X, Y, amp=0.08):
    """Gentle factory curvature / orange-peel of a real panel."""
    return amp * np.sin(2 * np.pi * X / 140) * np.cos(2 * np.pi * Y / 170)
