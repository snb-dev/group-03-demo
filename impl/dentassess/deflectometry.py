"""Phase-measuring deflectometry: stripe images -> phase -> slope -> height map.

Model: a screen at distance d shows sinusoidal stripes of period p. A surface slope s
shifts the reflected stripe phase by approximately (4*pi*d/p)*s (the reflection doubles
the angle). Two stripe periods are used: a coarse one that never wraps, and a fine one
for precision, combined by temporal (hierarchical) phase unwrapping.
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def phase_gain(distance_mm, pitch_mm):
    """Phase change per unit slope: k = 4*pi*d/p."""
    return 4 * np.pi * distance_mm / pitch_mm


def simulate_fringes(Z, dx, distance_mm, pitch_mm, axis=1, noise=0.02, rng=None, steps=4):
    """Simulate the `steps` phase-shifted stripe images reflected by surface Z (one direction)."""
    rng = np.random.default_rng(0) if rng is None else rng
    gy, gx = np.gradient(Z, dx)
    slope = gx if axis == 1 else gy
    n = Z.shape[1] if axis == 1 else Z.shape[0]
    coord = np.arange(n) * dx
    carrier = 2 * np.pi * coord / pitch_mm
    carrier = carrier[None, :] if axis == 1 else carrier[:, None]
    phi = carrier + phase_gain(distance_mm, pitch_mm) * slope
    imgs = [0.5 + 0.5 * np.cos(phi + j * 2 * np.pi / steps) + rng.normal(0, noise, Z.shape) for j in range(steps)]
    return np.array(imgs), np.broadcast_to(carrier, Z.shape)


def wrapped_phase(imgs):
    """N-step phase-shifting algorithm (4-step: atan2(I3 - I1, I0 - I2))."""
    n = len(imgs)
    d = np.arange(n) * 2 * np.pi / n
    num = -np.tensordot(np.sin(d), imgs, axes=1)
    den = np.tensordot(np.cos(d), imgs, axes=1)
    return np.arctan2(num, den)


def wrap(a):
    return np.angle(np.exp(1j * a))


def temporal_unwrap(dphi_fine_wrapped, dphi_coarse, ratio):
    """Hierarchical unwrapping: the coarse (unwrapped) phase predicts the fine fringe order."""
    order = np.round((dphi_coarse * ratio - dphi_fine_wrapped) / (2 * np.pi))
    return dphi_fine_wrapped + 2 * np.pi * order


def slope_map(Z, dx, distance_mm=150.0, pitch_fine=40.0, pitch_coarse=400.0, axis=1, noise=0.02, rng=None,
              distance_error=0.0):
    """Measure one slope component of Z using a coarse and a fine stripe set.

    distance_error: relative error in the measured screen distance (a calibration error);
    the true distance used to create the images differs from the one assumed in decoding.
    """
    rng = np.random.default_rng(1) if rng is None else rng
    d_true = distance_mm * (1 + distance_error)
    Ic, cc = simulate_fringes(Z, dx, d_true, pitch_coarse, axis, noise, rng)
    If, cf = simulate_fringes(Z, dx, d_true, pitch_fine, axis, noise, rng)
    dphi_c = wrap(wrapped_phase(Ic) - cc)             # never wraps for the coarse period
    dphi_f = wrap(wrapped_phase(If) - cf)
    dphi_f = temporal_unwrap(dphi_f, dphi_c, pitch_coarse / pitch_fine)
    return dphi_f / phase_gain(distance_mm, pitch_fine)


def integrate_slopes(gx, gy, dx):
    """Frankot-Chellappa least-squares integration (solves the Poisson equation with the FFT)."""
    ny, nx = gx.shape
    # mirror-extend to remove the periodic-boundary assumption of the FFT
    gx_e = np.block([[gx, -gx[:, ::-1]], [gx[::-1, :], -gx[::-1, ::-1]]])
    gy_e = np.block([[gy, gy[:, ::-1]], [-gy[::-1, :], -gy[::-1, ::-1]]])
    u = fftfreq(2 * nx, dx) * 2 * np.pi
    v = fftfreq(2 * ny, dx) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    den = U ** 2 + V ** 2
    den[0, 0] = 1.0
    Zf = (-1j * U * fft2(gx_e) - 1j * V * fft2(gy_e)) / den
    Zf[0, 0] = 0
    return np.real(ifft2(Zf))[:ny, :nx]


def level_to_border(Z, width=6):
    """Remove the constant offset using the undamaged border of the patch as the reference plane."""
    b = np.concatenate([Z[:width].ravel(), Z[-width:].ravel(), Z[:, :width].ravel(), Z[:, -width:].ravel()])
    return Z - np.median(b)


def reconstruct(Z_true, dx, **kw):
    """Full simulated measurement of a surface: returns the reconstructed height map (mm)."""
    rng = np.random.default_rng(kw.pop("seed", 2))
    gx = slope_map(Z_true, dx, axis=1, rng=rng, **kw)
    gy = slope_map(Z_true, dx, axis=0, rng=rng, **kw)
    return level_to_border(integrate_slopes(gx, gy, dx))
