"""Curvature, strain and a finite-difference thin-plate solver."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter


# ------------------------------------------------------------------ curvature & strain
def curvatures(Z, dx):
    """Mean H, Gaussian K and principal curvatures k1 >= k2 (1/mm) of z(x, y)."""
    zy, zx = np.gradient(Z, dx)
    zxy, zxx = np.gradient(zx, dx)
    zyy, _ = np.gradient(zy, dx)
    g = 1 + zx ** 2 + zy ** 2
    H = ((1 + zx ** 2) * zyy - 2 * zx * zy * zxy + (1 + zy ** 2) * zxx) / (2 * g ** 1.5)
    K = (zxx * zyy - zxy ** 2) / g ** 2
    disc = np.sqrt(np.maximum(H ** 2 - K, 0))
    return H, K, H + disc, H - disc


def smooth(Z, dx, smooth_mm=2.0):
    """Gaussian smoothing before differentiation: curvature amplifies measurement noise."""
    return gaussian_filter(Z, smooth_mm / dx) if smooth_mm > 0 else Z


def strain_maps(Z, dx, thickness_mm, smooth_mm=2.0):
    """Bending strain (t/2)*|k|max and membrane strain 0.5*|grad z|^2 (dimensionless)."""
    Z = smooth(Z, dx, smooth_mm)
    _, _, k1, k2 = curvatures(Z, dx)
    kmax = np.maximum(np.abs(k1), np.abs(k2))
    zy, zx = np.gradient(Z, dx)
    eps_b = 0.5 * thickness_mm * kmax
    eps_m = 0.5 * (zx ** 2 + zy ** 2)
    return eps_b, eps_m, eps_b + eps_m


def crease_score(Z, dx, smooth_mm=2.0):
    """Ratio of peak to median curvature inside the dent: high values indicate a sharp crease."""
    Z = smooth(Z, dx, smooth_mm)
    _, _, k1, k2 = curvatures(Z, dx)
    kmax = np.maximum(np.abs(k1), np.abs(k2))
    inside = Z < 0.1 * Z.min()
    if inside.sum() < 10:
        return 0.0
    return float(kmax[inside].max() / (np.median(kmax[inside]) + 1e-9))


def pdr_suitability(Z, dx, thickness_mm=0.7, eps_crack=0.010, margin=0.8):
    """Strain-based PDR check: suitable when peak strain < margin * paint cracking strain."""
    eb, em, e = strain_maps(Z, dx, thickness_mm)
    peak = float(e.max())
    limit = margin * eps_crack
    suitable = peak < limit
    reason = (f"peak strain {peak*100:.2f}% is below the limit {limit*100:.2f}%" if suitable else
              f"peak strain {peak*100:.2f}% exceeds the limit {limit*100:.2f}%: paint is likely to crack")
    return {"suitable": suitable, "peak_strain": peak, "limit": limit,
            "bending_share": float(eb.max() / (peak + 1e-12)), "depth_mm": float(-Z.min()), "reason": reason}


# ------------------------------------------------------------------ plate solver
def flexural_rigidity(E_mpa, t_mm, nu):
    """D = E t^3 / (12 (1 - nu^2)) in N*mm."""
    return E_mpa * t_mm ** 3 / (12 * (1 - nu ** 2))


class ClampedPlate:
    """Solve D * del^4 w = q on a masked grid with clamped edges (w = 0, dw/dn = 0).

    13-point biharmonic stencil; points outside the mask are fixed at w = 0 and the
    zero-slope condition is imposed with mirrored ghost points (w_{i-2} = w_i when i-1 is
    on the boundary). The sparse matrix is factorised once, so many load cases are cheap.
    """
    OFFS = [((0, 0), 20.0), ((1, 0), -8.0), ((-1, 0), -8.0), ((0, 1), -8.0), ((0, -1), -8.0),
            ((1, 1), 2.0), ((1, -1), 2.0), ((-1, 1), 2.0), ((-1, -1), 2.0),
            ((2, 0), 1.0), ((-2, 0), 1.0), ((0, 2), 1.0), ((0, -2), 1.0)]

    def __init__(self, mask, h, D):
        self.mask, self.h, self.D = mask, h, D
        self.idx = -np.ones(mask.shape, int)
        pts = np.argwhere(mask)
        self.idx[mask] = np.arange(len(pts))
        ny, nx = mask.shape
        inside = lambda i, j: 0 <= i < ny and 0 <= j < nx and mask[i, j]
        rows, cols, vals = [], [], []
        for k, (i, j) in enumerate(pts):
            for (di, dj), c in self.OFFS:
                ii, jj = i + di, j + dj
                if inside(ii, jj):
                    rows.append(k); cols.append(self.idx[ii, jj]); vals.append(c)
                elif abs(di) == 2 or abs(dj) == 2:
                    mi, mj = i + di // 2, j + dj // 2          # the node in between
                    if not inside(mi, mj):                      # ghost beyond a boundary node
                        rows.append(k); cols.append(k); vals.append(c)   # w_ghost = w_i (mirror)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(len(pts), len(pts))) * (D / h ** 4)
        self.lu = spla.splu(A)

    def solve(self, q):
        """q: load per unit area (N/mm^2) on the grid. Returns deflection w (mm)."""
        w = np.zeros(self.mask.shape)
        w[self.mask] = self.lu.solve(q[self.mask])
        return w

    def point_load(self, i, j, P):
        q = np.zeros(self.mask.shape)
        q[i, j] = P / self.h ** 2
        return self.solve(q)


def circular_plate(radius_mm, h, E_mpa=200e3, t_mm=0.7, nu=0.3):
    n = int(round(2 * radius_mm / h)) + 5
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    mask = ((xx - c) * h) ** 2 + ((yy - c) * h) ** 2 < radius_mm ** 2
    D = flexural_rigidity(E_mpa, t_mm, nu)
    return ClampedPlate(mask, h, D), int(c), D


def verify_circular(radius_mm=50.0, h=1.0, P=10.0):
    """Compare the solver with the exact centre deflection of a clamped circular plate, P a^2/(16 pi D)."""
    plate, c, D = circular_plate(radius_mm, h)
    w = plate.point_load(c, c, P)
    exact = P * radius_mm ** 2 / (16 * np.pi * D)
    q = np.full(plate.mask.shape, 1e-3)
    wu = plate.solve(q)
    exact_u = 1e-3 * radius_mm ** 4 / (64 * D)
    return {"point_numeric": float(w[c, c]), "point_exact": float(exact),
            "point_error_pct": float(100 * abs(w[c, c] - exact) / exact),
            "uniform_numeric": float(wu[c, c]), "uniform_exact": float(exact_u),
            "uniform_error_pct": float(100 * abs(wu[c, c] - exact_u) / exact_u)}
