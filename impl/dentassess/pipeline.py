"""End-to-end assessment: height map -> suitability -> method -> confidence -> report dict."""
import numpy as np
from . import mechanics as M, repair as R, confidence as C


def dent_diameter(Z, dx, frac=0.1):
    """Equivalent diameter of the area deeper than frac * max depth (mm)."""
    area = (Z < frac * Z.min()).sum() * dx * dx
    return float(2 * np.sqrt(area / np.pi))


def assess(Z, dx, panel: R.Panel, eps_crack=0.010, eps_crack_sd=0.002, n_mc=300, seed=0):
    suit = M.pdr_suitability(Z, dx, panel.thickness_mm, eps_crack)
    diam = dent_diameter(Z, dx)
    rec = R.select_method(panel, suit, diam)
    _, _, e = M.strain_maps(Z, dx, panel.thickness_mm)
    peak = float(e.max())

    def ok(s):  # success if the sampled peak strain stays below the sampled cracking strain
        return peak * s["scale"] < 0.8 * s["eps_crack"]
    p = C.monte_carlo_success(ok, {"scale": (1.0, 0.05), "eps_crack": (eps_crack, eps_crack_sd)}, n_mc, seed)
    return {
        "depth_mm": round(float(-Z.min()), 2), "diameter_mm": round(diam, 1),
        "peak_strain_pct": round(100 * peak, 2), "pdr_suitable": suit["suitable"],
        "pdr_success_probability": round(p, 2), "method": rec.method, "reasons": rec.reasons,
        "pull_force_n": round(rec.force_n, 1), "glue_tab_diameter_mm": round(rec.tab_diameter_mm, 1),
        "pulls": rec.pulls, "manual_check": 0.2 < p < 0.8,
    }
