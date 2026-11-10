"""Repair-method selection: push repair, dent pulling (glue-pull / stud-pull) or replacement."""
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import nnls


def plastic_moment(sigma_y_mpa, t_mm):
    """Mp = sigma_y t^2 / 4 (N*mm per mm)."""
    return sigma_y_mpa * t_mm ** 2 / 4


def collapse_force(sigma_y_mpa, t_mm):
    """Yield-line collapse load of a point-loaded plate: Pc = 2*pi*Mp = (pi/2) sigma_y t^2 (N)."""
    return 2 * math.pi * plastic_moment(sigma_y_mpa, t_mm)


def glue_tab_area(Pc_n, safety=2.0, sigma_glue_mpa=1.5, sigma_adh_mpa=1.0):
    """Minimum tab area so that pull stress stays below glue strength and paint adhesion (mm^2)."""
    return safety * Pc_n / min(sigma_glue_mpa, sigma_adh_mpa)


def number_of_pulls(sigma_y_mpa, t_mm, depth_mm, hammer_kg=1.0, speed_m_s=1.5, efficiency=0.15):
    """N ~ ceil(2*pi*Mp*delta / (eta * 0.5 m v^2)). Energies in N*mm."""
    work = 2 * math.pi * plastic_moment(sigma_y_mpa, t_mm) * depth_mm
    per_pull = efficiency * 0.5 * hammer_kg * speed_m_s ** 2 * 1000.0
    return max(1, math.ceil(work / per_pull))


@dataclass
class Panel:
    material: str = "mild steel"
    thickness_mm: float = 0.7
    sigma_y_mpa: float = 200.0
    rear_access: bool = True
    paint_damaged: bool = False


@dataclass
class Recommendation:
    method: str
    reasons: list = field(default_factory=list)
    force_n: float = 0.0
    tab_diameter_mm: float = 0.0
    pulls: int = 0


def select_method(panel: Panel, suitability: dict, dent_diameter_mm: float, max_size_mm=150.0):
    """Rule layer on top of the mechanics results. Every decision carries its reasons."""
    r, Pc = [], collapse_force(panel.sigma_y_mpa, panel.thickness_mm)
    if dent_diameter_mm > max_size_mm:
        return Recommendation("replace / conventional repair", [f"dent ({dent_diameter_mm:.0f} mm) is larger than the PDR scope ({max_size_mm:.0f} mm)"])
    if panel.paint_damaged or not suitability["suitable"]:
        r.append("paint already damaged" if panel.paint_damaged else suitability["reason"])
        n = number_of_pulls(panel.sigma_y_mpa, panel.thickness_mm, suitability["depth_mm"])
        return Recommendation("stud-pull + refinish", r + ["refinishing is required, so welded studs can be used"], Pc, 0.0, n)
    r.append(suitability["reason"])
    if panel.rear_access:
        r.append("rear side is accessible, so the dent can be pushed out with rods")
        return Recommendation("push repair", r, Pc)
    A = glue_tab_area(Pc)
    dtab = 2 * math.sqrt(A / math.pi)
    if dtab > 0.8 * dent_diameter_mm:
        r.append(f"required glue tab ({dtab:.0f} mm) is too large for the dent")
        return Recommendation("stud-pull + refinish", r, Pc, dtab, number_of_pulls(panel.sigma_y_mpa, panel.thickness_mm, suitability["depth_mm"]))
    n = number_of_pulls(panel.sigma_y_mpa, panel.thickness_mm, suitability["depth_mm"])
    r.append(f"no rear access: glue-pull with a {dtab:.0f} mm tab, about {n} pulls")
    return Recommendation("glue-pull", r, Pc, dtab, n)


def plan_pushes(residual, plate, candidates, threshold=0.05):
    """Non-negative least squares: forces f >= 0 so that G f cancels the dent (elastic approximation)."""
    cols = [plate.point_load(i, j, 1.0)[plate.mask] for i, j in candidates]
    G = np.stack(cols, axis=1)
    f, res = nnls(G, -residual[plate.mask])
    keep = f > threshold * f.max() if f.max() > 0 else f > 0
    return [(candidates[k], float(f[k])) for k in np.where(keep)[0]], float(res)
