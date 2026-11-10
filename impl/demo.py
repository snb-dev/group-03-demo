"""Generates the prototype results reported in Progress Report I (simulated data only)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dentassess import synthetic as S, deflectometry as Df, mechanics as M, repair as R, pipeline as PL

OUT = "results/"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
X, Y, dx = S.grid()
res = {}

# 1. reconstruction accuracy over a set of dents, with camera noise and a +/-2 % distance calibration error
rows = []
rng = np.random.default_rng(7)
cases = [(1.0, 40), (2.0, 30), (3.0, 45), (4.0, 25), (5.0, 35), (6.0, 28), (7.0, 40), (8.0, 22), (9.0, 32), (10.0, 38)]
for k, (d, r) in enumerate(cases):
    Z = S.dent(X, Y, depth=d, radius=r, sharpness=1.3 + 0.1 * (k % 3)) + S.panel_waviness(X, Y)
    Zt = Df.level_to_border(Z)
    derr = float(rng.uniform(-0.02, 0.02))
    Zr = Df.reconstruct(Z, dx, noise=0.05, distance_error=derr, seed=k)
    rows.append({"true_depth": round(float(-Zt.min()), 3), "measured_depth": round(float(-Zr.min()), 3),
                 "depth_error": round(float(abs(Zr.min() - Zt.min())), 3), "map_mae": round(float(np.abs(Zr - Zt).mean()), 4),
                 "radius": r, "distance_error_pct": round(100 * derr, 2)})
res["reconstruction"] = rows
t = np.array([r["true_depth"] for r in rows]); m = np.array([r["measured_depth"] for r in rows])
res["reconstruction_summary"] = {"mean_abs_depth_error_mm": round(float(np.mean(np.abs(t - m))), 3),
                                 "max_abs_depth_error_mm": round(float(np.max(np.abs(t - m))), 3),
                                 "r2": round(float(1 - np.sum((t - m) ** 2) / np.sum((t - t.mean()) ** 2)), 4)}

# figure: pipeline on one dent
Z = S.dent(X, Y, depth=6, radius=28) + S.panel_waviness(X, Y)
I, carrier = Df.simulate_fringes(Z, dx, 150, 40, axis=1, noise=0.05)
wr = Df.wrap(Df.wrapped_phase(I) - carrier)
gx = Df.slope_map(Z, dx, axis=1, noise=0.05)
Zr = Df.reconstruct(Z, dx, noise=0.05)
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3), dpi=170)
ext = [-100, 100, -100, 100]
for a, img, title, cmap in [(ax[0], I[0], "(a) Reflected stripes (1 of 4)", "gray"), (ax[1], wr, "(b) Wrapped phase change", "twilight"),
                             (ax[2], gx, "(c) Unwrapped slope ∂z/∂x", "RdBu_r"), (ax[3], Zr, "(d) Height map z (mm)", "viridis")]:
    im = a.imshow(img, cmap=cmap, extent=ext); a.set_title(title, fontsize=12); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
plt.colorbar(im, ax=ax[3], fraction=0.046)
plt.tight_layout(); plt.savefig(OUT + "fig_deflectometry.png", facecolor="white"); plt.close()

# figure: measured vs true depth + profile
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=170)
ax[0].plot([0, 11], [0, 11], color="#999", ls="--", lw=1)
ax[0].scatter(t, m, color="black", s=40, zorder=3)
ax[0].set_xlabel("true depth (mm)"); ax[0].set_ylabel("measured depth (mm)")
ax[0].set_title(f"(a) Depth: MAE {res['reconstruction_summary']['mean_abs_depth_error_mm']} mm, R² {res['reconstruction_summary']['r2']}")
c = Z.shape[0] // 2
ax[1].plot(X[c], Df.level_to_border(Z)[c], color="black", lw=2.2, label="true profile")
ax[1].plot(X[c], Zr[c], color="#D55E00", lw=1.4, ls="--", label="reconstructed")
ax[1].set_xlabel("x (mm)"); ax[1].set_ylabel("z (mm)"); ax[1].legend(frameon=False); ax[1].set_title("(b) Centre-line profile, 6 mm dent")
plt.tight_layout(); plt.savefig(OUT + "fig_depth_accuracy.png", facecolor="white"); plt.close()

# 2. plate solver verification and convergence
conv = []
for h in (2.0, 1.0, 0.5):
    v = M.verify_circular(50, h); v["h_mm"] = h; conv.append(v)
res["plate_verification"] = conv
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=170)
hs = [c_["h_mm"] for c_ in conv]
ax.loglog(hs, [c_["point_error_pct"] for c_ in conv], "o-", color="black", label="central point load, P a²/(16πD)")
ax.loglog(hs, [c_["uniform_error_pct"] for c_ in conv], "s--", color="#666", label="uniform load, q a⁴/(64D)")
ax.set_xlabel("grid spacing h (mm)"); ax.set_ylabel("error vs exact solution (%)"); ax.legend(frameon=False, fontsize=10)
ax.set_title("Plate solver verification (clamped circular plate, a = 50 mm)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig_plate_verification.png", facecolor="white"); plt.close()

# 3. suitability map over depth x radius
ds = np.linspace(0.5, 10, 20); rs = np.linspace(15, 60, 19)
P = np.zeros((len(ds), len(rs)))
for i, d in enumerate(ds):
    for j, r in enumerate(rs):
        P[i, j] = 100 * M.strain_maps(S.dent(X, Y, depth=d, radius=r), dx, 0.7)[2].max()
fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=170)
cs = ax.contourf(rs, ds, P, levels=[0, 0.2, 0.4, 0.6, 0.8, 1.2, 2, 4, 8, 20], cmap="Greys")
ax.contour(rs, ds, P, levels=[0.8], colors="#D55E00", linewidths=2.2)
plt.colorbar(cs, ax=ax, label="peak strain (%)")
ax.text(48, 2.0, "PDR suitable", fontsize=12, weight="bold"); ax.text(20, 8.6, "not suitable", fontsize=12, weight="bold", color="white")
ax.set_xlabel("dent radius parameter (mm)"); ax.set_ylabel("dent depth (mm)")
ax.set_title("Strain-based PDR suitability (orange: 0.8% limit, assumed)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig_suitability_map.png", facecolor="white"); plt.close()

# 4. example end-to-end assessments
ex = []
for name, d, r, sh, panel in [("Shallow door dent, rear access", 1.5, 40, 1.3, R.Panel(rear_access=True)),
                              ("Shallow roof dent, no rear access", 1.8, 45, 1.3, R.Panel(rear_access=False)),
                              ("Sharp crease", 6.0, 18, 2.5, R.Panel(rear_access=True)),
                              ("Dent with damaged paint", 3.0, 35, 1.4, R.Panel(paint_damaged=True))]:
    Zc = S.dent(X, Y, d, r, sh) + S.panel_waviness(X, Y)
    out = PL.assess(Df.reconstruct(Zc, dx, noise=0.05), dx, panel)
    out["case"] = name; ex.append(out)
res["examples"] = ex
res["repair_constants"] = {"collapse_force_n_0.7mm_200MPa": round(R.collapse_force(200, 0.7), 1),
                           "glue_tab_area_mm2": round(R.glue_tab_area(R.collapse_force(200, 0.7)), 0)}
json.dump(res, open(OUT + "results.json", "w"), indent=1)
print(json.dumps(res["reconstruction_summary"]), json.dumps(conv, indent=0)[:400])
for e in ex: print(e)
