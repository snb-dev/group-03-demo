import numpy as np
from fastapi.testclient import TestClient
from dentassess import synthetic as S, deflectometry as Df, mechanics as M, repair as R, confidence as C


def _ref(Z):
    return Df.level_to_border(Z)


def test_phase_shift_recovers_phase():
    phi = np.linspace(-3, 3, 50)
    imgs = np.array([0.5 + 0.4 * np.cos(phi + k * np.pi / 2) for k in range(4)])
    assert np.allclose(Df.wrapped_phase(imgs), phi, atol=1e-9)


def test_reconstruction_depth_within_half_mm():
    X, Y, dx = S.grid()
    for depth in (1.5, 4.0, 8.0):
        Z = S.dent(X, Y, depth=depth, radius=30) + S.panel_waviness(X, Y)
        Zr = Df.reconstruct(Z, dx, noise=0.05, distance_error=0.02)
        assert abs(Zr.min() - _ref(Z).min()) <= 0.5


def test_plate_solver_matches_exact_solution():
    v = M.verify_circular(radius_mm=50, h=1.0)
    assert v["point_error_pct"] < 3 and v["uniform_error_pct"] < 4


def test_flat_panel_is_suitable_and_deep_crease_is_not():
    X, Y, dx = S.grid()
    shallow = S.dent(X, Y, depth=1.0, radius=45)
    crease = S.dent(X, Y, depth=8.0, radius=12, sharpness=3.0)
    assert M.pdr_suitability(shallow, dx)["suitable"]
    assert not M.pdr_suitability(crease, dx)["suitable"]


def test_yield_line_force_example():
    # 0.7 mm mild steel, 200 MPa -> about 154 N
    assert abs(R.collapse_force(200, 0.7) - 153.9) < 0.5


def test_method_selection_rules():
    suit = {"suitable": True, "reason": "ok", "depth_mm": 3.0}
    assert R.select_method(R.Panel(rear_access=True), suit, 60).method == "push repair"
    assert R.select_method(R.Panel(rear_access=False), suit, 60).method == "glue-pull"
    assert R.select_method(R.Panel(paint_damaged=True), suit, 60).method == "stud-pull + refinish"
    assert R.select_method(R.Panel(), suit, 400).method.startswith("replace")


def test_conformal_coverage():
    # the guarantee is on average over calibration sets, so average over many repetitions
    rng = np.random.default_rng(0)
    covers = []
    for _ in range(300):
        y = rng.standard_t(4, 1200)
        q = C.conformal_halfwidth(y[:200], np.zeros(200), alpha=0.1)
        covers.append(np.mean(np.abs(y[200:]) <= q))
    assert np.mean(covers) >= 0.895


def test_api_end_to_end():
    from app.main import app
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    r = c.post("/api/assess/synthetic", json={"depth_mm": 2.0, "radius_mm": 40, "panel": {"rear_access": False}}).json()
    assert r["method"] in ("glue-pull", "stud-pull + refinish") and 0 <= r["pdr_success_probability"] <= 1
