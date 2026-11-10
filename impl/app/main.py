"""Prototype REST API (FastAPI). Run: uvicorn app.main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dentassess import synthetic as S, deflectometry as Df, mechanics as M, repair as R, pipeline as PL

app = FastAPI(title="Dent Assessment API (prototype)", version="0.1.0")


class PanelIn(BaseModel):
    material: str = "mild steel"
    thickness_mm: float = Field(0.7, gt=0.3, lt=2.0)
    sigma_y_mpa: float = Field(200, gt=50, lt=1500)
    rear_access: bool = True
    paint_damaged: bool = False


class SyntheticDentIn(BaseModel):
    depth_mm: float = Field(3.0, gt=0, le=20)
    radius_mm: float = Field(30.0, gt=5, le=80)
    sharpness: float = Field(1.4, ge=1.0, le=4.0)
    panel: PanelIn = PanelIn()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/assess/synthetic")
def assess_synthetic(req: SyntheticDentIn):
    """Simulate a deflectometry scan of a synthetic dent and run the full assessment."""
    X, Y, dx = S.grid()
    Z = S.dent(X, Y, req.depth_mm, req.radius_mm, req.sharpness) + S.panel_waviness(X, Y)
    Zr = Df.reconstruct(Z, dx)
    return PL.assess(Zr, dx, R.Panel(**req.panel.model_dump()))


@app.get("/api/solver/verify")
def verify_solver():
    return M.verify_circular()
