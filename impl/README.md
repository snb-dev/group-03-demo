# dentassess – early prototype (Progress Report I)

Physics-informed dent assessment pipeline, Group 03 (EER6689).
All results in this folder come from **simulated** dents; real test-panel validation is the next step.

| Module | Contents |
|---|---|
| `dentassess/synthetic.py` | Synthetic dents with exact height maps |
| `dentassess/deflectometry.py` | 4-step phase shifting, two-period temporal unwrapping, slope from phase, Frankot–Chellappa integration |
| `dentassess/mechanics.py` | Curvature, bending + membrane strain, strain-based PDR check, clamped-plate finite-difference solver |
| `dentassess/repair.py` | Yield-line force, glue-tab area, number of pulls, push / glue-pull / stud-pull / replace selection, NNLS push planning |
| `dentassess/confidence.py` | Monte Carlo success probability, split-conformal intervals, Brier score |
| `dentassess/pipeline.py` | End-to-end assessment returning an explained report |
| `app/main.py` | FastAPI prototype: `/api/health`, `/api/assess/synthetic`, `/api/solver/verify` |

```bash
pip install numpy scipy matplotlib fastapi uvicorn pytest httpx
python -m pytest -v          # 8 tests
python demo.py               # writes results/*.png and results/results.json
uvicorn app.main:app --reload
```
