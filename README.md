# SeleneTRN

GPS-denied lunar **terrain-relative navigation** demo. NASA-IBM Lunar Foundation Model crater detections arrive as **precomputed fixtures**. The live path is a plane-induced Robustness Filter plus a triangle-fingerprint / `solvePnP` Localization Engine on CPU.

```
fixtures (u,v) + pitch/roll
    → Robustness Filter   (H = K_n R_n R_t⁻¹ K_t⁻¹)
    → Localization Engine (catalog triangles + solvePnP)
    → NavSolution         (lat, lon, alt, residual, LOCK|COAST|SEARCH)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No Hugging Face download. No Ultralytics. First launch will generate the procedural descent clip if `data/media/demo_descent.mp4` is missing.

```bash
pip install pytest
python -m pytest
```

## Geometry notes

- Aerospace attitude, yaw = 0: pitch about body-Y, roll about body-X. Nadir camera looks along world −Z.
- Do **not** use `H = K R K⁻¹`. Production formula is the same-center plane-induced homography `H = K_n R_n R_t⁻¹ K_t⁻¹` (see comments in `core/robustness.py`).
- Lat/lon uses site `meters_per_deg_*` at the catalog origin, not a global 111 km/deg.
- Residual = reprojection RMSE (pixels × metres/pixel from current altitude / `fx`).

## Layout

```
app.py                      Streamlit flight deck
core/robustness.py          Subsystem 1
core/localization.py        Subsystem 2
core/engine.py              NavSolution orchestrator
core/catalog.py
inference/load_fixtures.py
inference/generate_demo.py  Procedural media + fixtures
inference/lfm_runner.py     Optional offline LFM
data/catalog/
data/fixtures/demo_run/
data/media/demo_descent.mp4
assets/styles.css
tests/
docs/devpost.md
```
