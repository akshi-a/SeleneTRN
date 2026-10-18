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

## 60-second demo script

1. **0:00–0:10** — `streamlit run app.py`. Header reads **SeleneTRN**, status **LOCK**, `GEOM` latency in milliseconds (the filter + matcher, not a fake detector FPS). Residual sits under the 5 m envelope.
2. **0:10–0:25** — Drag **PITCH** and **ROLL** across the ±30° range with **ROBUSTNESS FILTER** still on. Cyan (corrected) marks stay on the craters; amber (raw) marks slew. Telemetry holds lock; the residual chart stays inside the envelope.
3. **0:25–0:40** — Uncheck **ROBUSTNESS FILTER**. Same attitude. Triangle ratios drift, matched IDs drop or vanish, status goes **SEARCH** / residual leaves 5 m. Re-enable the filter: lock returns.
4. **0:40–0:55** — Scrub **FRAME** through the descent. Altitude tracks the trajectory; the chart is real reprojection RMSE, not a decay animation.
5. **0:55–1:00** — Footer: `detections: NASA-IBM LFM NAC LoRA · precomputed`. Point at `docs/devpost.md` for the honest claim boundary.

## Demo path vs optional LFM

| Path | What runs | When |
|---|---|---|
| **Demo (default)** | `data/fixtures/demo_run/` + `data/media/demo_descent.mp4` | Judges, CI, laptops |
| **Regenerate demo media** | `python -m inference.generate_demo` | After catalog edits |
| **LFM on GPU** | `pip install -r requirements-lfm.txt` then `python -m inference.lfm_runner` | Optional fixture rebuild from NAC tiles |

Details: [`scripts/generate_fixtures.md`](scripts/generate_fixtures.md). The LFM runner writes the **same JSON schema**; it does not have to run for the flight deck to work.

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
