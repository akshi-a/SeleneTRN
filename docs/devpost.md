# SeleneTRN: NASA-IBM Lunar Foundation Model for Edge Terrain Relative Navigation

## Elevator pitch

SeleneTRN is a GPS-denied lunar terrain-relative navigation stack. A NASA-IBM Lunar Foundation Model crater detector proposes landmarks offline; a plane-induced **Robustness Filter** maps those detections off a tilted camera onto a nadir image plane; a **triangle-fingerprint matcher** plus `solvePnP` recovers lat, lon, and altitude with a real reprojection residual. The live demo is the geometry loop — sub-10 ms on CPU — not a multi-gigabyte model in the browser.

## The problem

Landing-site TRN still leans on normalized cross-correlation against map patches. NCC is brittle under off-nadir attitude and changing illumination. Lost-in-space (no prior pose) is worse: you cannot slide a template until you already know where you are. Craters are attitude-stable landmarks *if* you compare them in a consistent image plane and identify them by geometry rather than appearance.

## What we built

Three layers, with honest boundaries between them:

1. **Landmark front-end (offline).** NASA-IBM LFM NAC LoRA Faster R-CNN detections, stored as per-frame JSON (`boxes`, `centroids`). The flight deck never loads TerraTorch or Hugging Face weights. An optional runner (`inference/lfm_runner.py`) documents how to regenerate the same schema on a GPU machine.

2. **Robustness Filter (live).** Pitch about body-Y, roll about body-X, yaw fixed. Ground is the local `Z = 0` plane. The two cameras share an optical center, so the mapping is the plane-induced homography `H = K_n R_n R_t⁻¹ K_t⁻¹`. This is not `H = K R K⁻¹` — that formula uses the wrong relative rotation and the wrong axes for an OpenCV camera under aerospace attitude. Centroids (and box quads) are warped with `cv2.perspectiveTransform` semantics onto an equivalent nadir plane so triangle side-ratios stop drifting with tilt.

3. **Localization Engine (live).** Unordered triplets with min-side / area gates, fingerprint `[s0/s2, s1/s2]`, a precomputed catalog index, L2 ratio match under a tight gate, diameter-order tie-break, then `solvePnP` (IPPE or ITERATIVE / P3P) and inlier expansion. Altitude is camera height above the plane. Lat/lon is an ENU offset using the site’s meters-per-degree at origin latitude — not a global `÷ 111000`. The chart is reprojection RMSE in metres, not a cosmetic decay.

`NavSolution.status` is `LOCK` inside a 5 m residual envelope with a consistent match, `COAST` when a recent lock is held through a weaker frame, `SEARCH` otherwise.

## How the demo is fair

The optical plane is a short looping gray lunar descent whose crater pattern **is** the catalog, so the attitude experiment is causal: filter on, lock holds across the ±30° pitch/roll sliders; filter off, ratios drift and the match breaks or the residual leaves the envelope. Site coordinates are demo-consistent, not a geodetic product. Latency shown in the header is the geometry path that just ran.

## Why it matters

Crewed and robotic lunar descent will spend long stretches without GNSS. A detector that only fires on the ground does not close the loop; the loop is attitude-stable correspondence plus a pose with an error you can plot. SeleneTRN keeps the foundation-model work where it belongs (offline, NAC-scale landmarks) and puts a correct geometric TRN filter on the edge — small enough to interact with, honest enough to measure.

## Built with

Python, OpenCV, NumPy, Streamlit, Plotly. Optional: PyTorch / TerraTorch / Hugging Face for fixture regeneration. No Ultralytics in the core path.
