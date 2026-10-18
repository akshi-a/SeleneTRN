# Generating SeleneTRN fixtures

The Streamlit flight deck **never** downloads the NASA-IBM Lunar Foundation Model. Judges and CI run entirely on the files under `data/`.

## Demo path (what ships)

Procedural nadir descent whose crater field is the site catalog, plus per-frame detection JSON that matches those projected craters.

```bash
python -m inference.generate_demo
```

Writes:

- `data/catalog/site_craters.csv` and `site_meta.json`
- `data/fixtures/demo_run/detections_*.json` + `meta.json`
- `data/media/demo_descent.mp4`

`meta.json` tags `projection: "nadir"` and `model_id: "NASA-IBM LFM NAC LoRA · precomputed"`. The UI sliders then induce a tilted observation and the robustness filter maps it back. That is how lock holds with the filter on and degrades with it off, without a multi-GB checkpoint.

## Optional LFM regeneration (GPU)

When you have the NASA-IBM LFM stack and want detections from real NAC tiles:

1. `pip install -r requirements-lfm.txt`
2. Install [NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) (TerraTorch Faster R-CNN LoRA).
3. Weights:
   - backbone `nasa-ibm-lunar-fm`
   - crater checkpoint `nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model`
4. Wire `predict_frame()` in `inference/lfm_runner.py` to the TerraTorch predict path (the rest of the JSON writer is already in place).
5. Run:

```bash
python -m inference.lfm_runner --download-only          # fetch weights only
python -m inference.lfm_runner --frames-dir tiles/ --out data/fixtures/lfm_run
```

Output schema is identical to the demo fixtures:

```json
{"frame_id": 0, "boxes": [[x1, y1, x2, y2, score]], "centroids": [[u, v]]}
```

Set `projection` in `meta.json` to `tilted` if the tiles were captured off-nadir, and the engine will treat centroids as already-tilted (`fixtures_are_nadir=False`).

Schema-only smoke (no weights):

```bash
python -m inference.lfm_runner --schema-demo --out /tmp/selenetrn-schema
```
