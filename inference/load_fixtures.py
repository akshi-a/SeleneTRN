"""Load precomputed per-frame crater detections (no live LFM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "data" / "fixtures" / "demo_run"


@dataclass
class FrameDetections:
    frame_id: int
    boxes: np.ndarray  # (N, 5) x1,y1,x2,y2,score
    centroids: np.ndarray  # (N, 2)


@dataclass
class FixtureRun:
    meta: dict
    frames: list[FrameDetections]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])

    @property
    def focal_px(self) -> float:
        return float(self.meta["focal_px"])

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps", 15))

    @property
    def gsd_m_per_px(self) -> float:
        return float(self.meta.get("gsd_m_per_px", 1.0))

    @property
    def model_id(self) -> str:
        return str(self.meta.get("model_id", "precomputed"))

    @property
    def fixtures_are_nadir(self) -> bool:
        return str(self.meta.get("projection", "nadir")).lower() == "nadir"

    def frame(self, frame_id: int) -> FrameDetections:
        return self.frames[int(frame_id)]


def load_fixture_run(run_dir: Path | str | None = None) -> FixtureRun:
    run_dir = Path(run_dir) if run_dir else DEFAULT_RUN
    meta = json.loads((run_dir / "meta.json").read_text())
    n = int(meta.get("n_frames", 0))
    paths = sorted(run_dir.glob("detections_*.json"))
    if n and len(paths) != n:
        # Prefer glob; n_frames is advisory if files were regenerated.
        pass
    frames: list[FrameDetections] = []
    for path in paths:
        payload = json.loads(path.read_text())
        boxes = np.asarray(payload.get("boxes", []), dtype=np.float64)
        if boxes.size == 0:
            boxes = np.zeros((0, 5), dtype=np.float64)
        elif boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        cents = np.asarray(payload.get("centroids", []), dtype=np.float64)
        if cents.size == 0:
            cents = np.zeros((0, 2), dtype=np.float64)
        elif cents.ndim == 1:
            cents = cents.reshape(1, 2)
        frames.append(
            FrameDetections(
                frame_id=int(payload["frame_id"]),
                boxes=boxes,
                centroids=cents,
            )
        )
    frames.sort(key=lambda f: f.frame_id)
    return FixtureRun(meta=meta, frames=frames)
