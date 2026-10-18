"""Procedural south-polar descent media + detection fixtures that match the catalog.

The crater field is the catalog projected through a nadir camera. Detections
are those projected centroids/boxes — so the attitude-filter demo locks when
the robustness filter is on and degrades when it is off.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core.catalog import meters_per_deg_lon
from core.robustness import NADIR_R, camera_intrinsics, project_world_points

REPO_ROOT = Path(__file__).resolve().parents[1]

WIDTH = 960
HEIGHT = 540
FOCAL_PX = 720.0
FPS = 15
N_FRAMES = 90
ORIGIN_LAT = -89.54
ORIGIN_LON = 129.78
M_PER_DEG_LAT = 111132.954
SITE_ID = "demo-south-polar-a"

# Distinctive, non-symmetric crater field (east_m, north_m, diameter_m).
CRATERS: list[tuple[str, float, float, float]] = [
    ("C01", -160.0, 95.0, 48.0),
    ("C02", -45.0, 125.0, 32.0),
    ("C03", 130.0, 88.0, 56.0),
    ("C04", -175.0, -15.0, 28.0),
    ("C05", 8.0, 22.0, 72.0),
    ("C06", 155.0, -28.0, 24.0),
    ("C07", -70.0, -105.0, 44.0),
    ("C08", 62.0, -118.0, 30.0),
    ("C09", 188.0, -95.0, 18.0),
    ("C10", -195.0, 55.0, 21.0),
    ("C11", 88.0, 48.0, 16.0),
    ("C12", -22.0, -52.0, 38.0),
    ("C13", 205.0, 18.0, 26.0),
    ("C14", -118.0, -58.0, 20.0),
    ("C15", 38.0, 108.0, 22.0),
    ("C16", -95.0, 35.0, 14.0),
    ("C17", 25.0, -88.0, 19.0),
]


def camera_center(frame_i: int, n_frames: int = N_FRAMES) -> np.ndarray:
    t = frame_i / max(n_frames - 1, 1)
    alt = 700.0 - 260.0 * t
    east = 16.0 * np.sin(2.0 * np.pi * t)
    north = 11.0 * np.cos(2.0 * np.pi * t)
    return np.array([east, north, alt], dtype=np.float64)


def _ll(east_m: float, north_m: float) -> tuple[float, float]:
    m_lon = meters_per_deg_lon(ORIGIN_LAT, M_PER_DEG_LAT)
    lat = ORIGIN_LAT + north_m / M_PER_DEG_LAT
    lon = ORIGIN_LON + east_m / m_lon
    return lat, lon


def write_catalog(root: Path = REPO_ROOT) -> None:
    cat_dir = root / "data" / "catalog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    m_lon = meters_per_deg_lon(ORIGIN_LAT, M_PER_DEG_LAT)
    rows = []
    for cid, x, y, d in CRATERS:
        lat, lon = _ll(x, y)
        rows.append(
            {
                "id": cid,
                "x_m": x,
                "y_m": y,
                "z_m": 0.0,
                "lat": lat,
                "lon": lon,
                "diameter_m": d,
            }
        )
    pd.DataFrame(rows).to_csv(cat_dir / "site_craters.csv", index=False)
    meta = {
        "site_id": SITE_ID,
        "origin_lat_deg": ORIGIN_LAT,
        "origin_lon_deg": ORIGIN_LON,
        "meters_per_deg_lat": M_PER_DEG_LAT,
        "meters_per_deg_lon": m_lon,
        "gsd_m_per_px_ref": 1.0,
        "description": (
            "Demo-consistent south-polar style site. ENU origin and crater "
            "positions are internally consistent with the fixture imagery; "
            "they are not a claim of true lunar geodesy."
        ),
    }
    (cat_dir / "site_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _value_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.55
    for o in range(octaves):
        gh = max(2, 4 << o)
        gw = max(2, 6 << o)
        grid = rng.random((gh, gw), dtype=np.float32)
        acc += cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        amp *= 0.55
    acc -= acc.min()
    acc /= max(float(acc.max()), 1e-6)
    return acc


def _stamp_crater(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    rng: np.random.Generator,
    catalog: bool,
) -> None:
    if radius_px < 2.0:
        return
    h, w = img.shape[:2]
    pad = int(radius_px * 2.4) + 2
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dx = (xx - cx) / radius_px
    dy = (yy - cy) / radius_px
    rho = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    bowl = np.clip(1.15 - rho, 0.0, 1.0)
    interior = (rho < 1.02).astype(np.float32)
    rim = np.exp(-((rho - 1.0) ** 2) / 0.018).astype(np.float32)
    ejecta = np.exp(-((rho - 1.25) ** 2) / 0.12).astype(np.float32) * (rho > 1.0)
    shade = (-0.40 * dx - 0.22 * dy).astype(np.float32)
    depth = 0.55 if catalog else 0.28
    rim_w = 0.28 if catalog else 0.16
    delta = (
        -depth * interior * bowl
        + rim_w * rim
        + 0.08 * ejecta
        + 0.32 * shade * interior
    )
    jitter = (rng.random(delta.shape).astype(np.float32) - 0.5) * (0.04 if catalog else 0.02)
    img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] + delta + jitter, 0.0, 1.0)


def render_frame(
    frame_i: int,
    rng: np.random.Generator,
    clutter: list[tuple[float, float, float]],
) -> tuple[np.ndarray, list[list[float]], list[list[float]]]:
    C = camera_center(frame_i)
    K = camera_intrinsics(WIDTH, HEIGHT, FOCAL_PX)
    gsd = float(C[2] / FOCAL_PX)

    # Terrain in a camera-centred ENU patch, then stamp projected craters.
    img = 0.18 + 0.22 * _value_noise(HEIGHT, WIDTH, rng)
    # Low-frequency albedo streaks.
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    img += 0.04 * np.sin((xx + 1.4 * yy) / 70.0)
    img = np.clip(img, 0.0, 1.0)

    clutter_xyz = np.array([[x, y, 0.0] for x, y, _d in clutter], dtype=np.float64)
    if len(clutter_xyz):
        uv_c = project_world_points(clutter_xyz, NADIR_R, C, K)
        for (u, v), (_x, _y, d) in zip(uv_c, clutter):
            if not np.isfinite(u):
                continue
            _stamp_crater(img, float(u), float(v), (d / 2.0) / gsd, rng, catalog=False)

    world = np.array([[x, y, 0.0] for _i, x, y, _d in CRATERS], dtype=np.float64)
    uv = project_world_points(world, NADIR_R, C, K)
    boxes: list[list[float]] = []
    centroids: list[list[float]] = []
    for (u, v), (_cid, _x, _y, diam) in zip(uv, CRATERS):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        r = (diam / 2.0) / gsd
        if u < -r or v < -r or u > WIDTH + r or v > HEIGHT + r:
            continue
        _stamp_crater(img, float(u), float(v), r, rng, catalog=True)
        if 4.0 <= u < WIDTH - 4.0 and 4.0 <= v < HEIGHT - 4.0 and r >= 4.0:
            x1, y1 = float(u - r), float(v - r)
            x2, y2 = float(u + r), float(v + r)
            score = float(np.clip(0.78 + 0.18 * (r / 40.0) + 0.04 * rng.random(), 0.55, 0.99))
            boxes.append([x1, y1, x2, y2, score])
            centroids.append([float(u), float(v)])

    # Sort detections by score descending so order is not catalog order.
    if boxes:
        order = np.argsort([-b[4] for b in boxes])
        boxes = [boxes[i] for i in order]
        centroids = [centroids[i] for i in order]

    grain = rng.normal(0.0, 0.012, img.shape).astype(np.float32)
    img = np.clip(img + grain, 0.0, 1.0)
    # Vignette.
    ny = (yy - HEIGHT / 2.0) / (HEIGHT / 2.0)
    nx = (xx - WIDTH / 2.0) / (WIDTH / 2.0)
    vig = np.clip(1.0 - 0.28 * (nx * nx + ny * ny), 0.35, 1.0)
    img *= vig
    gray = (img * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Cool lunar grade.
    bgr = bgr.astype(np.float32)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0] * 1.06, 0, 255)
    bgr[:, :, 2] = np.clip(bgr[:, :, 2] * 0.92, 0, 255)
    return bgr.astype(np.uint8), boxes, centroids


def write_media_and_fixtures(root: Path = REPO_ROOT, n_frames: int = N_FRAMES) -> Path:
    media_dir = root / "data" / "media"
    fix_dir = root / "data" / "fixtures" / "demo_run"
    media_dir.mkdir(parents=True, exist_ok=True)
    fix_dir.mkdir(parents=True, exist_ok=True)
    for old in fix_dir.glob("detections_*.json"):
        old.unlink()

    rng = np.random.default_rng(20261018)
    clutter = [
        (
            float(rng.uniform(-320, 320)),
            float(rng.uniform(-220, 220)),
            float(rng.uniform(4.0, 11.0)),
        )
        for _ in range(48)
    ]

    mp4_path = media_dir / "demo_descent.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open mp4v encoder")

    for i in range(n_frames):
        frame_rng = np.random.default_rng(20261018 + i)
        bgr, boxes, centroids = render_frame(i, frame_rng, clutter)
        writer.write(bgr)
        payload = {"frame_id": i, "boxes": boxes, "centroids": centroids}
        (fix_dir / f"detections_{i:06d}.json").write_text(json.dumps(payload) + "\n")
    writer.release()

    meta = {
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "focal_px": FOCAL_PX,
        "gsd_m_per_px": 1.0,
        "gsd_ref_altitude_m": FOCAL_PX,
        "site_id": SITE_ID,
        "model_id": "NASA-IBM LFM NAC LoRA · precomputed",
        "projection": "nadir",
        "n_frames": n_frames,
        "n_craters_catalog": len(CRATERS),
    }
    (fix_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return mp4_path


def generate_all(root: Path = REPO_ROOT) -> None:
    write_catalog(root)
    write_media_and_fixtures(root)


if __name__ == "__main__":
    generate_all()
    print("wrote catalog, fixtures, and", REPO_ROOT / "data" / "media" / "demo_descent.mp4")
