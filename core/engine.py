"""Thin orchestrator: attitude → robustness filter → matcher/PnP → NavSolution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core.catalog import SiteCatalog, load_site_catalog
from core.localization import TripletIndex, build_triplet_index, match_and_pose
from core.robustness import (
    apply_homography,
    camera_intrinsics,
    plane_induced_homography,
)

LOCK_RESIDUAL_M = 5.0
COAST_RESIDUAL_M = 18.0
COAST_FRAMES = 10
SEARCH_RESIDUAL_M = 999.0

Status = Literal["LOCK", "COAST", "SEARCH"]


@dataclass
class NavSolution:
    lat: float
    lon: float
    altitude_m: float
    residual_m: float
    match_count: int
    latency_ms: float
    status: Status
    residual_px: float = 0.0
    matched_ids: tuple[str, ...] = ()
    observed_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    corrected_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    H: np.ndarray = field(default_factory=lambda: np.eye(3))
    box_quads_observed: np.ndarray | None = None
    box_quads_corrected: np.ndarray | None = None


def _box_quads(boxes: np.ndarray | None) -> np.ndarray | None:
    """Nx4x2 corner quads from axis-aligned ``[x1,y1,x2,y2,...]`` boxes."""
    if boxes is None:
        return None
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    if b.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float64)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )


def _warp_quads(quads: np.ndarray | None, H: np.ndarray) -> np.ndarray | None:
    if quads is None:
        return None
    q = np.asarray(quads, dtype=np.float64)
    if q.size == 0:
        return q
    flat = q.reshape(-1, 2)
    warped = apply_homography(flat, H).reshape(q.shape)
    return warped


def _diameters_from_boxes(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is None:
        return None
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    if b.size == 0:
        return None
    w = np.abs(b[:, 2] - b[:, 0])
    h = np.abs(b[:, 3] - b[:, 1])
    return np.maximum(w, h)


class Engine:
    """Interactive geometry loop. Catalog index is built once at startup."""

    def __init__(
        self,
        catalog: SiteCatalog | None = None,
        width: int = 960,
        height: int = 540,
        focal_px: float = 720.0,
        fixtures_are_nadir: bool = True,
        lock_residual_m: float = LOCK_RESIDUAL_M,
        coast_residual_m: float = COAST_RESIDUAL_M,
        coast_frames: int = COAST_FRAMES,
    ) -> None:
        self.catalog = catalog if catalog is not None else load_site_catalog()
        self.index: TripletIndex = build_triplet_index(self.catalog)
        self.width = int(width)
        self.height = int(height)
        self.focal_px = float(focal_px)
        self.K = camera_intrinsics(self.width, self.height, self.focal_px)
        self.fixtures_are_nadir = bool(fixtures_are_nadir)
        self.lock_residual_m = float(lock_residual_m)
        self.coast_residual_m = float(coast_residual_m)
        self.coast_frames = int(coast_frames)
        self._frame_i = 0
        self._last_lock_frame = -10**9
        self._last_lock: NavSolution | None = None

    def reset_coast(self) -> None:
        self._frame_i = 0
        self._last_lock_frame = -10**9
        self._last_lock = None

    def process_frame(
        self,
        centroids: np.ndarray,
        pitch_deg: float,
        roll_deg: float,
        *,
        use_filter: bool = True,
        boxes: np.ndarray | None = None,
        gsd_m_per_px: float | None = None,
    ) -> NavSolution:
        t0 = time.perf_counter()
        self._frame_i += 1
        centroids = np.atleast_2d(np.asarray(centroids, dtype=np.float64))
        H_t2n = plane_induced_homography(self.K, pitch_deg, roll_deg)
        H_n2t = np.linalg.inv(H_t2n)

        # Demo fixtures are generated from a nadir camera. The flight-deck
        # sliders are the vehicle attitude, so we first induce the tilted
        # observation (H_n→t) and then optionally map it back with the
        # robustness filter (H_t→n). Offline LFM fixtures from a truly
        # tilted camera should set fixtures_are_nadir=False.
        if self.fixtures_are_nadir:
            observed_uv = apply_homography(centroids, H_n2t)
            observed_quads = _warp_quads(_box_quads(boxes), H_n2t)
        else:
            observed_uv = centroids.copy()
            observed_quads = _box_quads(boxes)

        if use_filter:
            corrected_uv = apply_homography(observed_uv, H_t2n)
            corrected_quads = _warp_quads(observed_quads, H_t2n)
            H_used = H_t2n
        else:
            corrected_uv = observed_uv.copy()
            corrected_quads = observed_quads
            H_used = np.eye(3, dtype=np.float64)

        finite = np.isfinite(corrected_uv).all(axis=1)
        query_uv = corrected_uv[finite]
        diams = _diameters_from_boxes(boxes)
        if diams is not None:
            diams = diams[finite[: len(diams)]] if len(diams) == len(corrected_uv) else None
            if self.fixtures_are_nadir and not use_filter and observed_quads is not None:
                # After tilt, box size is a better diameter proxy than the nadir box.
                q = observed_quads.reshape(-1, 4, 2)
                span = np.linalg.norm(q[:, 0] - q[:, 2], axis=1)
                diams = span[finite[: len(span)]] if len(span) == len(corrected_uv) else diams

        gsd = float(gsd_m_per_px) if gsd_m_per_px is not None else self.catalog.gsd_m_per_px_ref
        match = match_and_pose(
            query_uv,
            self.catalog,
            self.index,
            self.K,
            diameters_px=diams,
            gsd_m_per_px=gsd,
        )

        if match is None:
            sol = NavSolution(
                lat=float("nan"),
                lon=float("nan"),
                altitude_m=float("nan"),
                residual_m=SEARCH_RESIDUAL_M,
                match_count=0,
                latency_ms=0.0,
                status="SEARCH",
                residual_px=SEARCH_RESIDUAL_M,
                matched_ids=(),
                observed_uv=observed_uv,
                corrected_uv=corrected_uv,
                H=H_used,
                box_quads_observed=observed_quads,
                box_quads_corrected=corrected_quads,
            )
        else:
            east, north, up = match.camera_enu
            lat, lon = self.catalog.enu_to_latlon(float(east), float(north))
            sol = NavSolution(
                lat=lat,
                lon=lon,
                altitude_m=float(up),
                residual_m=float(match.residual_m),
                match_count=int(match.inlier_count),
                latency_ms=0.0,
                status="SEARCH",
                residual_px=float(match.residual_px),
                matched_ids=match.ids,
                observed_uv=observed_uv,
                corrected_uv=corrected_uv,
                H=H_used,
                box_quads_observed=observed_quads,
                box_quads_corrected=corrected_quads,
            )

        sol.status = self._status(sol)
        if sol.status == "LOCK":
            self._last_lock_frame = self._frame_i
            self._last_lock = sol
        elif (
            sol.status == "COAST"
            and not np.isfinite(sol.lat)
            and self._last_lock is not None
        ):
            sol.lat = self._last_lock.lat
            sol.lon = self._last_lock.lon
            sol.altitude_m = self._last_lock.altitude_m
            sol.matched_ids = self._last_lock.matched_ids
            sol.match_count = self._last_lock.match_count

        sol.latency_ms = (time.perf_counter() - t0) * 1000.0
        return sol

    def _status(self, sol: NavSolution) -> Status:
        locked = (
            sol.match_count >= 5
            and np.isfinite(sol.residual_m)
            and sol.residual_m <= self.lock_residual_m
            and np.isfinite(sol.altitude_m)
            and sol.altitude_m > 0
        )
        if locked:
            return "LOCK"
        recently = (self._frame_i - self._last_lock_frame) <= self.coast_frames
        weak = sol.match_count >= 3 and sol.residual_m <= self.coast_residual_m
        if recently and (weak or self._last_lock is not None):
            return "COAST"
        return "SEARCH"
