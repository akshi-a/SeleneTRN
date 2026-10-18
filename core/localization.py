"""Subsystem 2 — triangle fingerprints, catalog match, solvePnP, residuals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import cv2
import numpy as np

from core.catalog import SiteCatalog
from core.robustness import camera_center_from_rt

# Side-ratio L2 gate. Ratios live in (0, 1]; 0.02 is tight for nadir-stable triangles.
RATIO_L2_THRESH = 0.018
MIN_SIDE_M = 25.0
MIN_AREA_M2 = 180.0
MIN_SIDE_PX = 12.0
MIN_AREA_PX = 80.0
INLIER_PX = 4.0
PERM_SCALE_CV_MAX = 0.08
MIN_INLIERS = 5
MAX_RMSE_PX = 2.0
ALTITUDE_RANGE_M = (150.0, 2500.0)
# Localization consumes nadir-plane measurements. Reject poses whose optical
# axis is more than this far from world −Z (filter-off tilted pixels fail here).
NADIR_AXIS_MAX_DEG = 5.0


@dataclass
class TripletIndex:
    """Precomputed unordered catalog triplets."""

    ratios: np.ndarray  # (M, 2) sorted side ratios [s0/s2, s1/s2]
    id_triples: list[tuple[str, str, str]]
    world: np.ndarray  # (M, 3, 3) XYZ of the three craters (catalog order)
    diameters: np.ndarray  # (M, 3)
    indices: np.ndarray  # (M, 3) integer rows into the catalog


@dataclass
class MatchResult:
    ids: tuple[str, ...]
    object_xyz: np.ndarray
    image_uv: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    residual_px: float
    residual_m: float
    inlier_count: int
    camera_enu: np.ndarray


def _pairwise_sides(pts: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(pts[0] - pts[1])
    b = np.linalg.norm(pts[1] - pts[2])
    c = np.linalg.norm(pts[2] - pts[0])
    return np.array([a, b, c], dtype=np.float64)


def _triangle_area(pts: np.ndarray) -> float:
    # 2-D cross magnitude; np.cross in NumPy 2+ requires 3-vectors.
    d1 = pts[1] - pts[0]
    d2 = pts[2] - pts[0]
    return 0.5 * abs(float(d1[0] * d2[1] - d1[1] * d2[0]))


def sorted_side_ratios(pts: np.ndarray) -> np.ndarray | None:
    """Fingerprint of an unordered triplet: sorted side ratios ``[s0/s2, s1/s2]``."""
    sides = np.sort(_pairwise_sides(pts[:, :2]))
    if sides[0] <= 1e-9 or sides[2] <= 1e-9:
        return None
    return np.array([sides[0] / sides[2], sides[1] / sides[2]], dtype=np.float64)


def build_triplet_index(
    catalog: SiteCatalog,
    min_side_m: float = MIN_SIDE_M,
    min_area_m2: float = MIN_AREA_M2,
) -> TripletIndex:
    xy = catalog.xy()
    xyz = catalog.xyz()
    diams = catalog.diameters()
    ids = catalog.ids
    ratios: list[np.ndarray] = []
    id_triples: list[tuple[str, str, str]] = []
    world: list[np.ndarray] = []
    diam_rows: list[np.ndarray] = []
    index_rows: list[tuple[int, int, int]] = []
    n = len(ids)
    for i, j, k in combinations(range(n), 3):
        pts = xy[[i, j, k]]
        sides = np.sort(_pairwise_sides(pts))
        if sides[0] < min_side_m:
            continue
        area = _triangle_area(pts)
        if area < min_area_m2:
            continue
        r = sorted_side_ratios(pts)
        if r is None:
            continue
        ratios.append(r)
        id_triples.append((ids[i], ids[j], ids[k]))
        world.append(xyz[[i, j, k]])
        diam_rows.append(diams[[i, j, k]])
        index_rows.append((i, j, k))
    if not ratios:
        raise ValueError("catalog produced no valid triplets; relax gates or add craters")
    return TripletIndex(
        ratios=np.vstack(ratios),
        id_triples=id_triples,
        world=np.stack(world, axis=0),
        diameters=np.stack(diam_rows, axis=0),
        indices=np.array(index_rows, dtype=np.int32),
    )


def _query_triplets(
    uv: np.ndarray,
    diameters: np.ndarray | None = None,
) -> list[tuple[np.ndarray, tuple[int, int, int], np.ndarray]]:
    """Vectorized unordered query triplets (indices, ratios, diameters)."""
    n = uv.shape[0]
    if n < 3:
        return []
    finite = np.isfinite(uv).all(axis=1)
    if diameters is None:
        diameters = np.full(n, np.nan)
    idx = np.array(list(combinations(np.flatnonzero(finite), 3)), dtype=np.int32)
    if idx.size == 0:
        return []
    a, b, c = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
    s01 = np.linalg.norm(a - b, axis=1)
    s12 = np.linalg.norm(b - c, axis=1)
    s20 = np.linalg.norm(c - a, axis=1)
    sides = np.sort(np.stack([s01, s12, s20], axis=1), axis=1)
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    ok = (sides[:, 0] >= MIN_SIDE_PX) & (area >= MIN_AREA_PX) & (sides[:, 2] > 1e-9)
    idx = idx[ok]
    sides = sides[ok]
    if len(idx) == 0:
        return []
    ratios = sides[:, :2] / sides[:, 2:3]
    out: list[tuple[np.ndarray, tuple[int, int, int], np.ndarray]] = []
    for t, r in zip(idx, ratios):
        trip = (int(t[0]), int(t[1]), int(t[2]))
        out.append((r, trip, diameters[list(trip)]))
    return out


def _assignment_error(q_xy: np.ndarray, c_xy: np.ndarray) -> float:
    """Scale-coefficient-of-variation of corresponding pairwise distances."""
    dq = np.array(
        [
            np.linalg.norm(q_xy[0] - q_xy[1]),
            np.linalg.norm(q_xy[1] - q_xy[2]),
            np.linalg.norm(q_xy[2] - q_xy[0]),
        ]
    )
    dc = np.array(
        [
            np.linalg.norm(c_xy[0] - c_xy[1]),
            np.linalg.norm(c_xy[1] - c_xy[2]),
            np.linalg.norm(c_xy[2] - c_xy[0]),
        ]
    )
    scale = dq / np.clip(dc, 1e-9, None)
    mean = float(np.mean(scale))
    if mean <= 1e-12:
        return 1e9
    return float(np.std(scale) / mean)


def assign_triplet(
    query_uv: np.ndarray,
    catalog_xy: np.ndarray,
    query_diameters: np.ndarray | None = None,
    catalog_diameters: np.ndarray | None = None,
) -> tuple[tuple[int, int, int] | None, float]:
    """Map 3 query vertices onto 3 catalog vertices.

    Diameter-order is used as a tie-break when both diameter vectors are finite;
    otherwise all 6 permutations are scored by pairwise scale consistency.
    """
    best_perm: tuple[int, int, int] | None = None
    best_err = 1e9
    if (
        query_diameters is not None
        and catalog_diameters is not None
        and np.isfinite(query_diameters).all()
        and np.isfinite(catalog_diameters).all()
        and len(set(np.round(query_diameters, 3))) == 3
        and len(set(np.round(catalog_diameters, 3))) == 3
    ):
        q_order = np.argsort(-query_diameters)
        c_order = np.argsort(-catalog_diameters)
        perm_via_diam = [0, 0, 0]
        for qpos, cpos in zip(q_order, c_order):
            perm_via_diam[int(qpos)] = int(cpos)
        perm_t = tuple(perm_via_diam)  # type: ignore[arg-type]
        err = _assignment_error(query_uv, catalog_xy[list(perm_t)])
        if err < PERM_SCALE_CV_MAX:
            return perm_t, err

    for perm in permutations(range(3)):
        err = _assignment_error(query_uv, catalog_xy[list(perm)])
        if err < best_err:
            best_err = err
            best_perm = perm  # type: ignore[assignment]
    if best_perm is None or best_err > PERM_SCALE_CV_MAX:
        return None, best_err
    return best_perm, best_err


def _solve_pnp(
    object_xyz: np.ndarray,
    image_uv: np.ndarray,
    K: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    obj = np.ascontiguousarray(object_xyz.reshape(-1, 1, 3).astype(np.float64))
    img = np.ascontiguousarray(image_uv.reshape(-1, 1, 2).astype(np.float64))
    dist = np.zeros((4, 1), dtype=np.float64)
    n = obj.shape[0]
    if n < 3:
        return False, np.zeros((3, 1)), np.zeros((3, 1))

    if n == 3:
        try:
            ok, rvecs, tvecs = cv2.solveP3P(
                obj, img, K, dist, flags=cv2.SOLVEPNP_P3P
            )
        except cv2.error:
            ok, rvecs, tvecs = 0, [], []
        if ok and rvecs:
            best = None
            for rvec, tvec in zip(rvecs, tvecs):
                R, _ = cv2.Rodrigues(rvec)
                C = camera_center_from_rt(R, tvec)
                proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image_uv) ** 2, axis=1))))
                # Prefer a camera above the ground plane.
                score = rmse + (0.0 if C[2] > 20.0 else 1e3)
                if best is None or score < best[0]:
                    best = (score, rvec, tvec)
            if best is not None and best[0] < 1e3:
                return True, best[1], best[2]
        return False, np.zeros((3, 1)), np.zeros((3, 1))

    flags = cv2.SOLVEPNP_IPPE if n == 4 else cv2.SOLVEPNP_ITERATIVE
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
    except cv2.error:
        ok = False
        rvec = tvec = None
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    return bool(ok), rvec, tvec


def _reprojection_rmse(
    object_xyz: np.ndarray,
    image_uv: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        object_xyz.reshape(-1, 1, 3),
        rvec,
        tvec,
        K,
        np.zeros((4, 1)),
    )
    err = proj.reshape(-1, 2) - image_uv.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _expand_inliers(
    catalog: SiteCatalog,
    uv: np.ndarray,
    seed_det: list[int],
    seed_ids: list[str],
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    inlier_px: float = INLIER_PX,
) -> tuple[list[int], list[str]]:
    xyz = catalog.xyz()
    ids = catalog.ids
    proj, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, tvec, K, np.zeros((4, 1)))
    proj = proj.reshape(-1, 2)
    used_det = set(seed_det)
    used_ids = set(seed_ids)
    det_idx = list(seed_det)
    cat_ids = list(seed_ids)
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    for i, p in enumerate(uv):
        if i in used_det or not np.isfinite(p).all():
            continue
        d = np.linalg.norm(proj - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= inlier_px and ids[j] not in used_ids:
            used_det.add(i)
            used_ids.add(ids[j])
            det_idx.append(i)
            cat_ids.append(ids[j])
            id_to_row[ids[j]] = j
    return det_idx, cat_ids


def match_and_pose(
    uv: np.ndarray,
    catalog: SiteCatalog,
    index: TripletIndex,
    K: np.ndarray,
    diameters_px: np.ndarray | None = None,
    ratio_thresh: float = RATIO_L2_THRESH,
    gsd_m_per_px: float = 1.0,
) -> MatchResult | None:
    """Triangle-fingerprint match + PnP. Returns None when no consistent pose exists."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    if uv.shape[0] < 3:
        return None
    # Fingerprint the strongest detections (keeps C(n,3) small) then grow inliers.
    if diameters_px is not None and uv.shape[0] > 12:
        order = np.argsort(-np.asarray(diameters_px, dtype=np.float64))[:12]
        queries = _query_triplets(uv[order], diameters_px[order])
        remap = {i: int(order[i]) for i in range(len(order))}
        queries = [(r, (remap[a], remap[b], remap[c]), d) for r, (a, b, c), d in queries]
    else:
        queries = _query_triplets(uv, diameters_px)
    if not queries:
        return None

    q_ratios = np.stack([q[0] for q in queries], axis=0)
    dist = np.linalg.norm(q_ratios[:, None, :] - index.ratios[None, :, :], axis=2)
    nearest = np.argmin(dist, axis=1)
    nearest_d = dist[np.arange(len(queries)), nearest]
    keep = np.flatnonzero(nearest_d <= ratio_thresh)
    if keep.size == 0:
        return None
    order = keep[np.argsort(nearest_d[keep])]
    scored: list[tuple[float, int, int]] = [
        (float(nearest_d[qi]), int(qi), int(nearest[qi])) for qi in order
    ]

    catalog_xy = catalog.xy()
    best: MatchResult | None = None

    for _dist, qi, mi in scored[:20]:
        q_ratio, q_idx, q_diam = queries[qi]
        q_pts = uv[list(q_idx)]
        c_pts = catalog_xy[index.indices[mi]]
        perm, aerr = assign_triplet(
            q_pts,
            c_pts,
            query_diameters=q_diam,
            catalog_diameters=index.diameters[mi],
        )
        if perm is None:
            continue
        assigned_local = list(perm)
        obj3 = index.world[mi][assigned_local]
        img3 = q_pts
        ids3 = [index.id_triples[mi][p] for p in assigned_local]
        ok, rvec, tvec = _solve_pnp(obj3, img3, K)
        if not ok:
            continue
        det_idx, cat_ids = _expand_inliers(
            catalog, uv, list(q_idx), ids3, rvec, tvec, K
        )
        if len(cat_ids) < MIN_INLIERS:
            continue
        id_to_xyz = {c.id: np.array([c.x_m, c.y_m, c.z_m]) for c in catalog.craters}
        obj = np.stack([id_to_xyz[i] for i in cat_ids], axis=0)
        img = uv[det_idx]
        ok2, rvec2, tvec2 = _solve_pnp(obj, img, K)
        if not ok2:
            rvec2, tvec2 = rvec, tvec
        rmse_px = _reprojection_rmse(obj, img, rvec2, tvec2, K)
        R, _ = cv2.Rodrigues(rvec2)
        C = camera_center_from_rt(R, tvec2)
        if not (ALTITUDE_RANGE_M[0] <= C[2] <= ALTITUDE_RANGE_M[1]):
            continue
        nadir_deg = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))
        if nadir_deg > NADIR_AXIS_MAX_DEG:
            continue
        if rmse_px > MAX_RMSE_PX:
            continue
        gsd = max(gsd_m_per_px, 1e-6)
        if C[2] > 1.0 and K[0, 0] > 1.0:
            gsd = float(C[2] / K[0, 0])
        residual_m = rmse_px * gsd
        cand = MatchResult(
            ids=tuple(cat_ids),
            object_xyz=obj,
            image_uv=img,
            rvec=rvec2,
            tvec=tvec2,
            residual_px=rmse_px,
            residual_m=residual_m,
            inlier_count=len(cat_ids),
            camera_enu=C,
        )
        if best is None or (cand.inlier_count, -cand.residual_px) > (
            best.inlier_count,
            -best.residual_px,
        ):
            best = cand
            if cand.inlier_count >= 6 and cand.residual_px < 2.0:
                break
    return best
