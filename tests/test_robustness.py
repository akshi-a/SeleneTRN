"""Synthetic attitude tests for the plane-induced robustness filter."""

from __future__ import annotations

import cv2
import numpy as np

from core.robustness import (
    NADIR_R,
    apply_homography,
    attitude_matrix,
    camera_intrinsics,
    nadir_rotation,
    normalize_centroids,
    plane_induced_homography,
    project_world_points,
    tilted_rotation,
)
from core.localization import sorted_side_ratios


def _mask_moments(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    assert len(xs) > 50
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    cx, cy = x.mean(), y.mean()
    cov_xx = np.mean((x - cx) ** 2)
    cov_yy = np.mean((y - cy) ** 2)
    cov_xy = np.mean((x - cx) * (y - cy))
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eig = np.sort(np.linalg.eigvalsh(cov))
    aspect = float(np.sqrt(eig[1] / max(eig[0], 1e-9)))
    return float(cx), float(cy), aspect


def test_homography_is_plane_induced_not_k_r_kinv() -> None:
    K = camera_intrinsics(640, 480, 500.0)
    pitch, roll = 17.0, -11.0
    H = plane_induced_homography(K, pitch, roll)

    R_n = nadir_rotation()
    R_t = tilted_rotation(pitch, roll)
    H_expected = K @ R_n @ np.linalg.inv(R_t) @ np.linalg.inv(K)
    np.testing.assert_allclose(H, H_expected, atol=1e-10)

    # The rejected draft formula: K R_att K^{-1} (and K R_t K^{-1}).
    R_att = attitude_matrix(pitch, roll)
    H_wrong_att = K @ R_att @ np.linalg.inv(K)
    H_wrong_t = K @ R_t @ np.linalg.inv(K)
    assert not np.allclose(H, H_wrong_att, atol=1e-3)
    assert not np.allclose(H, H_wrong_t, atol=1e-3)

    H0 = plane_induced_homography(K, 0.0, 0.0)
    np.testing.assert_allclose(H0, np.eye(3), atol=1e-12)


def _contour_aspect(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    cov = np.cov(np.vstack([x - x.mean(), y - y.mean()]))
    eig = np.sort(np.linalg.eigvalsh(cov))
    return float(np.sqrt(eig[1] / max(eig[0], 1e-12)))


def test_circle_becomes_ellipse_and_recovers() -> None:
    """A nadir image circle shears to an ellipse under H_n→t and round-trips."""
    w = h = 640
    K = camera_intrinsics(w, h, 320.0)
    H_t2n = plane_induced_homography(K, pitch_deg=24.0, roll_deg=16.0)
    H_n2t = np.linalg.inv(H_t2n)

    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    # Off-axis disk so the homography is not locally a similarity.
    circle = np.stack([430.0 + 70.0 * np.cos(theta), 200.0 + 70.0 * np.sin(theta)], axis=1)
    ellipse = apply_homography(circle, H_n2t)
    recovered = apply_homography(ellipse, H_t2n)
    assert np.isfinite(ellipse).all()
    assert _contour_aspect(circle) < 1.02
    assert _contour_aspect(ellipse) > 1.25
    np.testing.assert_allclose(recovered, circle, atol=1e-6)

    # Same story on a rasterized disk: tilt elongates, inverse warp restores.
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(img, (430, 200), 70, 255, thickness=-1)
    canvas = (w * 2, h * 2)
    tilted = cv2.warpPerspective(img, H_n2t, canvas, flags=cv2.INTER_NEAREST, borderValue=0)
    recovered_img = cv2.warpPerspective(
        tilted, H_t2n, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    _, _, aspect0 = _mask_moments(img > 127)
    _, _, aspect_t = _mask_moments(tilted > 127)
    assert aspect0 < 1.08
    assert aspect_t > 1.20
    orig = img > 127
    rec = recovered_img > 127
    # Nearest-neighbour rasterisation loses rim pixels; the contour round-trip
    # above is the exact inverse. Require a substantial overlap, not identity.
    iou = float((orig & rec).sum()) / float((orig | rec).sum())
    assert iou > 0.55


def test_triangle_ratios_stable_with_filter() -> None:
    K = camera_intrinsics(960, 540, 720.0)
    C = np.array([0.0, 0.0, 620.0])
    world = np.array(
        [
            [-80.0, 40.0, 0.0],
            [70.0, 55.0, 0.0],
            [10.0, -90.0, 0.0],
        ],
        dtype=np.float64,
    )
    uv_n = project_world_points(world, NADIR_R, C, K)
    assert np.isfinite(uv_n).all()

    pitch, roll = 22.0, -16.0
    R_t = tilted_rotation(pitch, roll)
    uv_t = project_world_points(world, R_t, C, K)
    assert np.isfinite(uv_t).all()

    uv_f, H = normalize_centroids(uv_t, pitch, roll, K)
    assert H.shape == (3, 3)

    r_n = sorted_side_ratios(uv_n)
    r_t = sorted_side_ratios(uv_t)
    r_f = sorted_side_ratios(uv_f)
    assert r_n is not None and r_t is not None and r_f is not None

    drift_raw = float(np.linalg.norm(r_t - r_n))
    drift_filt = float(np.linalg.norm(r_f - r_n))
    assert drift_raw > 0.02
    assert drift_filt < 5e-4
    assert drift_filt < 0.15 * drift_raw


def test_apply_homography_empty_and_roundtrip() -> None:
    K = camera_intrinsics(200, 200, 180.0)
    H = plane_induced_homography(K, 8.0, -5.0)
    empty = apply_homography(np.zeros((0, 2)), H)
    assert empty.shape == (0, 2)
    pts = np.array([[100.0, 90.0], [40.0, 150.0]])
    back = apply_homography(apply_homography(pts, H), np.linalg.inv(H))
    np.testing.assert_allclose(back, pts, atol=1e-8)
