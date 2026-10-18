"""Subsystem 1 — Robustness Filter.

Map crater centroids observed by a tilted camera onto the equivalent nadir
image plane so triangle side-ratios stay attitude-stable.

Coordinate frames
-----------------
World / site: local ENU, Z up, ground plane Z = 0.

Camera (OpenCV pinhole): +Z optical axis, +X right, +Y down in the image.
A nadir camera therefore looks along world −Z. Its world-to-camera rotation is

    R_n = [[ 1,  0,  0],
           [ 0, −1,  0],
           [ 0,  0, −1]]

Attitude uses the aerospace 3-2-1 convention with yaw fixed at 0 for the demo:
pitch about body-Y, then roll about body-X. Body axes after the nadir alignment
coincide with the camera axes, so

    R_att = R_x(roll) @ R_y(pitch)
    R_t   = R_att @ R_n

Plane-induced homography
------------------------
The two cameras share an optical center (attitude-only). Image points of the
ground plane are then related by a homography of the relative rotation:

    H_{t→n} = K_n  R_n  R_t^{-1}  K_t^{-1}

This is **not** ``H = K R K^{-1}``. That formula is a calibrated-ray rotation
with a single matrix ``R``. It is the wrong production model here because:

1. The nadir and tilted cameras have distinct world-to-camera rotations; the
   relative rotation is ``R_n R_t^{-1}``, not the vehicle attitude ``R_att``
   by itself (and not ``R_t``).
2. Dropping ``R_n`` and plugging aerospace ``R_x R_y`` into ``K R K^{-1}``
   rotates about the wrong axes relative to the OpenCV camera frame.
3. If the cameras did *not* share a center, the plane-induced homography would
   be ``K' (R − t nᵀ / d) K^{-1}``; the extra term vanishes only for t = 0,
   which is the same-center case already captured by (1).
"""

from __future__ import annotations

import numpy as np

# World-to-camera rotation of a nadir (optical axis = world −Z) OpenCV camera.
NADIR_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def camera_intrinsics(
    width: int,
    height: int,
    focal_px: float | None = None,
) -> np.ndarray:
    """Pinhole ``K`` with principal point at the image centre."""
    f = float(focal_px if focal_px is not None else max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rx(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ry(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def attitude_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Aerospace 3-2-1 with yaw = 0: ``R_x(roll) @ R_y(pitch)``."""
    return _rx(np.deg2rad(roll_deg)) @ _ry(np.deg2rad(pitch_deg))


def nadir_rotation() -> np.ndarray:
    return NADIR_R.copy()


def tilted_rotation(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """World-to-camera rotation of the pitched/rolled camera."""
    return attitude_matrix(pitch_deg, roll_deg) @ NADIR_R


def plane_induced_homography(
    K_t: np.ndarray,
    pitch_deg: float,
    roll_deg: float,
    K_n: np.ndarray | None = None,
) -> np.ndarray:
    """Homography mapping tilted-camera pixels onto the nadir image plane.

    ``H = K_n @ R_n @ inv(R_t) @ inv(K_t)``.
    When pitch = roll = 0 this is the identity (for ``K_n = K_t``).
    """
    K_n = K_t if K_n is None else K_n
    R_n = NADIR_R
    R_t = tilted_rotation(pitch_deg, roll_deg)
    # H_{t→n} = K_n R_n R_t^{-1} K_t^{-1}
    H = K_n @ R_n @ np.linalg.inv(R_t) @ np.linalg.inv(K_t)
    return H


def apply_homography(points_uv: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project 2-D points with a 3×3 homography. Empty input stays empty."""
    pts = np.asarray(points_uv, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.atleast_2d(pts)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts[:, :2], ones]) @ H.T
    w = hom[:, 2:3]
    valid = np.abs(w) > 1e-12
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid[:, 0]] = hom[valid[:, 0], :2] / w[valid[:, 0]]
    return uv


def normalize_centroids(
    centroids_uv: np.ndarray,
    pitch_deg: float,
    roll_deg: float,
    K_t: np.ndarray,
    K_n: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map tilted-camera centroids onto the nadir plane.

    Returns ``(normalized_uv, H_t_to_n)``.
    """
    H = plane_induced_homography(K_t, pitch_deg, roll_deg, K_n=K_n)
    return apply_homography(centroids_uv, H), H


def project_world_points(
    world_xyz: np.ndarray,
    R_wc: np.ndarray,
    camera_center: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Perspective project ``X_cam = R (X − C)``; points behind the camera are NaN."""
    X = np.atleast_2d(np.asarray(world_xyz, dtype=np.float64))
    C = np.asarray(camera_center, dtype=np.float64).reshape(3)
    Xc = (R_wc @ (X[:, :3] - C).T).T
    uv = np.full((X.shape[0], 2), np.nan, dtype=np.float64)
    z = Xc[:, 2]
    ok = z > 1e-9
    pix = (K @ Xc[ok].T).T
    uv[ok] = pix[:, :2] / pix[:, 2:3]
    return uv


def camera_center_from_rt(R_wc: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """``C = −Rᵀ t`` for ``X_cam = R X + t``."""
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (-R_wc.T @ t).reshape(3)
