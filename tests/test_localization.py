"""Catalog triangle matcher, solvePnP, residuals, and filter-on/off lock."""

from __future__ import annotations

import numpy as np
import pytest

from core.catalog import load_site_catalog
from core.engine import Engine
from core.localization import (
    RATIO_L2_THRESH,
    build_triplet_index,
    match_and_pose,
    sorted_side_ratios,
)
from core.robustness import (
    NADIR_R,
    camera_intrinsics,
    project_world_points,
    tilted_rotation,
)


def test_catalog_loads_and_enu_to_latlon() -> None:
    cat = load_site_catalog()
    assert len(cat.craters) >= 12
    lat, lon = cat.enu_to_latlon(0.0, 0.0)
    assert lat == pytest.approx(cat.origin_lat_deg)
    assert lon == pytest.approx(cat.origin_lon_deg)
    lat2, _ = cat.enu_to_latlon(0.0, cat.meters_per_deg_lat)
    assert lat2 == pytest.approx(cat.origin_lat_deg + 1.0)
    # Must not be a global 111000 dump.
    assert cat.meters_per_deg_lon != pytest.approx(111000.0, rel=0.05)
    assert cat.meters_per_deg_lon == pytest.approx(
        cat.meters_per_deg_lat * np.cos(np.deg2rad(cat.origin_lat_deg)),
        rel=1e-6,
    )


def test_triplet_index_unique_enough() -> None:
    cat = load_site_catalog()
    index = build_triplet_index(cat)
    assert len(index.id_triples) > 20
    # Fingerprints should not all collapse to one point.
    span = index.ratios.max(axis=0) - index.ratios.min(axis=0)
    assert float(span.min()) > 0.02


def test_match_and_pnp_recovers_known_nadir_pose() -> None:
    cat = load_site_catalog()
    index = build_triplet_index(cat)
    K = camera_intrinsics(960, 540, 720.0)
    C_true = np.array([12.0, -8.0, 580.0])
    uv = project_world_points(cat.xyz(), NADIR_R, C_true, K)
    vis = np.isfinite(uv).all(axis=1)
    vis &= (uv[:, 0] > 2) & (uv[:, 0] < 958) & (uv[:, 1] > 2) & (uv[:, 1] < 538)
    query = uv[vis]
    diams = (cat.diameters()[vis] / (C_true[2] / K[0, 0]))
    rng = np.random.default_rng(3)
    order = rng.permutation(len(query))
    match = match_and_pose(
        query[order],
        cat,
        index,
        K,
        diameters_px=diams[order],
        gsd_m_per_px=C_true[2] / K[0, 0],
    )
    assert match is not None
    assert match.inlier_count >= 6
    np.testing.assert_allclose(match.camera_enu, C_true, atol=3.0)
    assert match.residual_px < 1.0
    # Residual in meters is reprojection RMSE * GSD, not a made-up decay.
    gsd = match.camera_enu[2] / K[0, 0]
    assert match.residual_m == pytest.approx(match.residual_px * gsd, rel=1e-6)


def test_engine_lock_with_filter_breaks_without() -> None:
    cat = load_site_catalog()
    engine = Engine(catalog=cat, width=960, height=540, focal_px=720.0, fixtures_are_nadir=True)
    C_true = np.array([5.0, 4.0, 640.0])
    uv_n = project_world_points(cat.xyz(), NADIR_R, C_true, engine.K)
    vis = np.isfinite(uv_n).all(axis=1)
    vis &= (uv_n[:, 0] > 8) & (uv_n[:, 0] < 952) & (uv_n[:, 1] > 8) & (uv_n[:, 1] < 532)
    centroids = uv_n[vis]
    pitch, roll = 18.0, -12.0

    on = engine.process_frame(centroids, pitch, roll, use_filter=True)
    engine.reset_coast()
    off = engine.process_frame(centroids, pitch, roll, use_filter=False)

    assert on.status == "LOCK"
    assert on.match_count >= 5
    assert on.residual_m < 5.0
    assert on.altitude_m == pytest.approx(C_true[2], abs=8.0)
    lat, lon = cat.enu_to_latlon(C_true[0], C_true[1])
    assert on.lat == pytest.approx(lat, abs=5e-4)
    assert on.lon == pytest.approx(lon, abs=5e-3)

    assert off.status in {"SEARCH", "COAST"}
    assert off.match_count < on.match_count or off.residual_m > on.residual_m
    assert off.residual_m > 5.0 or off.match_count == 0


def test_zero_attitude_is_identity_and_locks() -> None:
    cat = load_site_catalog()
    engine = Engine(catalog=cat, fixtures_are_nadir=True)
    C_true = np.array([0.0, 0.0, 600.0])
    uv = project_world_points(cat.xyz(), NADIR_R, C_true, engine.K)
    vis = np.isfinite(uv).all(axis=1)
    vis &= (uv[:, 0] > 5) & (uv[:, 0] < 955) & (uv[:, 1] > 5) & (uv[:, 1] < 535)
    sol = engine.process_frame(uv[vis], 0.0, 0.0, use_filter=True)
    assert sol.status == "LOCK"
    np.testing.assert_allclose(sol.H, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(sol.observed_uv, sol.corrected_uv, atol=1e-8)


def test_true_tilted_fixtures_need_the_filter() -> None:
    """When detections already live in a tilted camera, filter maps them to nadir."""
    cat = load_site_catalog()
    engine = Engine(catalog=cat, fixtures_are_nadir=False)
    C_true = np.array([-6.0, 9.0, 610.0])
    pitch, roll = -15.0, 10.0
    R_t = tilted_rotation(pitch, roll)
    uv_t = project_world_points(cat.xyz(), R_t, C_true, engine.K)
    vis = np.isfinite(uv_t).all(axis=1)
    vis &= (uv_t[:, 0] > 10) & (uv_t[:, 0] < 950) & (uv_t[:, 1] > 10) & (uv_t[:, 1] < 530)
    on = engine.process_frame(uv_t[vis], pitch, roll, use_filter=True)
    engine.reset_coast()
    off = engine.process_frame(uv_t[vis], pitch, roll, use_filter=False)
    assert on.status == "LOCK"
    assert on.altitude_m == pytest.approx(C_true[2], abs=12.0)
    assert off.residual_m > on.residual_m


def test_demo_fixtures_lock_with_filter() -> None:
    from inference.load_fixtures import load_fixture_run

    run = load_fixture_run()
    assert run.n_frames >= 30
    engine = Engine(
        width=run.width,
        height=run.height,
        focal_px=run.focal_px,
        fixtures_are_nadir=run.fixtures_are_nadir,
    )
    fr = run.frame(run.n_frames // 2)
    on = engine.process_frame(
        fr.centroids, 20.0, -15.0, use_filter=True, boxes=fr.boxes
    )
    engine.reset_coast()
    off = engine.process_frame(
        fr.centroids, 20.0, -15.0, use_filter=False, boxes=fr.boxes
    )
    assert on.status == "LOCK"
    assert on.residual_m < 5.0
    assert on.latency_ms < 25.0
    assert off.status == "SEARCH"


def test_sorted_ratios_scale_invariant() -> None:
    pts = np.array([[0.0, 0.0], [40.0, 0.0], [10.0, 30.0]])
    r1 = sorted_side_ratios(pts)
    r2 = sorted_side_ratios(pts * 3.7 + np.array([12.0, -5.0]))
    assert r1 is not None and r2 is not None
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    assert float(np.linalg.norm(r1)) < np.sqrt(2) + 1e-9
    assert RATIO_L2_THRESH < 0.05
