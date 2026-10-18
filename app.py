"""SeleneTRN lunar-ops flight deck. Precomputed detections; live geometry only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.engine import SEARCH_RESIDUAL_M, Engine, NavSolution
from inference.generate_demo import generate_all
from inference.load_fixtures import FixtureRun, load_fixture_run

ROOT = Path(__file__).resolve().parent
RAW_BGR = (116, 165, 196)
CORR_BGR = (200, 200, 126)
ENVELOPE_M = 5.0


def _inject_css() -> None:
    css = (ROOT / "assets" / "styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _ensure_assets() -> None:
    media = ROOT / "data" / "media" / "demo_descent.mp4"
    meta = ROOT / "data" / "fixtures" / "demo_run" / "meta.json"
    cat = ROOT / "data" / "catalog" / "site_craters.csv"
    if not (media.exists() and meta.exists() and cat.exists()):
        generate_all(ROOT)


@st.cache_resource(show_spinner=False)
def _load_run() -> FixtureRun:
    _ensure_assets()
    return load_fixture_run()


@st.cache_resource(show_spinner=False)
def _load_frames(n_frames: int) -> list[np.ndarray]:
    _ensure_assets()
    path = ROOT / "data" / "media" / "demo_descent.mp4"
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) >= n_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to read any frames from {path}")
    return frames


@st.cache_resource(show_spinner=False)
def _make_engine(width: int, height: int, focal_px: float, nadir: bool) -> Engine:
    return Engine(
        width=width,
        height=height,
        focal_px=focal_px,
        fixtures_are_nadir=nadir,
    )


def _draw_quad(img: np.ndarray, quad: np.ndarray, color: tuple[int, int, int], w: int) -> None:
    pts = np.round(quad).astype(np.int32)
    if not np.isfinite(quad).all():
        return
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=w)


def _draw_cross(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], s: int = 6) -> None:
    if not np.isfinite(uv).all():
        return
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.line(img, (u - s, v), (u + s, v), color, 1, cv2.LINE_AA)
    cv2.line(img, (u, v - s), (u, v + s), color, 1, cv2.LINE_AA)


def render_optical(frame_bgr: np.ndarray, sol: NavSolution) -> np.ndarray:
    vis = frame_bgr.copy()
    if sol.box_quads_observed is not None:
        for q in sol.box_quads_observed:
            _draw_quad(vis, q, RAW_BGR, 1)
    if sol.box_quads_corrected is not None:
        for q in sol.box_quads_corrected:
            _draw_quad(vis, q, CORR_BGR, 1)
    for p in sol.observed_uv:
        _draw_cross(vis, p, RAW_BGR, 5)
    for p in sol.corrected_uv:
        if not np.isfinite(p).all():
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (u, v), 5, CORR_BGR, 1, cv2.LINE_AA)

    cv2.rectangle(vis, (12, 12), (210, 52), (11, 13, 16), -1)
    cv2.putText(vis, "RAW", (20, 38), cv2.FONT_HERSHEY_PLAIN, 1.1, RAW_BGR, 1, cv2.LINE_AA)
    cv2.putText(vis, "CORRECTED", (78, 38), cv2.FONT_HERSHEY_PLAIN, 1.1, CORR_BGR, 1, cv2.LINE_AA)
    return vis


def _fmt(value: float, spec: str, unit: str = "", empty: str = "—") -> str:
    if value is None or not np.isfinite(value):
        return empty
    text = format(value, spec)
    if unit:
        return f'{text}<span class="unit">{unit}</span>'
    return text


def _fmt_residual(sol: NavSolution, empty: str = "—") -> str:
    if sol.status == "SEARCH":
        return empty
    if not np.isfinite(sol.residual_m) or sol.residual_m >= SEARCH_RESIDUAL_M:
        return empty
    return _fmt(sol.residual_m, ".2f", "m")


def residual_series(
    engine: Engine,
    run: FixtureRun,
    pitch: float,
    roll: float,
    use_filter: bool,
) -> tuple[list[int], list[float], list[str]]:
    engine.reset_coast()
    xs: list[int] = []
    ys: list[float] = []
    stt: list[str] = []
    for fr in run.frames:
        sol = engine.process_frame(
            fr.centroids,
            pitch,
            roll,
            use_filter=use_filter,
            boxes=fr.boxes,
            gsd_m_per_px=run.gsd_m_per_px,
        )
        xs.append(fr.frame_id)
        ys.append(float(sol.residual_m) if np.isfinite(sol.residual_m) else 40.0)
        stt.append(sol.status)
    engine.reset_coast()
    return xs, ys, stt


def residual_figure(xs: list[int], ys: list[float], current: int) -> go.Figure:
    clipped = [min(y, 40.0) for y in ys]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=ENVELOPE_M, fillcolor="rgba(126,200,200,0.07)", line_width=0)
    fig.add_hline(
        y=ENVELOPE_M,
        line_dash="dot",
        line_color="#7EC8C8",
        annotation_text="5 m envelope",
        annotation_font_color="#7EC8C8",
        annotation_font_family="IBM Plex Mono",
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=clipped,
            mode="lines",
            line=dict(color="#7EC8C8", width=2),
            name="reproj. RMSE",
            hovertemplate="frame %{x}<br>%{y:.2f} m<extra></extra>",
        )
    )
    if current in xs:
        fig.add_trace(
            go.Scatter(
                x=[current],
                y=[clipped[xs.index(current)]],
                mode="markers",
                marker=dict(size=9, color="#e8ecec", line=dict(color="#7EC8C8", width=1)),
                name="now",
                showlegend=False,
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,13,16,0.35)",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        margin=dict(l=44, r=12, t=28, b=40),
        height=260,
        xaxis=dict(title="frame", gridcolor="rgba(126,200,200,0.08)", zeroline=False),
        yaxis=dict(
            title="residual (m)",
            gridcolor="rgba(126,200,200,0.08)",
            zeroline=False,
            range=[0, max(12.0, min(40.0, max(clipped + [6.0]) * 1.15))],
        ),
        showlegend=False,
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="SeleneTRN",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()
    run = _load_run()
    frames = _load_frames(run.n_frames)
    engine = _make_engine(run.width, run.height, run.focal_px, run.fixtures_are_nadir)

    st.session_state.setdefault("pitch", 0.0)
    st.session_state.setdefault("roll", 0.0)
    st.session_state.setdefault("use_filter", True)
    st.session_state.setdefault("frame_i", run.n_frames // 2)

    pitch = float(st.session_state["pitch"])
    roll = float(st.session_state["roll"])
    use_filter = bool(st.session_state["use_filter"])
    frame_i = int(st.session_state["frame_i"])
    fr = run.frame(min(max(frame_i, 0), len(run.frames) - 1))

    key = (round(pitch, 2), round(roll, 2), bool(use_filter))
    if st.session_state.get("_series_key") != key:
        xs, ys, _st = residual_series(engine, run, pitch, roll, use_filter)
        st.session_state["_series_key"] = key
        st.session_state["_series"] = (xs, ys)
    xs, ys = st.session_state["_series"]

    sol = engine.process_frame(
        fr.centroids,
        pitch,
        roll,
        use_filter=use_filter,
        boxes=fr.boxes,
        gsd_m_per_px=run.gsd_m_per_px,
    )

    alert = ""
    if sol.status == "SEARCH" or (
        np.isfinite(sol.residual_m) and sol.residual_m > ENVELOPE_M and sol.status != "LOCK"
    ):
        alert = " status-alert"

    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="brand-name">SeleneTRN</div>
            <div class="brand-sub">GPS-denied terrain-relative navigation · {run.meta.get("site_id", "demo")}</div>
          </div>
          <div class="status-cluster">
            <div class="status-word status-{sol.status}{alert}">{sol.status}</div>
            <div class="lat-read">GEOM <b>{sol.latency_ms:.2f} ms</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        st.markdown('<div class="panel-label">OPTICAL PLANE</div>', unsafe_allow_html=True)
        idx = min(max(frame_i, 0), len(frames) - 1)
        optical = render_optical(frames[idx], sol)
        st.markdown('<div class="optical-wrap">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(optical, cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        st.slider("FRAME", min_value=0, max_value=max(run.n_frames - 1, 0), key="frame_i")
        c1, c2, c3 = st.columns([1, 1, 0.85])
        with c1:
            st.slider("PITCH  deg", min_value=-30.0, max_value=30.0, step=0.5, key="pitch")
        with c2:
            st.slider("ROLL  deg", min_value=-30.0, max_value=30.0, step=0.5, key="roll")
        with c3:
            st.write("")
            st.checkbox("ROBUSTNESS FILTER", key="use_filter")

    with right:
        st.markdown(
            f"""
            <div class="telem-grid">
              <div class="telem-item">
                <div class="label">ALTITUDE</div>
                <div class="value">{_fmt(sol.altitude_m, ".1f", "m")}</div>
              </div>
              <div class="telem-item">
                <div class="label">RESIDUAL</div>
                <div class="value">{_fmt_residual(sol)}</div>
              </div>
              <div class="telem-item">
                <div class="label">LAT</div>
                <div class="value">{_fmt(sol.lat, ".5f", "°")}</div>
              </div>
              <div class="telem-item">
                <div class="label">LON</div>
                <div class="value">{_fmt(sol.lon, ".5f", "°")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="panel-label">REPROJECTION RMSE</div>', unsafe_allow_html=True)
        st.plotly_chart(
            residual_figure(xs, ys, frame_i),
            width="stretch",
            config={"displayModeBar": False},
        )
        ids = "  ".join(sol.matched_ids) if sol.matched_ids else "—"
        st.markdown('<div class="panel-label">MATCHED CRATERS</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ids-list">{ids}<br><span style="color:#8b949e">{sol.match_count} inliers</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="footer-src">detections: {run.model_id}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
