"""Landing-site crater catalog (demo-consistent ENU + lat/lon, not a geodetic product)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "catalog" / "site_craters.csv"
DEFAULT_META = REPO_ROOT / "data" / "catalog" / "site_meta.json"


@dataclass(frozen=True)
class Crater:
    id: str
    x_m: float
    y_m: float
    z_m: float
    lat: float
    lon: float
    diameter_m: float


@dataclass
class SiteCatalog:
    site_id: str
    origin_lat_deg: float
    origin_lon_deg: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float
    craters: list[Crater]
    gsd_m_per_px_ref: float = 1.0
    description: str = ""

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.craters]

    def xyz(self) -> np.ndarray:
        return np.array([[c.x_m, c.y_m, c.z_m] for c in self.craters], dtype=np.float64)

    def xy(self) -> np.ndarray:
        return np.array([[c.x_m, c.y_m] for c in self.craters], dtype=np.float64)

    def diameters(self) -> np.ndarray:
        return np.array([c.diameter_m for c in self.craters], dtype=np.float64)

    def enu_to_latlon(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Local tangent-plane conversion using site meters-per-degree (not a global 111000)."""
        lat = self.origin_lat_deg + north_m / self.meters_per_deg_lat
        lon = self.origin_lon_deg + east_m / self.meters_per_deg_lon
        return float(lat), float(lon)

    def crater_by_id(self, crater_id: str) -> Crater:
        for c in self.craters:
            if c.id == crater_id:
                return c
        raise KeyError(crater_id)


def meters_per_deg_lon(lat_deg: float, meters_per_deg_lat: float) -> float:
    return float(meters_per_deg_lat * np.cos(np.deg2rad(lat_deg)))


def load_site_catalog(
    csv_path: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> SiteCatalog:
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    meta_path = Path(meta_path) if meta_path else DEFAULT_META
    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path)
    required = {"id", "x_m", "y_m", "z_m", "lat", "lon", "diameter_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"catalog CSV missing columns: {sorted(missing)}")
    craters = [
        Crater(
            id=str(row["id"]),
            x_m=float(row["x_m"]),
            y_m=float(row["y_m"]),
            z_m=float(row["z_m"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            diameter_m=float(row["diameter_m"]),
        )
        for row in df.to_dict(orient="records")
    ]
    return SiteCatalog(
        site_id=str(meta["site_id"]),
        origin_lat_deg=float(meta["origin_lat_deg"]),
        origin_lon_deg=float(meta["origin_lon_deg"]),
        meters_per_deg_lat=float(meta["meters_per_deg_lat"]),
        meters_per_deg_lon=float(meta["meters_per_deg_lon"]),
        craters=craters,
        gsd_m_per_px_ref=float(meta.get("gsd_m_per_px_ref", 1.0)),
        description=str(meta.get("description", "")),
    )
