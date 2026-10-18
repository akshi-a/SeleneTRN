from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.generate_demo import write_catalog  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_catalog() -> None:
    if not (ROOT / "data" / "catalog" / "site_craters.csv").exists():
        write_catalog(ROOT)
