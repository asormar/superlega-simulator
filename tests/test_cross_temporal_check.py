"""Smoke test for ``src.models.cross_temporal_check`` — verifies the report
shape and that the 2025 row is sourced from ``precision_improved.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.models.cross_temporal_check import cross_temporal_report


def test_cross_temporal_default_years_returns_table() -> None:
    """The default call (2022-2025) returns a 4-row DataFrame indexed by year."""
    df = cross_temporal_report([2022, 2023, 2024, 2025])
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == [2022, 2023, 2024, 2025]
    # Each row has the 5 standard metrics + a source column.
    for col in ("n", "logloss", "auc", "brier", "acc", "source"):
        assert col in df.columns, f"missing column {col}"
    # The 2025 row must be sourced from the canonical precision JSON.
    assert df.loc[2025, "source"] == "precision_improved.json"
    # Sanity: 2024 and 2025 metrics are similar (cross-temporal robustness).
    assert abs(df.loc[2025, "logloss"] - df.loc[2024, "logloss"]) < 0.05
    assert abs(df.loc[2025, "auc"] - df.loc[2024, "auc"]) < 0.05


def test_cross_temporal_single_year() -> None:
    """A single-year call returns a 1-row DataFrame."""
    df = cross_temporal_report([2024])
    assert len(df) == 1
    assert df.loc[2024, "source"] == "cross_temporal_check"
    assert df.loc[2024, "n"] > 0
