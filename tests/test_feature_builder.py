"""Regression tests for feature builder and point-probability feature helpers."""

import threading

import pandas as pd
import pytest

from src.models.point_probability import build_features_from_strengths
from src.simulation.feature_builder import (
    RuntimeFeatureBuilder,
    ASSUMED_REST_DAYS,
    ELO_BASE,
)

from src.data.feature_store import MATCH_FEATURE_COLS

_ALL_H_A_COLS = [c for c in MATCH_FEATURE_COLS if c.startswith("h_") or c.startswith("a_")]


def _make_minimal_csv(tmp_path):
    """Write a minimal CSV with all required h_/a_ columns to tmp_path."""
    import os

    path = os.path.join(str(tmp_path), "test.csv")
    base = ["local", "visitante", "gana_local", "temporada"]
    cols = base + list(_ALL_H_A_COLS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        f.write(
            ",".join(["Trento", "Perugia", "1", "2024/2025"] + ["0.5"] * len(_ALL_H_A_COLS)) + "\n"
        )
    return path


class TestBuildFeaturesFromStrengths:
    """build_features_from_strengths constructs the 6-feature dict for PointProbabilityModel."""

    def test_pin_elo_diff_x200(self):
        """REGRESSION #7: elo_diff carries the *200 scaling."""
        feats = build_features_from_strengths(0.60, 0.40)
        assert feats["elo_diff"] == pytest.approx(0.20 * 200)

    @pytest.mark.parametrize(
        "home,away,expected",
        [
            (0.40, 0.60, -0.20 * 200),
            (0.50, 0.50, 0.0),
        ],
    )
    def test_elo_diff_various_strengths(self, home, away, expected):
        """elo_diff scaling holds for negative and zero diffs."""
        feats = build_features_from_strengths(home, away)
        assert feats["elo_diff"] == pytest.approx(expected)
        if expected == 0.0:
            for key, val in feats.items():
                assert val == pytest.approx(0.0), f"{key} should be 0"


class TestAssumedRestDays:
    """ASSUMED_REST_DAYS is pinned at 7 in the feature builder module."""

    def test_pin_assumed_rest_days_7(self):
        """REGRESSION #10: ASSUMED_REST_DAYS must always be 7."""
        from src.simulation.constants import AVG_POINTS_PER_SET

        assert ASSUMED_REST_DAYS == 7
        assert AVG_POINTS_PER_SET == 23.5


class TestRuntimeFeatureBuilder:
    """RuntimeFeatureBuilder construction, schema, and lock presence."""

    def test_pin_runtime_lock_present(self, tmp_path):
        """REGRESSION #8: RuntimeFeatureBuilder has a threading.Lock."""
        csv_path = _make_minimal_csv(tmp_path)
        try:
            builder = RuntimeFeatureBuilder(csv_path=csv_path)
            lock_type = type(threading.Lock())
            assert isinstance(builder._lock, lock_type)
        finally:
            import os

            os.unlink(csv_path)

    def test_build_features_returns_dataframe(self, synthetic_feature_builder):
        """build_features returns a non-empty DataFrame with expected columns."""
        df = synthetic_feature_builder.build_features("Trento", "Perugia", jornada=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # single row
        assert "elo_diff" in df.columns
        assert "h_win_rate_global" in df.columns
        assert "a_win_rate_global" in df.columns

    def test_pin_build_features_schema(self, tmp_path):
        """REGRESSION #5: build_features returns stable schema."""
        csv_path = _make_minimal_csv(tmp_path)
        try:
            builder = RuntimeFeatureBuilder(csv_path=csv_path)
            df = builder.build_features("Trento", "Perugia", jornada=1)
            assert len(df) == 1
            assert df.iloc[0]["elo_h"] == ELO_BASE
            assert df.iloc[0]["elo_a"] == ELO_BASE
            assert df.iloc[0]["diff_win_rate_global"] == 0.0
            assert df.iloc[0]["elo_diff"] == 0.0
        finally:
            import os

            os.unlink(csv_path)


class TestWinRateAsymmetry:
    """With no prior results, win_rates fall back to 0.5 and pts_fav_exp to AVG."""

    def test_no_results_defaults(self, synthetic_feature_builder):
        """Before any matches, win_rates=0.5 and pts_fav_exp=AVG_POINTS_PER_SET."""
        from src.simulation.constants import AVG_POINTS_PER_SET

        df = synthetic_feature_builder.build_features("Trento", "Perugia", jornada=1)
        assert df.iloc[0]["h_win_rate_global"] == df.iloc[0]["a_win_rate_global"] == 0.5
        assert df.iloc[0]["h_pts_fav_exp"] == df.iloc[0]["a_pts_fav_exp"] == AVG_POINTS_PER_SET

    def test_asymmetric_win_rate_after_update(self, synthetic_feature_builder):
        """After one match, win_rates diverge (home≠away)."""
        builder = synthetic_feature_builder
        # Simulate a Trento home win
        builder.update(
            "Trento",
            "Perugia",
            sets_local=3,
            sets_visitante=1,
            winner="home",
            points_local=75,
            points_visitante=60,
        )

        df = builder.build_features("Trento", "Perugia", jornada=2)
        h_wr = df.iloc[0]["h_win_rate_global"]
        a_wr = df.iloc[0]["a_win_rate_global"]
        assert h_wr == 1.0  # Trento won their only match
        assert a_wr == 0.0  # Perugia lost their only match
        # Asymmetry: home≠away
        assert h_wr != a_wr
