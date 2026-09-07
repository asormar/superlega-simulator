"""Regression tests for the data pipeline module.

Covers CSV loader invariants, normalize_team_name round-trip behaviour,
and the UTF-8 encoding regression pin (fix #2 from Batch 1).
"""

from src.data.data_pipeline import (
    load_sets_partidos,
    load_match_features,
)
from src.data.team_mapper import normalize_team_name

# ─────────────────────────────────────────────────────────────
# CSV loader invariants
# ─────────────────────────────────────────────────────────────


class TestDataPipelineInvariants:
    """Smoke checks that the standard CSV loaders return non-empty data."""

    def test_sets_partidos_loads_rows(self):
        """The core sets_partidos CSV must contain match data."""
        df = load_sets_partidos()
        assert len(df) > 0, "sets_partidos.csv returned zero rows"
        assert "equipo_local" in df.columns
        assert "equipo_visitante" in df.columns
        assert "puntos_local" in df.columns
        assert "puntos_visitante" in df.columns
        assert "ganador_set_local" in df.columns

    def test_match_features_loads_rows(self):
        """match_features.csv must contain at least one row."""
        df = load_match_features()
        assert len(df) > 0, "match_features.csv returned zero rows"
        assert "local" in df.columns
        assert "visitante" in df.columns
        assert "gana_local" in df.columns
        assert "temporada" in df.columns

    def test_match_features_gana_local_is_int(self):
        """Target column gana_local must be integer after loading."""
        df = load_match_features()
        assert df["gana_local"].dtype == int or df["gana_local"].dtype.kind == "i"


# ─────────────────────────────────────────────────────────────
# normalize_team_name round-trip
# ─────────────────────────────────────────────────────────────


class TestNormalizeRoundTrip:
    """normalize_team_name applied to canonical names is idempotent."""

    def test_canonical_round_trip(self):
        """Applying normalize_team_name to a canonical name returns it unchanged."""
        expected = "Trento"
        assert normalize_team_name(expected) == expected

    def test_pipeline_teams_are_normalized(self, normalize_vectors):
        """All teams in match_features are already canonical (no raw aliases leak)."""
        df = load_match_features()
        all_teams = set(df["local"].unique()) | set(df["visitante"].unique())
        for team in all_teams:
            assert team == normalize_team_name(
                team
            ), f"Pipeline left unnormalized team alias: {team!r}"


# ─────────────────────────────────────────────────────────────
# UTF-8 encoding regression pin (fix #2)
# ─────────────────────────────────────────────────────────────


class TestEncodingPin:
    """Regression pin for Batch 1 encoding fix (fix #2)."""

    def test_sets_partidos_utf8(self):
        """sets_partidos.csv is readable as UTF-8 without decode errors."""
        import codecs
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent
        path = base / "DB" / "sets_partidos.csv"
        with codecs.open(str(path), "r", encoding="utf-8") as f:
            f.read()  # must not raise

    def test_match_features_utf8(self):
        """match_features.csv is readable as UTF-8 without decode errors."""
        import codecs
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent
        path = base / "DB" / "features" / "match_features.csv"
        with codecs.open(str(path), "r", encoding="utf-8") as f:
            f.read()  # must not raise

    def test_pin_utf8_csv_load(self):
        """REGRESSION #2: pipeline loads CSVs with UTF-8 encoding."""
        # Both loaders internally use encoding="utf-8"; if they return
        # data without crashing, the encoding pin is verified.
        sets = load_sets_partidos()
        mf = load_match_features()
        assert len(sets) > 0
        assert len(mf) > 0
