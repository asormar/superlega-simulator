"""
REQ-026 regression tests for T-1 roster continuity.

Anchored on known T-1 figures from the real dataset. Truth-table values
are the continuity output of ``compute_roster_continuity`` from the
production player_stats file (1820 player-season rows across 22 teams
and 10 seasons).

Key guardrails:
  - ``test_id_mapping_coverage``: ensures all 22 ID_EQUIPO codes are
    mappable and that the Q4 fix produces non-zero continuity for >=5
    key teams (was only 3 pre-fix, per design #214 Amd.1).
  - ``test_apg_continuity``: records the APG (Perugia) 2025/26 continuity
    value; if drift exceeds 0.01, the roster data must have changed.
  - ``test_bastia_continuity``: same for BASTIA (Grottazzolina).
  - ``test_filename_regression``: ensures the code keys on ID_EQUIPO,
    NOT on the misleading filenames (Perugia_historial_10_años.csv IS
    BASTIA data; Grottazzolina_historial_10_años.csv IS APG data).
"""

from src.data.roster_continuity import compute_roster_continuity
from src.data.team_id_mapper import get_canonical_team, ID_EQUIPO_MAP
from src.data.data_pipeline import load_player_stats

# ─────────────────────────────────────────────────────────────
# Shared fixture: real data is expensive (1820 rows parsed from CSV)
# so we load once per module session.
# ─────────────────────────────────────────────────────────────


def _real_player_stats():
    """Load the production player_stats once and cache in the module."""
    if not hasattr(_real_player_stats, "_cache"):
        _real_player_stats._cache = load_player_stats()
    return _real_player_stats._cache


def _real_continuity():
    """Compute roster continuity once and cache."""
    if not hasattr(_real_continuity, "_cache"):
        ps = _real_player_stats()
        _real_continuity._cache = compute_roster_continuity(ps)
    return _real_continuity._cache


# ─────────────────────────────────────────────────────────────
# Truth-table values (computed 2026-07-23 against production data)
# ─────────────────────────────────────────────────────────────
# These values are the output of compute_roster_continuity on the
# production DB/stats_por_equipo_completo/ dataset. They encode the
# exact T-1 roster continuity for key teams. If the underlying
# roster data changes, these values must be updated.
#
# Source: ``python -m src.data.roster_continuity --dry-run`` output.

_APG_2025_CONTINUITY = 0.5142
_BASTIA_2025_CONTINUITY = 0.9497


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────


class TestIdMappingCoverage:
    """REQ-026 §1: ID-mapping coverage + Q4 fix verification."""

    def test_all_ids_map_to_canonical(self):
        """GIVEN all 22 ID_EQUIPO_MAP codes
        WHEN passed to get_canonical_team
        THEN every one returns a non-None canonical name."""
        for eid in ID_EQUIPO_MAP:
            name = get_canonical_team(eid)
            assert name is not None, f"{eid} returned None"
            assert isinstance(name, str), f"{eid} returned non-string: {type(name)}"
            assert len(name) > 0, f"{eid} returned empty string"

    def test_lt_correction(self):
        """LT → 'Cisterna Top Volley' (not just 'Cisterna')."""
        assert get_canonical_team("LT") == "Cisterna Top Volley"

    def test_pc_correction(self):
        """PC → 'Piacenza Copra' (not just 'Piacenza')."""
        assert get_canonical_team("PC") == "Piacenza Copra"

    def test_modern_teams_distinct(self):
        """CIS-VOLLEY stays 'Cisterna'; PIACENZAYOU stays 'Piacenza'."""
        assert get_canonical_team("CIS-VOLLEY") == "Cisterna"
        assert get_canonical_team("PIACENZAYOU") == "Piacenza"

    def test_unknown_id_returns_none(self):
        """An unknown ID returns None (not a misleading passthrough)."""
        assert get_canonical_team("UNKNOWN") is None
        assert get_canonical_team("") is None

    def test_q4_fix_produces_non_zero_continuity(self):
        """GIVEN the Q4 fix (team_id_mapper applied before normalize_team_name)
        WHEN continuity is computed for all 22 teams
        THEN >=5 of {Modena, Trento, Perugia, Siena, Cisterna} have non-zero
        continuity in >=1 season (was only 3 pre-fix per design #214 Amd.1)."""
        result = _real_continuity()
        # Filter to non-imputed rows only (computed continuity, not league median)
        non_imputed = result[~result["imputed"]]

        # The key teams that were broken pre-fix
        key_teams_ids = {
            "MO",  # Modena
            "TN-ITAS",  # Trento
            "APG",  # Perugia (new team, only 2024+)
            "SIENA-EMMAS",  # Siena
            "CIS-VOLLEY",  # Cisterna (modern)
            "LT",  # Cisterna Top Volley (historical)
        }

        # For each team ID, check if it has any non-zero continuity rows
        non_zero_teams = set()
        for eid in key_teams_ids:
            team_rows = non_imputed[non_imputed.index.get_level_values("equipo_id") == eid]
            if team_rows["continuity"].gt(0).any():
                non_zero_teams.add(eid)

        assert (
            len(non_zero_teams) >= 5
        ), f"Only {len(non_zero_teams)}/{len(key_teams_ids)} key teams have non-zero continuity: {non_zero_teams}"


class TestApgContinuity:
    """REQ-026 §2: APG (Perugia) truth-table."""

    def test_apg_2025_continuity(self):
        """GIVEN APG 2024/25 roster + puntos
        WHEN continuity computed for 2025/26
        THEN the value equals the known truth-table value."""
        result = _real_continuity()
        value = result.loc[("APG", "2025/2026"), "continuity"]
        assert (
            abs(value - _APG_2025_CONTINUITY) < 0.01
        ), f"APG 2025/2026 continuity drifted: {value:.4f} (expected {_APG_2025_CONTINUITY})"

    def test_apg_2024_is_imputed(self):
        """APG's first season (2024/2025) should be imputed (no T-1 data)."""
        result = _real_continuity()
        assert result.loc[("APG", "2024/2025"), "imputed"]


class TestBastiaContinuity:
    """REQ-026 §3: BASTIA (Grottazzolina) truth-table."""

    def test_bastia_2025_continuity(self):
        """GIVEN BASTIA T-1 roster
        WHEN continuity computed
        THEN the value equals the known truth-table value."""
        result = _real_continuity()
        value = result.loc[("BASTIA", "2025/2026"), "continuity"]
        assert (
            abs(value - _BASTIA_2025_CONTINUITY) < 0.01
        ), f"BASTIA 2025/2026 continuity drifted: {value:.4f} (expected {_BASTIA_2025_CONTINUITY})"

    def test_bastia_continuity_trend(self):
        """BASTIA continuity should be reasonably high (stable roster)."""
        result = _real_continuity()
        bastia_rows = result[result.index.get_level_values("equipo_id") == "BASTIA"]
        non_imp = bastia_rows[~bastia_rows["imputed"]]
        assert non_imp["continuity"].mean() > 0.50, "BASTIA should have moderate continuity"

    def test_bastia_first_season_imputed(self):
        """BASTIA's earliest season should be imputed."""
        result = _real_continuity()
        first = result[result.index.get_level_values("equipo_id") == "BASTIA"].iloc[0]
        assert first["imputed"], "BASTIA first season should be imputed"


class TestFilenameRegression:
    """REQ-026 §4: Guards the Perugia/Grottazzolina filename swap landmine.

    ``DB/stats_por_equipo_completo/Perugia_historial_10_años.csv`` actually
    holds BASTIA (Grottazzolina) data in ALL 10 seasons.
    ``...Grottazzolina_historial_10_años.csv`` actually holds APG (Perugia) data.

    A naive implementation keying on CSV filename instead of ``ID_Equipo``
    would produce the WRONG continuity values. This test ensures the correct
    ID_EQUIPO-based join is proven by asserting APG and BASTIA values are
    recognisably distinct.
    """

    def test_apg_and_bastia_are_distinct(self):
        """APG and BASTIA continuity should differ (they are different teams)."""
        result = _real_continuity()

        # Get all non-imputed continuity values for both teams across all seasons
        apg = result[result.index.get_level_values("equipo_id") == "APG"]
        bastia = result[result.index.get_level_values("equipo_id") == "BASTIA"]

        apg_vals = apg[~apg["imputed"]]["continuity"]
        bastia_vals = bastia[~bastia["imputed"]]["continuity"]

        # Compute the means — they must be different
        assert len(apg_vals) > 0, "APG should have at least 1 computed row"
        assert len(bastia_vals) > 0, "BASTIA should have at least 1 computed row"

        apg_mean = apg_vals.mean()
        bastia_mean = bastia_vals.mean()

        assert (
            abs(apg_mean - bastia_mean) > 0.05
        ), f"APG ({apg_mean:.4f}) and BASTIA ({bastia_mean:.4f}) look too similar — filename swap regression?"


class TestContinuityInvariants:
    """General invariants that must always hold."""

    def test_continuity_range(self):
        """All continuity values are in [0.0, 1.0]."""
        result = _real_continuity()
        vals = result["continuity"].dropna()
        assert vals.min() >= 0.0, f"Negative continuity: {vals.min()}"
        assert vals.max() <= 1.0, f"Continuity > 1.0: {vals.max()}"

    def test_imputed_rows_have_league_median(self):
        """Imputed rows carry the league median, not zero."""
        result = _real_continuity()
        imputed = result[result["imputed"]]
        non_imp = result[~result["imputed"]]["continuity"].dropna()
        if len(non_imp) > 0 and len(imputed) > 0:
            league_median = non_imp.median()
            # Every imputed row should have the same median value
            assert (
                imputed["continuity"].nunique() == 1
            ), "Not all imputed rows share the same league median"
            assert (
                abs(imputed["continuity"].iloc[0] - league_median) < 0.001
            ), "Imputed value != league median"

    def test_ligue_median_reasonable(self):
        """League median should be in a plausible range (not 0, not 1)."""
        result = _real_continuity()
        non_imp = result[~result["imputed"]]["continuity"].dropna()
        median = non_imp.median()
        assert 0.1 <= median <= 0.9, f"League median {median:.4f} outside [0.1, 0.9]"
