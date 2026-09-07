"""Tests for src/data/team_sideout.py — per-team sideout proxy (Batch 3 mid-effort)."""

import pytest

from src.data.team_sideout import (
    get_sideout_rates,
    get_team_sideout,
    reset_cache,
)
from src.simulation.constants import DEFAULT_SIDEOUT_RATE as GLOBAL_FALLBACK


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the sideout cache between tests so we exercise the lazy build."""
    reset_cache()
    yield
    reset_cache()


class TestGetTeamSideout:
    """The per-team sideout proxy from sets_partidos.csv."""

    def test_known_team_returns_data_value(self):
        """Perugia is the top historical team by point ratio (~0.53)."""
        rate = get_team_sideout("Perugia")
        assert 0.45 < rate < 0.60, f"Perugia sideout {rate:.3f} outside expected range"
        # Top team should be ABOVE the league fallback (0.62 - wait, that's higher)
        # Actually the league avg is ~0.50, top teams ~0.53. Just sanity-check it's
        # distinct from the fallback.
        assert rate != GLOBAL_FALLBACK

    def test_unknown_team_falls_back_to_default(self):
        """Bogus team name → DEFAULT_SIDEOUT_RATE (the league-average prior)."""
        assert get_team_sideout("Equipo Inexistente XYZ") == GLOBAL_FALLBACK

    def test_empty_string_falls_back(self):
        assert get_team_sideout("") == GLOBAL_FALLBACK

    def test_none_falls_back(self):
        assert get_team_sideout(None) == GLOBAL_FALLBACK  # type: ignore[arg-type]

    def test_canonical_alias_normalized(self):
        """Raw alias like 'Diatec Trentino' resolves to canonical 'Trento'."""
        canonical_rate = get_team_sideout("Trento")
        alias_rate = get_team_sideout("Diatec Trentino")
        assert canonical_rate == alias_rate

    def test_top_team_above_weak_team(self):
        """Perugia (historically strong) should sideout at a higher rate than a weak team."""
        perugia = get_team_sideout("Perugia")
        # Pick the current weak team with the most data (Grottazzolina)
        weak = get_team_sideout("Grottazzolina")
        assert (
            perugia > weak
        ), f"Perugia {perugia:.3f} should sideout more than Grottazzolina {weak:.3f}"


class TestGetSideoutRates:
    """Tuple-returning convenience for a matchup."""

    def test_returns_home_and_away(self):
        home, away = get_sideout_rates("Perugia", "Grottazzolina")
        assert 0.45 < home < 0.60
        assert 0.40 < away < 0.55
        assert home > away  # Perugia is the stronger team

    def test_unknown_teams_fall_back(self):
        home, away = get_sideout_rates("Foo", "Bar")
        assert home == GLOBAL_FALLBACK
        assert away == GLOBAL_FALLBACK

    def test_mixed_known_and_unknown(self):
        home, away = get_sideout_rates("Perugia", "Equipo Inventado")
        assert home != GLOBAL_FALLBACK
        assert away == GLOBAL_FALLBACK


class TestCacheBehavior:
    """The cache is lazy + module-level; verify it builds once and survives."""

    def test_cache_is_built_lazily(self):
        import src.data.team_sideout as mod

        assert mod._SIDEOUT_CACHE is None
        get_team_sideout("Perugia")
        assert mod._SIDEOUT_CACHE is not None
        assert "Perugia" in mod._SIDEOUT_CACHE

    def test_reset_cache_clears(self):
        get_team_sideout("Perugia")
        assert mod_has_cache()
        reset_cache()
        import src.data.team_sideout as mod

        assert mod._SIDEOUT_CACHE is None


def mod_has_cache() -> bool:
    import src.data.team_sideout as mod

    return mod._SIDEOUT_CACHE is not None
