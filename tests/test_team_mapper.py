"""Regression tests for team name normalization and magic constants.

Constants from multiple source modules are folded into this file to stay
under the 800-line review budget (spec Requirement 11).  Duplication
between source modules is FLAGGED by assertion but NOT fixed here
(batch scope: out-of-scope).
"""

from src.data.team_mapper import normalize_team_name, get_all_viable_teams

# ─────────────────────────────────────────────────────────────
# Normalize vectors
# ─────────────────────────────────────────────────────────────


class TestNormalizeVectors:
    """Canonical normalize_team_name examples that must not regress."""

    def test_monza_monza(self):
        assert normalize_team_name("MonzaMonza") == "Monza"

    def test_diatec_trentino(self):
        assert normalize_team_name("Diatec Trentino") == "Trento"

    def test_sir_safety_conad_perugia(self):
        assert normalize_team_name("Sir Safety Conad Perugia") == "Perugia"

    def test_azimut_modena(self):
        assert normalize_team_name("Azimut Modena") == "Modena"

    def test_kioene_padova_whitespace(self):
        assert normalize_team_name("  Kioene Padova  ") == "Padova"

    def test_gi_group_monza(self):
        assert normalize_team_name("Gi Group Monza") == "Monza"

    def test_verona_duplicated(self):
        assert normalize_team_name("VeronaVerona") == "Verona"

    def test_grottazzolina_duplicated(self):
        assert normalize_team_name("GrottazzolinaGrottazzolina") == "Grottazzolina"

    def test_vibo_valentia_duplicated(self):
        assert normalize_team_name("Vibo ValentiaVibo Valentia") == "Vibo Valentia"

    def test_emma_villas_siena(self):
        assert normalize_team_name("Emma Villas Siena") == "Siena"

    def test_videx_grottazzolina(self):
        assert normalize_team_name("Videx Grottazzolina") == "Grottazzolina"

    def test_lube_civitanova_duplicated(self):
        assert normalize_team_name("Lube CivitanovaLube Civitanova") == "Lube"


class TestViableTeamsThreshold:
    """get_all_viable_teams returns a non-empty list with stable contents."""

    def test_viable_teams_non_empty(self):
        teams = get_all_viable_teams()
        assert len(teams) >= 12  # SuperLega has 12 teams

    def test_viable_teams_contain_trento_perugia(self):
        names = {t["nombre"] for t in get_all_viable_teams()}
        assert "Trento" in names
        assert "Perugia" in names

    def test_viable_teams_include_historic(self):
        names = {t["nombre"] for t in get_all_viable_teams()}
        # Siena is a historic team, not in current season
        assert "Siena" in names

    def test_viable_teams_structure(self):
        for entry in get_all_viable_teams():
            assert "nombre" in entry
            assert "categoria" in entry
            assert entry["categoria"] in ("actual", "historico")


# ─────────────────────────────────────────────────────────────
# Constants pinning (folded — no separate test_constants.py)
# ─────────────────────────────────────────────────────────────


class TestDefaultSideoutRate:
    """DEFAULT_SIDEOUT_RATE centralized in src.simulation.constants (was duplicated in point_probability.py before Batch 3)."""

    def test_constants_module_sideout(self):
        from src.simulation.constants import DEFAULT_SIDEOUT_RATE

        assert DEFAULT_SIDEOUT_RATE == 0.62

    def test_point_probability_no_longer_has_class_attr(self):
        """The class-level duplicate was removed; only the central constant remains."""
        from src.models.point_probability import PointProbabilityModel

        assert not hasattr(
            PointProbabilityModel, "DEFAULT_SIDEOUT_RATE"
        ), "DEFAULT_SIDEOUT_RATE was deduplicated; the class attribute should be removed."


class TestClampRanges:
    """Clamp constants from the simulation module."""

    def test_default_clamp_range(self):
        from src.simulation.constants import DEFAULT_CLAMP_RANGE

        assert DEFAULT_CLAMP_RANGE == (0.20, 0.80)

    def test_adaptive_clamp_hard(self):
        from src.simulation.constants import POINT_PROB_CLIP_ADAPTIVE_HARD

        assert POINT_PROB_CLIP_ADAPTIVE_HARD == (0.10, 0.90)

    def test_clamp_margin(self):
        from src.simulation.constants import CLAMP_MARGIN

        assert CLAMP_MARGIN == 0.20


class TestMomentumConstants:
    """Momentum params — changing these shifts every simulation outcome."""

    def test_momentum_bonus(self):
        from src.simulation.constants import MOMENTUM_BONUS

        assert MOMENTUM_BONUS == 0.015

    def test_momentum_max_streak(self):
        from src.simulation.constants import MOMENTUM_MAX_STREAK

        assert MOMENTUM_MAX_STREAK == 4

    def test_momentum_decay(self):
        from src.simulation.constants import MOMENTUM_DECAY

        assert MOMENTUM_DECAY == 0.5

    def test_simulator_class_attrs_match(self):
        """MatchSimulator re-exports MOMENTUM_* from constants.py."""
        from src.simulation.simulator import MatchSimulator
        from src.simulation.constants import (
            MOMENTUM_BONUS,
            MOMENTUM_MAX_STREAK,
            MOMENTUM_DECAY,
        )

        assert MatchSimulator.MOMENTUM_BONUS == MOMENTUM_BONUS
        assert MatchSimulator.MOMENTUM_MAX_STREAK == MOMENTUM_MAX_STREAK
        assert MatchSimulator.MOMENTUM_DECAY == MOMENTUM_DECAY


class TestAssumedRestDays:
    """ASSUMED_REST_DAYS is a key feature-building constant."""

    def test_assumed_rest_days_value(self):
        from src.simulation.feature_builder import ASSUMED_REST_DAYS

        assert ASSUMED_REST_DAYS == 7
