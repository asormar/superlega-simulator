"""Tests for SeasonSimulator — round-robin scheduling, standings, and half flow."""

import pytest

from src.simulation.constants import (
    MATCH_PREDICTOR_DAMPING,
    adaptive_damping,
    ADAPTIVE_DAMPING_START,
    ADAPTIVE_DAMPING_END,
    SUPERLEGA_TOTAL_JORNADAS,
)
from src.simulation.season_simulator import (
    SeasonSimulator,
    TeamStanding,
)
from src.simulation.simulator import (
    MatchSimulator,
    MatchResult,
    SetResult,
)


def _make_match(
    home_team, away_team, sets_home=3, sets_away=0, winner="home", resultado="3-0", set_specs=None
):
    """Build a MatchResult from set_specs: list of (score_home, score_away, winner, home_stats, away_stats)."""
    if set_specs is None:
        set_specs = [(25, 20, "home")]
    sets = []
    for i, spec in enumerate(set_specs, 1):
        sh, sa, sw = spec[0], spec[1], spec[2]
        home_stats = spec[3] if len(spec) > 3 else []
        away_stats = spec[4] if len(spec) > 4 else []
        sets.append(
            SetResult(i, sh, sa, sw, home_player_stats=home_stats, away_player_stats=away_stats)
        )
    return MatchResult(
        home_team=home_team,
        away_team=away_team,
        sets_home=sets_home,
        sets_away=sets_away,
        winner=winner,
        resultado=resultado,
        sets=sets,
    )


class TestGenerateReturnLeg:
    """_generate_return_leg produces N*(N-1)/2 reversed pairings, no self-matches."""

    def test_return_leg_invariants(self):
        """Return leg has correct count, no self-matches, no dupes, all teams appear."""
        teams = ["Trento", "Perugia", "Monza", "Lube"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        assert len(return_leg) == len(teams) * (len(teams) - 1) // 2
        seen, appeared = set(), set()
        for home, away in return_leg:
            assert home != away
            assert (home, away) not in seen
            seen.add((home, away))
            appeared.update([home, away])
        for t in teams:
            assert t in appeared

    def test_return_leg_all_reverse_pairs(self):
        """Every return-leg match (H,A) has (A,H) absent — it's the reverse."""
        teams = ["Trento", "Perugia", "Monza"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        for home, away in return_leg:
            assert (away, home) not in return_leg


class TestAccumulatePlayerStats:
    """_accumulate_player_stats correctly accumulates stats across sets."""

    def test_same_player_two_sets_accumulates(self):
        """A player appearing in multiple sets has summed stats (not overwritten)."""
        season_sim = SeasonSimulator()
        stats = {}
        match = _make_match(
            "Trento",
            "Perugia",
            sets_home=3,
            sets_away=1,
            winner="home",
            resultado="3-1",
            set_specs=[
                (25, 20, "home", [{"jugador": "PlayerA", "puntos": 5, "aces": 1}]),
                (25, 22, "home", [{"jugador": "PlayerA", "puntos": 3, "aces": 0}]),
                (20, 25, "away", [{"jugador": "PlayerA", "puntos": 4, "aces": 0}]),
                (25, 18, "home", [{"jugador": "PlayerA", "puntos": 2, "aces": 0}]),
            ],
        )
        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        key = "Trento|PlayerA"
        assert key in stats
        assert stats[key]["puntos"] == 14  # 5+3+4+2
        assert stats[key]["aces"] == 1
        assert stats[key]["sets"] == 4
        assert stats[key]["partidos"] == 1

    def test_two_players_accumulated_separately(self):
        """Two different players in the same match have separate stats."""
        season_sim = SeasonSimulator()
        stats = {}
        match = _make_match(
            "Trento",
            "Perugia",
            set_specs=[
                (
                    25,
                    18,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 5}, {"jugador": "PlayerB", "puntos": 3}],
                ),
                (
                    25,
                    20,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 4}, {"jugador": "PlayerB", "puntos": 6}],
                ),
                (
                    25,
                    22,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 3}, {"jugador": "PlayerB", "puntos": 2}],
                ),
            ],
        )
        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        assert stats["Trento|PlayerA"]["puntos"] == 12  # 5+4+3
        assert stats["Trento|PlayerB"]["puntos"] == 11  # 3+6+2

    def test_away_side_accumulated_separately(self):
        """Away players are stored under the away team key."""
        season_sim = SeasonSimulator()
        stats = {}
        match = _make_match(
            "Trento",
            "Perugia",
            set_specs=[
                (25, 20, "home", [], [{"jugador": "OppPlayer", "puntos": 4}]),
            ],
        )
        season_sim._accumulate_player_stats(stats, match, "Perugia", "away")
        assert "Perugia|OppPlayer" in stats
        assert stats["Perugia|OppPlayer"]["puntos"] == 4

    def test_no_stats_with_empty_sets(self):
        """Sets with empty player stats produce no entries."""
        season_sim = SeasonSimulator()
        stats = {}
        match = _make_match(
            "Trento",
            "Perugia",
            set_specs=[
                (25, 20, "home"),
                (25, 22, "home"),
                (25, 18, "home"),
            ],
        )
        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        assert len(stats) == 0


class TestStandingsRoundTrip:
    """TeamStanding serialization and parsing are lossless."""

    def _sample_standings_list(self):
        entries = [
            TeamStanding(
                team="Trento",
                points=15,
                matches_played=5,
                wins=5,
                losses=0,
                sets_won=15,
                sets_lost=2,
                points_scored=400,
                points_conceded=300,
                wins_3_0=3,
                wins_3_1=1,
                wins_3_2=1,
                losses_2_3=0,
                losses_1_3=0,
                losses_0_3=0,
            ),
            TeamStanding(
                team="Perugia",
                points=12,
                matches_played=5,
                wins=4,
                losses=1,
                sets_won=13,
                sets_lost=5,
                points_scored=380,
                points_conceded=320,
                wins_3_0=2,
                wins_3_1=1,
                wins_3_2=1,
                losses_2_3=0,
                losses_1_3=1,
                losses_0_3=0,
            ),
        ]
        return entries

    def test_serialize_standings_fields(self):
        """serialize_standings produces the expected keys."""
        entries = self._sample_standings_list()
        serialized = SeasonSimulator.serialize_standings(entries)
        assert len(serialized) == 2
        for s in serialized:
            assert "equipo" in s
            assert "puntos" in s
            assert "pj" in s
            assert "pg" in s
            assert "pp" in s
            assert "sg" in s
            assert "sp" in s
            assert "sr" in s
            assert "pts_favor" in s
            assert "pts_contra" in s

    def test_parse_standings_round_trip(self):
        """parse_standings(serialize_standings(x)) recovers all fields."""
        entries = self._sample_standings_list()
        serialized = SeasonSimulator.serialize_standings(entries)
        parsed = SeasonSimulator.parse_standings(serialized)

        for original in entries:
            recovered = parsed[original.team]
            assert recovered.team == original.team
            assert recovered.points == original.points
            assert recovered.matches_played == original.matches_played
            assert recovered.wins == original.wins
            assert recovered.losses == original.losses
            assert recovered.sets_won == original.sets_won
            assert recovered.sets_lost == original.sets_lost
            assert recovered.points_scored == original.points_scored
            assert recovered.points_conceded == original.points_conceded
            assert recovered.wins_3_0 == original.wins_3_0

    def test_empty_and_none_standings(self):
        """Empty list or None handles cleanly."""
        assert SeasonSimulator.serialize_standings([]) == []
        assert SeasonSimulator.parse_standings([]) == {}
        assert SeasonSimulator.parse_standings(None) == {}


class TestTwoPassHalfFlow:
    """Two-pass season: half='first' then half='second' + first_half_state."""

    def test_two_pass_half_first_second(self):
        """Combining half='first' and half='second' produces a full round-robin."""
        teams = ["Trento", "Perugia", "Monza", "Lube", "Milano", "Verona"]
        sim = SeasonSimulator(simulator=MatchSimulator(), team_strengths={t: 0.5 for t in teams})
        first = sim.simulate_season(teams=teams, double_round_robin=True, seed=42, half="first")
        second = sim.simulate_season(
            teams=teams,
            double_round_robin=True,
            seed=42,
            half="second",
            first_half_state={
                "standings": SeasonSimulator.serialize_standings(first["standings"]),
                "player_season_stats": first.get("player_season_stats", {}),
            },
        )
        n = len(teams)
        assert len(first["matches"]) == n * (n - 1) // 2
        assert len(second["matches"]) == n * (n - 1) // 2
        second_st = {s.team: s for s in second["standings"]}
        for t in teams:
            assert second_st[t].matches_played == (n - 1) * 2


class TestPlayerStatsCollision:
    """Regression pin for player-stats collision bug (N4/N2)."""

    def test_pin_player_stats_no_collision(self):
        """REGRESSION N4: player stats accumulate without collision across matches.

        Two different matches with overlapping player names must not corrupt
        each other's accumulated stats.
        """
        season_sim = SeasonSimulator()
        stats = {}
        match1 = _make_match(
            "Trento",
            "Perugia",
            set_specs=[
                (
                    25,
                    20,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 5}],
                    [{"jugador": "OppA", "puntos": 4}],
                ),
                (
                    25,
                    22,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 3}],
                    [{"jugador": "OppA", "puntos": 4}],
                ),
                (
                    25,
                    18,
                    "home",
                    [{"jugador": "PlayerA", "puntos": 2}],
                    [{"jugador": "OppA", "puntos": 5}],
                ),
            ],
        )
        season_sim._accumulate_player_stats(stats, match1, "Trento", "home")
        season_sim._accumulate_player_stats(stats, match1, "Perugia", "away")
        match2 = _make_match(
            "Monza",
            "Verona",
            sets_home=3,
            sets_away=1,
            winner="home",
            resultado="3-1",
            set_specs=[
                (25, 21, "home", [{"jugador": "PlayerA", "puntos": 4}]),
                (23, 25, "away", [{"jugador": "PlayerA", "puntos": 3}]),
                (25, 19, "home", [{"jugador": "PlayerA", "puntos": 5}]),
                (25, 20, "home", [{"jugador": "PlayerA", "puntos": 2}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match2, "Monza", "home")

        # Trento's PlayerA must have 10 points (5+3+2), unchanged by Monza's PlayerA
        assert stats["Trento|PlayerA"]["puntos"] == 10
        assert stats["Trento|PlayerA"]["partidos"] == 1
        assert stats["Trento|PlayerA"]["sets"] == 3

        # Monza's PlayerA must have 14 points (4+3+5+2)
        assert stats["Monza|PlayerA"]["puntos"] == 14
        assert stats["Monza|PlayerA"]["partidos"] == 1
        assert stats["Monza|PlayerA"]["sets"] == 4

        # Both are separate entries — no collision
        assert len(stats) == 3  # Trento|PlayerA, Perugia|OppA, Monza|PlayerA

    def test_pin_player_stats_no_collision_partidos_unchanged(self):
        """REGRESSION N4: partidos counter is not inflated by cross-match collision."""
        season_sim = SeasonSimulator()
        stats = {}
        match1 = _make_match(
            "Trento",
            "Perugia",
            set_specs=[
                (25, 20, "home", [{"jugador": "PlayerA", "puntos": 5}]),
                (25, 22, "home", [{"jugador": "PlayerA", "puntos": 3}]),
                (25, 18, "home", [{"jugador": "PlayerA", "puntos": 2}]),
            ],
        )
        season_sim._accumulate_player_stats(stats, match1, "Trento", "home")
        assert stats["Trento|PlayerA"]["partidos"] == 1
        match2 = _make_match(
            "Trento",
            "Monza",
            sets_home=3,
            sets_away=1,
            winner="home",
            resultado="3-1",
            set_specs=[
                (25, 20, "home", [{"jugador": "PlayerA", "puntos": 4}]),
                (25, 22, "home", [{"jugador": "PlayerA", "puntos": 3}]),
                (20, 25, "away", [{"jugador": "PlayerA", "puntos": 2}]),
                (25, 18, "home", [{"jugador": "PlayerA", "puntos": 1}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match2, "Trento", "home")

        # After 2 matches, partidos should be 2
        assert stats["Trento|PlayerA"]["partidos"] == 2
        # Puntos should be 5+3+2 + 4+3+2+1 = 20
        assert stats["Trento|PlayerA"]["puntos"] == 20


class TestAdaptiveDamping:
    """adaptive_damping() function and its use in simulate_season (Batch 3 mid-effort #3)."""

    def test_adaptive_damping_linear(self):
        """Linear interpolation from start to end over total_jornadas."""
        # Start
        assert adaptive_damping(0) == ADAPTIVE_DAMPING_START
        # End (jornada >= total)
        assert adaptive_damping(SUPERLEGA_TOTAL_JORNADAS) == ADAPTIVE_DAMPING_END
        # Middle: linear interpolation
        halfway = adaptive_damping(SUPERLEGA_TOTAL_JORNADAS // 2)
        expected = (ADAPTIVE_DAMPING_START + ADAPTIVE_DAMPING_END) / 2
        assert halfway == pytest.approx(expected, abs=1e-9)

    def test_adaptive_damping_clamps_above_total(self):
        """Jornadas above total stay at damping_end."""
        assert adaptive_damping(100) == ADAPTIVE_DAMPING_END
        assert adaptive_damping(1000) == ADAPTIVE_DAMPING_END

    def test_adaptive_damping_clamps_below_zero(self):
        """Negative jornadas stay at damping_start."""
        assert adaptive_damping(-1) == ADAPTIVE_DAMPING_START

    def test_adaptive_damping_custom_range(self):
        """Custom start/end overrides defaults."""
        assert adaptive_damping(0, damping_start=0.9, damping_end=0.1) == 0.9
        assert adaptive_damping(10, total_jornadas=10, damping_start=0.9, damping_end=0.1) == 0.1

    def test_adaptive_damping_higher_at_start(self):
        """Adaptive should give LOWER damping early (more shrinkage when features are cold)."""
        early = adaptive_damping(2)
        late = adaptive_damping(SUPERLEGA_TOTAL_JORNADAS - 2)
        assert early < late, (
            f"Early-season damping {early} should be < late-season {late} "
            f"(more shrinkage early, more trust in model late)"
        )
        # Sanity: early is below fixed default (0.5), late is above
        assert early < MATCH_PREDICTOR_DAMPING
        assert late > MATCH_PREDICTOR_DAMPING

    def test_calibrate_strengths_responds_to_damping(self):
        """_calibrate_strengths must respect the damping parameter (direct unit test)."""
        sim = SeasonSimulator()
        h_str, a_str = 0.5, 0.5
        # Strong prediction: p_target = 0.7. Different dampings should give
        # different h_str outputs (only at p_target != 0.5, since 0.5 is neutral).
        # In the math k_damped = k ** damping, HIGHER damping = k closer to k
        # (more change), LOWER damping = k^0 = 1 (no change).
        h_low_damping, _ = sim._calibrate_strengths(h_str, a_str, p_target=0.7, damping=0.3)
        h_high_damping, _ = sim._calibrate_strengths(h_str, a_str, p_target=0.7, damping=0.7)
        # Higher damping = trust the model more = larger change
        assert h_high_damping > h_low_damping, (
            f"Higher damping {h_high_damping} should produce larger h_str shift "
            f"than lower damping {h_low_damping}"
        )
        # Sanity: at p_target=0.5 (neutral), no change regardless of damping
        h_neutral_low, _ = sim._calibrate_strengths(h_str, a_str, p_target=0.5, damping=0.3)
        h_neutral_high, _ = sim._calibrate_strengths(h_str, a_str, p_target=0.5, damping=0.7)
        assert h_neutral_low == h_neutral_high == 0.5
