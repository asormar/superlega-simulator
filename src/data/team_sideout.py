"""
team_sideout.py — Per-team sideout rate approximation (Batch 3 mid-effort).

Quick Win 2 (Optuna) had limited impact; this module attacks the bigger
finding from memoria/point_probability.md: the global DEFAULT_SIDEOUT_RATE
= 0.62 masks real per-team variation. Top teams sideout ~65% of the time
when receiving, weak teams ~58% (per the literature; this repo's data is
set-level, so we use the team's overall point ratio as a proxy).

Approximation
-------------
Without point-level data, we can't directly measure "rate of winning
when receiving". The team's overall point ratio (points won / points
played, summed across home and away) is the best available signal:
a team with a high point ratio is good at both serving and receiving;
a team with a low ratio is weak at both. We use this as the sideout
rate proxy. The spread across current SuperLega teams is ~0.07 (0.47-0.54).

The constant DEFAULT_SIDEOUT_RATE (0.62) is the league-average prior
from volleyball analytics; teams without enough history fall back to it.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.team_mapper import normalize_team_name
from src.simulation.constants import DEFAULT_SIDEOUT_RATE

# Module-level cache; computed lazily on first call.
_SIDEOUT_CACHE: Optional[dict[str, float]] = None
_MIN_SETS_THRESHOLD = 50  # Teams with fewer total sets fall back to DEFAULT_SIDEOUT_RATE


def _compute_team_sideout() -> dict[str, float]:
    """
    Read sets_partidos.csv and compute per-team point ratio (sideout proxy).

    Returns dict mapping canonical team name -> sideout rate in [0, 1].
    Only teams with >= _MIN_SETS_THRESHOLD total sets (as either local or
    visitor) are included. Teams with insufficient data fall back to
    DEFAULT_SIDEOUT_RATE in `get_team_sideout`.
    """
    csv_path = BASE_DIR / "DB" / "sets_partidos.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["local"] = df["equipo_local"].apply(normalize_team_name)
    df["visit"] = df["equipo_visitante"].apply(normalize_team_name)

    # Sum points won and conceded across BOTH home and away appearances
    pts_fav = (
        df.groupby("local")["puntos_local"]
        .sum()
        .add(df.groupby("visit")["puntos_visitante"].sum(), fill_value=0)
    )
    pts_con = (
        df.groupby("local")["puntos_visitante"]
        .sum()
        .add(df.groupby("visit")["puntos_local"].sum(), fill_value=0)
    )
    n_sets = df.groupby("local").size().add(df.groupby("visit").size(), fill_value=0)

    rates = {}
    for team in pts_fav.index:
        if n_sets.get(team, 0) < _MIN_SETS_THRESHOLD:
            continue
        total = pts_fav[team] + pts_con[team]
        if total <= 0:
            continue
        rates[team] = float(pts_fav[team] / total)
    return rates


def get_team_sideout(team_name: str) -> float:
    """
    Return the per-team sideout rate (point-ratio proxy) for a team.

    Falls back to DEFAULT_SIDEOUT_RATE if the team has insufficient data
    or is not in the cache. The cache is built lazily on first call.

    Args:
        team_name: raw or canonical team name (will be normalized).

    Returns:
        Sideout rate in [0, 1].
    """
    global _SIDEOUT_CACHE
    if _SIDEOUT_CACHE is None:
        _SIDEOUT_CACHE = _compute_team_sideout()

    canonical = normalize_team_name(team_name) if team_name else None
    if canonical and canonical in _SIDEOUT_CACHE:
        return _SIDEOUT_CACHE[canonical]
    return DEFAULT_SIDEOUT_RATE


def get_sideout_rates(home_team: str, away_team: str) -> tuple[float, float]:
    """
    Convenience: return (home_sideout, away_sideout) tuple for a matchup.

    Args:
        home_team: local team name (raw or canonical).
        away_team: visitor team name (raw or canonical).

    Returns:
        (home_sideout, away_sideout) tuple, each in [0, 1].
        Either may be DEFAULT_SIDEOUT_RATE if the team is unknown.
    """
    return (get_team_sideout(home_team), get_team_sideout(away_team))


def reset_cache() -> None:
    """Clear the sideout cache (useful for tests)."""
    global _SIDEOUT_CACHE
    _SIDEOUT_CACHE = None


if __name__ == "__main__":
    rates = _compute_team_sideout()
    print(f"Computed sideout rates for {len(rates)} teams")
    print()
    sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 (highest sideout proxy):")
    for team, rate in sorted_rates[:5]:
        print(f"  {team:<25} {rate:.3f}")
    print()
    print("Bottom 5 (lowest sideout proxy):")
    for team, rate in sorted_rates[-5:]:
        print(f"  {team:<25} {rate:.3f}")
    print()
    print(f"DEFAULT_SIDEOUT_RATE (fallback): {DEFAULT_SIDEOUT_RATE}")
