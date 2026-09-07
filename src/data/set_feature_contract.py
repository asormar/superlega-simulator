"""
set_feature_contract.py — Pure-function contract for set-feature computation.

Single source of truth for the 21 SET_FEATURE_COLS used to evaluate the
SetPredictor.  Called by BOTH the offline dataset builder (training) and the
runtime clamp path (serve), so train/serve skew becomes a compile-time error.

Usage:
    ctx = SetContext(...)
    feats = build_set_features(ctx)  # -> dict with exactly 21 keys
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.feature_store import SET_FEATURE_COLS


@dataclass(frozen=True)
class SetContext:
    """Frozen dataclass carrying EVERY input needed to build the 21 set features.

    Three field groups:
      (a) rolling team state (pre-match, from build_rolling_match_features or
          RuntimeFeatureBuilder.build_features),
      (b) in-match set state (sets won before this set, set number, ...),
      (c) derived constants (target_score).

    No field reads global state, mutable module attributes, or the filesystem.
    """

    # --- Identity (offline: from sets_partidos.csv row; online: from simulator) ---
    temporada_inicio: int = 0  # e.g. 2024  (also used for recency in trainer)
    jornada_num: int = 0  # 1..N within temporada
    match_id: str = ""  # offline: "{t}/{t+1}_m{idx:05d}"; online: any stable id
    set_index: int = 1  # 1..5 within the match (the set being played)
    equipo_local: str = ""  # canonical (post normalize_team_name)
    equipo_visitante: str = ""  # canonical

    # --- Strength / Elo ---
    elo_local: float = 1500.0  # raw Elo (pre-match)
    elo_visitante: float = 1500.0  # raw Elo
    strength_local: float = 0.5  # elo_to_strength(elo_local) — precomputed
    strength_visitante: float = 0.5

    # --- Rolling team state (pre-match) ---
    h_win_rate_global: float = 0.5  # all-games win rate expanding, [0,1]
    a_win_rate_global: float = 0.5
    h_set_win_rate: float = 0.5  # sets win rate expanding, [0,1]
    a_set_win_rate: float = 0.5
    h_form_ewma: float = 0.5  # EWMA form (ALL games, half-life=5), [0,1]
    a_form_ewma: float = 0.5
    h_set_diff_exp: float = 0.0  # EWMA set differential, [-1,1]-ish
    a_set_diff_exp: float = 0.0
    h_point_ratio: float = 0.5  # expanding point ratio (historical pts_fav), [0,1]
    a_point_ratio: float = 0.5
    h2h_win_rate: float = 0.5  # H2H decaying win rate (H2H_HALFLIFE=2), [0,1]

    # --- In-match set state ---
    sets_h_antes: int = 0  # sets won by local BEFORE this set (0..2)
    sets_a_antes: int = 0  # sets won by visitante BEFORE this set (0..2)
    prev_home_won: int = -1  # -1 unknown (set 1), 0 lost prev, 1 won prev — for momentum_h

    # --- Derived constants ---
    target_score: int = 25  # 25 for sets 1-4, 15 for set 5


def build_set_features(ctx: SetContext) -> Dict[str, float]:
    """Produce the 21 SET_FEATURE_COLS from a SetContext.

    Pure function — no I/O, no module-level state, no hidden RNG, no mutation
    of *ctx*.  Deterministic: the same *ctx* always yields the same dict.

    Each of the 21 feature branches is documented inline with:
      - feature name
      - definition / formula
      - expected range
      - source (rolling pre-match / in-match set state / derived constant)

    Returns:
        dict[str, float] with exactly the 21 keys declared in SET_FEATURE_COLS,
        in the canonical order.
    """
    feats: Dict[str, float] = {}

    # ═══════════════════════════════════════════════════════════
    # Step 1: team-level / rolling features (direct from ctx fields)
    # ═══════════════════════════════════════════════════════════

    # strength_h (1) — logistic from Elo, NOT elo/3000 linear.
    #   Definition: ctx.strength_local (precomputed elo_to_strength)
    #   Range: [0, 1]
    #   Source: rolling pre-match (strength field)
    feats["strength_h"] = ctx.strength_local

    # strength_a (2) — mirrored.
    #   Range: [0, 1]
    #   Source: rolling pre-match
    feats["strength_a"] = ctx.strength_visitante

    # set_wr_h (5) — expanding set win rate within season, pre-match.
    #   Range: [0, 1]
    #   Source: rolling pre-match (h_set_win_rate)
    feats["set_wr_h"] = ctx.h_set_win_rate

    # set_wr_a (6) — mirrored.
    #   Range: [0, 1]
    #   Source: rolling pre-match
    feats["set_wr_a"] = ctx.a_set_win_rate

    # forma_h (8) — ALL-games EWMA form (NOT home-only, decision #4).
    #   Range: [0, 1]
    #   Source: rolling pre-match (h_form_ewma)
    feats["forma_h"] = ctx.h_form_ewma

    # forma_a (9) — mirrored.
    #   Range: [0, 1]
    #   Source: rolling pre-match
    feats["forma_a"] = ctx.a_form_ewma

    # pts_fav_h (11) — historical expanding point ratio (NOT live score, decision #1).
    #   Range: [0, 1]
    #   Source: rolling pre-match (h_point_ratio)
    feats["pts_fav_h"] = ctx.h_point_ratio

    # pts_fav_a (12) — mirrored.
    #   Range: [0, 1]
    #   Source: rolling pre-match
    feats["pts_fav_a"] = ctx.a_point_ratio

    # h2h_diff (13) — signed H2H win rate scaled to [-1, 1].
    #   Definition: (ctx.h2h_win_rate - 0.5) * 2.0
    #   Range: [-1, 1]
    #   Source: rolling pre-match (h2h_win_rate)
    feats["h2h_diff"] = (ctx.h2h_win_rate - 0.5) * 2.0

    # ═══════════════════════════════════════════════════════════
    # Step 2: derived diffs (one-line arithmetic on the values just added)
    # ═══════════════════════════════════════════════════════════

    # strength_diff (3) — home minus away logistic strength.
    #   Definition: strength_h - strength_a
    #   Range: [-1, 1]
    #   Source: derived from strength_h, strength_a
    feats["strength_diff"] = feats["strength_h"] - feats["strength_a"]

    # elo_diff (4) — home minus away raw Elo.
    #   Definition: ctx.elo_local - ctx.elo_visitante
    #   Range: [-1500, 1500]
    #   Source: derived from elo fields
    feats["elo_diff"] = ctx.elo_local - ctx.elo_visitante

    # diff_set_wr (7) — set win rate difference.
    #   Definition: set_wr_h - set_wr_a
    #   Range: [-1, 1]
    #   Source: derived from set_wr_h, set_wr_a
    feats["diff_set_wr"] = feats["set_wr_h"] - feats["set_wr_a"]

    # diff_forma (10) — form difference.
    #   Definition: forma_h - forma_a
    #   Range: [-1, 1]
    #   Source: derived from forma_h, forma_a
    feats["diff_forma"] = feats["forma_h"] - feats["forma_a"]

    # diff_set_ratio (14) — same semantic as diff_set_wr (preserves legacy name).
    #   Definition: same as diff_set_wr
    #   Range: [-1, 1]
    #   Source: derived (alias)
    feats["diff_set_ratio"] = feats["diff_set_wr"]

    # diff_dominancia (15) — documented alias of diff_set_ratio (decision #5).
    #   Definition: same as diff_set_ratio
    #   Range: [-1, 1]
    #   Source: derived (alias)
    feats["diff_dominancia"] = feats["diff_set_ratio"]

    # ═══════════════════════════════════════════════════════════
    # Step 3: in-match features (set state and momentum)
    # ═══════════════════════════════════════════════════════════

    # set_num_norm (16) — normalised set number.
    #   Definition: (ctx.set_index - 1) / 4.0
    #   Range: [0, 1]  (set 1 -> 0.0, set 5 -> 1.0)
    #   Source: in-match (set_index)
    feats["set_num_norm"] = (ctx.set_index - 1) / 4.0

    # sets_h_antes (17) — sets won by local before this set.
    #   Range: [0, 2]
    #   Source: in-match (sets_h_antes)
    feats["sets_h_antes"] = float(ctx.sets_h_antes)

    # sets_a_antes (18) — sets won by visitante before this set.
    #   Range: [0, 2]
    #   Source: in-match (sets_a_antes)
    feats["sets_a_antes"] = float(ctx.sets_a_antes)

    # diff_sets_antes (19) — set difference before this set.
    #   Definition: sets_h_antes - sets_a_antes
    #   Range: [-2, 2]
    #   Source: derived from sets_h/a_antes
    feats["diff_sets_antes"] = float(ctx.sets_h_antes - ctx.sets_a_antes)

    # momentum_h (20) — DISCRETE {0.5, 1.0, 0.0} (decision #2).
    #   NOT the continuous (score_h - score_a) / total formula.
    #   - prev_home_won < 0 (first set)  -> 0.5
    #   - prev_home_won == 1 (home won)  -> 1.0
    #   - prev_home_won == 0 (home lost) -> 0.0
    #   Range: {0.0, 0.5, 1.0}
    #   Source: in-match (prev_home_won)
    if ctx.prev_home_won < 0:
        feats["momentum_h"] = 0.5
    elif ctx.prev_home_won == 1:
        feats["momentum_h"] = 1.0
    else:
        feats["momentum_h"] = 0.0

    # es_desempate (21) — tiebreak flag (1 when both teams have 2 sets).
    #   Definition: 1 if sets_h_antes == 2 and sets_a_antes == 2 else 0
    #   Range: {0, 1}
    #   Source: derived from sets_h/a_antes
    feats["es_desempate"] = 1.0 if (ctx.sets_h_antes == 2 and ctx.sets_a_antes == 2) else 0.0

    # ═══════════════════════════════════════════════════════════
    # Schema enforcement — fail fast if the contract drifts
    # ═══════════════════════════════════════════════════════════
    if set(feats.keys()) != set(SET_FEATURE_COLS) or len(feats) != 21:
        missing = set(SET_FEATURE_COLS) - set(feats.keys())
        extra = set(feats.keys()) - set(SET_FEATURE_COLS)
        msg = "build_set_features schema violation"
        if missing:
            msg += f"; missing={sorted(missing)}"
        if extra:
            msg += f"; extra={sorted(extra)}"
        raise RuntimeError(msg)

    # Return in canonical SET_FEATURE_COLS order
    return {col: feats[col] for col in SET_FEATURE_COLS}
