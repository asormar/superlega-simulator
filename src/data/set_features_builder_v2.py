"""
set_features_builder_v2.py — Generates set_features_v2.csv via the contract.

Walks DB/sets_partidos.csv chronologically, builds a SetContext for each set,
calls build_set_features(ctx) to produce the 21 SET_FEATURE_COLS, and writes
DB/features/set_features_v2.csv.

This is the TRAINING dataset generator.  At runtime, the same
build_set_features function is called from the simulator's clamp path, so
train/serve feature definitions are identical by construction.

The legacy DB/features/set_features.csv (B0b) is NOT modified.
"""

import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.feature_store import SET_FEATURE_COLS
from src.data.rolling_features import build_rolling_match_features, elo_to_strength
from src.data.set_feature_contract import SetContext, build_set_features
from src.data.team_mapper import normalize_team_name

V2_FEATURES_PATH = BASE_DIR / "DB" / "features" / "set_features_v2.csv"


def _set_sequences(sp: pd.DataFrame) -> dict:
    """(partido_id, local_norm) -> list of ganador_set_local in set order.

    Reconstructs the sets of each REAL match (grouping by (partido_id,
    equipo_local) to undo the ida/vuelta collision), ordered by set_num.
    Every team name is normalised (Guardrail 2).
    """
    df = sp.copy()
    df["local_norm"] = df["equipo_local"].apply(normalize_team_name)
    seqs = {}
    for (pid, loc), g in df.groupby(["partido_id", "local_norm"]):
        g = g.sort_values("set_num")
        seqs[(pid, loc)] = [int(x) for x in g["ganador_set_local"].tolist()]
    return seqs


def build_set_features_v2(sp: pd.DataFrame) -> pd.DataFrame:
    """Build a clean set-features DataFrame (1 row per set, PRE-set state).

    Args:
        sp: sets_partidos.csv as a DataFrame.

    Returns:
        DataFrame with columns:
          partido_id, set_num, temporada_inicio, SET_FEATURE_COLS (21),
          ganador_set_local.
    """
    match_features = build_rolling_match_features(sp)  # pre-match features
    seqs = _set_sequences(sp)

    rows = []
    missing = 0

    for idx, r in match_features.iterrows():
        key = (r["partido_id"], r["local"])
        seq = seqs.get(key)
        if not seq:
            missing += 1
            continue

        t = int(r["temporada_inicio"])
        # Unique partido_id: includes season + global index to undo collision.
        uid = f"{t}/{t + 1}_m{idx:05d}"

        # Rolling state from build_rolling_match_features
        elo_h = float(r["elo_h"])
        elo_a = float(r["elo_a"])
        strength_h = elo_to_strength(elo_h)
        strength_a = elo_to_strength(elo_a)

        h2h_win_rate = float(r.get("h2h_win_rate_h", 0.5))

        sets_h = 0
        sets_a = 0
        prev_home_won = -1  # -1 = first set

        for si, winner_set in enumerate(seq, start=1):
            ctx = SetContext(
                temporada_inicio=t,
                jornada_num=int(r["jornada_num"]),
                match_id=uid,
                set_index=si,
                equipo_local=r["local"],
                equipo_visitante=r["visitante"],
                elo_local=elo_h,
                elo_visitante=elo_a,
                strength_local=strength_h,
                strength_visitante=strength_a,
                h_win_rate_global=float(r.get("h_win_rate", 0.5)),
                a_win_rate_global=float(r.get("a_win_rate", 0.5)),
                h_set_win_rate=float(r.get("h_set_ratio", 0.5)),
                a_set_win_rate=float(r.get("a_set_ratio", 0.5)),
                h_form_ewma=float(r.get("h_form_ewma", 0.5)),
                a_form_ewma=float(r.get("a_form_ewma", 0.5)),
                h_set_diff_exp=float(r.get("h_set_diff_exp", 0.0)),
                a_set_diff_exp=float(r.get("a_set_diff_exp", 0.0)),
                h_point_ratio=float(r.get("h_point_ratio", 0.5)),
                a_point_ratio=float(r.get("a_point_ratio", 0.5)),
                h2h_win_rate=h2h_win_rate,
                sets_h_antes=sets_h,
                sets_a_antes=sets_a,
                prev_home_won=prev_home_won,
                target_score=25 if (sets_h + sets_a) < 8 else 15,
            )

            feats = build_set_features(ctx)
            feats["partido_id"] = uid
            feats["set_num"] = si
            feats["temporada_inicio"] = t
            feats["ganador_set_local"] = int(winner_set)
            rows.append(feats)

            # Update in-match state AFTER emitting the row (no leakage)
            if winner_set == 1:
                sets_h += 1
                prev_home_won = 1
            else:
                sets_a += 1
                prev_home_won = 0

    if missing:
        print(f"  [WARN] {missing} matches without set sequences (skipped).")

    ordered = (
        ["partido_id", "set_num", "temporada_inicio"]
        + list(SET_FEATURE_COLS)
        + ["ganador_set_local"]
    )
    df = pd.DataFrame(rows)
    return df[[c for c in ordered if c in df.columns]]


def main():
    print("=" * 70)
    print("  REGENERACION DE set_features_v2.csv (A3 — via contract)")
    print("=" * 70)

    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    df = build_set_features_v2(sp)

    print(f"  Rows (1 per set): {len(df)}")
    df_t = df.copy()
    print("  Sets per season:")
    print(df_t.groupby("temporada_inicio")["set_num"].count().to_string())
    print(
        f"  sets_h_antes range: [{df['sets_h_antes'].min()}, {df['sets_h_antes'].max()}] "
        f"(expected max 2)"
    )
    print(f"  Target balance (ganador_set_local): {df['ganador_set_local'].mean():.3f}")
    print(f"  Schema columns ({len(df.columns)}): {list(df.columns)}")

    # Atomic write: tmp + os.replace
    V2_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = V2_FEATURES_PATH.with_suffix(V2_FEATURES_PATH.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, V2_FEATURES_PATH)
    print(f"  Written: {V2_FEATURES_PATH}")


if __name__ == "__main__":
    main()
