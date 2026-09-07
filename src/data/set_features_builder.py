"""
set_features_builder.py — Regenera set_features.csv sin la colisión de partido_id.

(B0b del PLAN_MEJORAS_CONSOLIDADO)

El `DB/features/set_features.csv` original venía de un script externo (no está
en el repo) y arrastra el mismo bug de colisión que `_aggregate_matches`: sus
filas se agrupan por un `partido_id` que funde la ida y la vuelta del mismo
cruce, así que `sets_h_antes` llega a 5 (imposible en un best-of-5) y las
features de cada set se calculan sobre el estado mezclado de DOS partidos.
Como el SetPredictor v2 (`train_improved.train_set`) ENTRENA sobre este CSV,
su entrenamiento estaba corrupto.

Este módulo lo regenera de forma limpia:
  - Reconstruye los partidos reales (1 por `(partido_id, local)`, sin colisión)
    y reutiliza `build_rolling_match_features` (ya corregido) para las features
    pre-partido, que se calculan recorriendo el histórico en orden cronológico
    y usando SOLO información previa (sin leakage).
  - Expande cada partido a sus sets REALES, en orden, con el estado in-set
    (sets ganados antes, momentum del set anterior, quinto set) correcto y
    por-partido (ya no acumulado entre ida y vuelta).
  - Emite 1 fila por set (pre-set), con las 21 columnas de SET_FEATURE_COLS +
    `partido_id` (único por partido) + `set_num` + target `ganador_set_local`.

Definición de cada feature (documentada para el contrato train/serve — la
reconciliación completa con el runtime es el item A3 del plan):

  strength_h/a   = logística del Elo pre-partido (1/(1+10^(-(elo-1500)/400))).
  strength_diff  = strength_h - strength_a.
  elo_diff       = elo_h - elo_a (pre-partido).
  set_wr_h/a     = set win rate expanding dentro de temporada (pre-partido).
  diff_set_wr    = set_wr_h - set_wr_a.
  forma_h/a      = forma EWMA (half-life 5 partidos), pre-partido.
  diff_forma     = forma_h - forma_a.
  pts_fav_h/a    = point ratio expanding dentro de temporada (dominancia de
                   puntos pre-partido; ~0.5). [difiere de la escala del CSV
                   viejo, irrelevante para un LogReg; A3 reconcilia con runtime]
  h2h_diff       = (h2h_win_rate - 0.5) * 2  ∈ [-1,1]  [alineado con el runtime
                   _extract_set_team_features, no con la escala [-3,3] del viejo]
  diff_set_ratio = set_wr_h - set_wr_a (diff de dominancia de sets).
  diff_dominancia= igual a diff_set_ratio (redundante, como en el runtime).
  set_num_norm   = (set_num - 1) / 4.
  sets_h/a_antes = sets ganados por local/visitante ANTES de este set (0-2).
  diff_sets_antes= sets_h_antes - sets_a_antes.
  momentum_h     = 0.5 en el 1er set; 1.0 si el local ganó el set anterior,
                   0.0 si lo perdió.
  es_desempate   = 1 si es el 5º set (2-2), 0 si no.
  ganador_set_local = target (1 si el local ganó el set).
"""

import sys
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.team_mapper import normalize_team_name
from src.data.rolling_features import build_rolling_match_features, elo_to_strength
from src.data.feature_store import SET_FEATURE_COLS

SET_FEATURES_PATH = BASE_DIR / "DB" / "features" / "set_features.csv"
BACKUP_PATH = BASE_DIR / "DB" / "features" / "set_features_collided_backup.csv"


def _set_sequences(sp: pd.DataFrame) -> dict:
    """(partido_id, local_norm) -> lista de ganador_set_local en orden de set.

    Reconstruye los sets de cada partido REAL (agrupando por (partido_id,
    equipo_local) para deshacer la colisión ida/vuelta), ordenados por set_num.
    """
    df = sp.copy()
    df["local_norm"] = df["equipo_local"].apply(normalize_team_name)
    seqs = {}
    for (pid, loc), g in df.groupby(["partido_id", "local_norm"]):
        g = g.sort_values("set_num")
        seqs[(pid, loc)] = [int(x) for x in g["ganador_set_local"].tolist()]
    return seqs


def build_set_features(sp: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame de set features limpio (1 fila por set, pre-set).

    Returns:
        DataFrame con columnas: partido_id, set_num, SET_FEATURE_COLS, ganador_set_local.
    """
    match_features = build_rolling_match_features(sp)  # features pre-partido correctas
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
        # partido_id ÚNICO por partido: incluye la temporada (para que los
        # loaders extraigan bien el año) y un índice global -> deshace la
        # colisión también en el CSV resultante.
        uid = f"{t}/{t + 1}_m{idx:05d}"

        elo_h, elo_a = float(r["elo_h"]), float(r["elo_a"])
        s_h, s_a = elo_to_strength(elo_h), elo_to_strength(elo_a)
        set_wr_h, set_wr_a = float(r["h_set_ratio"]), float(r["a_set_ratio"])
        forma_h, forma_a = float(r["h_form_ewma"]), float(r["a_form_ewma"])
        base = {
            "strength_h": s_h,
            "strength_a": s_a,
            "strength_diff": s_h - s_a,
            "elo_diff": elo_h - elo_a,
            "set_wr_h": set_wr_h,
            "set_wr_a": set_wr_a,
            "diff_set_wr": set_wr_h - set_wr_a,
            "forma_h": forma_h,
            "forma_a": forma_a,
            "diff_forma": forma_h - forma_a,
            "pts_fav_h": float(r["h_point_ratio"]),
            "pts_fav_a": float(r["a_point_ratio"]),
            "h2h_diff": (float(r["h2h_win_rate_h"]) - 0.5) * 2.0,
            "diff_set_ratio": float(r["diff_set_ratio"]),
            "diff_dominancia": float(r["diff_set_ratio"]),
        }

        sets_h = sets_a = 0
        prev_home_won = None
        for si, winner_set in enumerate(seq, start=1):
            momentum = 0.5 if prev_home_won is None else (1.0 if prev_home_won else 0.0)
            es_desempate = 1 if (sets_h == 2 and sets_a == 2) else 0
            row = dict(base)
            row.update(
                {
                    "partido_id": uid,
                    "set_num": si,
                    "set_num_norm": (si - 1) / 4.0,
                    "sets_h_antes": sets_h,
                    "sets_a_antes": sets_a,
                    "diff_sets_antes": sets_h - sets_a,
                    "momentum_h": momentum,
                    "es_desempate": es_desempate,
                    "ganador_set_local": int(winner_set),
                }
            )
            rows.append(row)
            if winner_set == 1:
                sets_h += 1
                prev_home_won = True
            else:
                sets_a += 1
                prev_home_won = False

    if missing:
        print(f"  [WARN] {missing} partidos de dfm sin secuencia de sets (se omiten).")

    # Orden de columnas: partido_id, set_num, las 21 de SET_FEATURE_COLS, target.
    ordered = ["partido_id", "set_num"] + list(SET_FEATURE_COLS) + ["ganador_set_local"]
    df = pd.DataFrame(rows)
    return df[[c for c in ordered if c in df.columns]]


def main():
    print("=" * 70)
    print("  REGENERACIÓN DE set_features.csv (B0b) — sin colisión de partido_id")
    print("=" * 70)
    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    df = build_set_features(sp)

    print(f"  Filas (1 por set): {len(df)}")
    df_t = df.copy()
    df_t["t"] = df_t["partido_id"].str.split("/").str[0].astype(int)
    print("  Sets por temporada:")
    print(df_t.groupby("t")["set_num"].count().to_string())
    print(
        f"  sets_h_antes rango: [{df['sets_h_antes'].min()}, {df['sets_h_antes'].max()}] "
        f"(esperado max 2)"
    )
    print(f"  Balance target ganador_set_local: {df['ganador_set_local'].mean():.3f}")

    # Backup del CSV colisionado (solo la primera vez).
    if not BACKUP_PATH.exists() and SET_FEATURES_PATH.exists():
        SET_FEATURES_PATH.replace(BACKUP_PATH)
        print(f"  Backup del CSV viejo -> {BACKUP_PATH.name}")

    # Atomic write: tmp + os.replace (evita corrupción si el proceso muere entre medias).
    tmp_path = SET_FEATURES_PATH.with_suffix(SET_FEATURES_PATH.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, SET_FEATURES_PATH)
    print(f"  Escrito {SET_FEATURES_PATH}")


if __name__ == "__main__":
    main()
