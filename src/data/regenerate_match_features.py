"""
regenerate_match_features.py — Regenera match_features.csv con la colisión de partido_id corregida.

(B0 del PLAN_MEJORAS_CONSOLIDADO)

El `DB/features/match_features.csv` existente fue generado con el código PRE-B0,
que agrupaba por ``partido_id`` solamente (fundiendo ida y vuelta). Esto producía
solo ~725 partidos "válidos" cuando en realidad hay ~1322. El B0 corrige la
agregación en ``_aggregate_matches`` a ``(partido_id, local)``.

Este módulo regenera match_features.csv usando ``build_rolling_match_features``
(que internamente usa ``_aggregate_matches`` ya corregido) para producir features
pre-partido correctas, y luego mapea las columnas al formato histórico que esperan
los consumidores (``RuntimeFeatureBuilder``, API, modelos ML).

Uso:
    python -m src.data.regenerate_match_features
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.rolling_features import (
    build_rolling_match_features,
    _aggregate_matches,
    ELO_HOME_ADV,
)

MATCH_FEATURES_PATH = BASE_DIR / "DB" / "features" / "match_features.csv"
BACKUP_PATH = BASE_DIR / "DB" / "features" / "match_features_collided_backup.csv"


def _sp_to_temporada_str(year: int) -> str:
    """Convierte año de inicio a formato '2016/2017'."""
    return f"{year}/{year + 1}"


def build_match_features(sp: pd.DataFrame) -> pd.DataFrame:
    """Construye match_features con la agregación corregida (B0).

    Parte de ``build_rolling_match_features`` (que ya tiene el B0) y recompone
    las columnas al formato histórico que esperan los consumidores del CSV:

    - Renombra columnas rolling para coincidir con los nombres viejos.
    - Añade columnas que el rolling no produce explícitamente (pts_fav_exp,
      rank_season, descanso, SOS, dominancia, etc.) con valores coherentes.
    - Devuelve un DataFrame con 66 columnas, mismo orden que el CSV original.
    """
    df = build_rolling_match_features(sp)

    # -- Mapa de renombres: rolling -> formato histórico --
    rename_map = {
        "h_win_rate": "h_win_rate_global",
        "a_win_rate": "a_win_rate_global",
        "h_set_ratio": "h_set_win_rate",
        "a_set_ratio": "a_set_win_rate",
        "h_point_ratio": "point_ratio_h",
        "a_point_ratio": "point_ratio_a",
        "h_form_ewma": "h_forma_home",
        "a_form_ewma": "a_forma_home",
        "h_streak": "h_racha",
        "a_streak": "a_racha",
        "diff_win_rate": "diff_win_rate_global",
        "diff_set_ratio": "diff_set_win_rate",
        "diff_form_ewma": "diff_forma_efectiva",
        "diff_streak": "diff_racha",
        "h2h_win_rate_h": "h_h2h_win_rate",
    }
    df = df.rename(columns=rename_map)

    # -- h2h_set_diff_exp: (h2h_win_rate - 0.5) * 2 --
    df["h_h2h_set_diff_exp"] = (df["h_h2h_win_rate"] - 0.5) * 2.0

    # -- a-side H2H (simétrico) --
    df["a_h2h_win_rate"] = 1.0 - df["h_h2h_win_rate"]
    df["a_h2h_set_diff_exp"] = (df["a_h2h_win_rate"] - 0.5) * 2.0

    # -- temporada string --
    df["temporada"] = df["temporada_inicio"].apply(_sp_to_temporada_str)

    # -- win_rate_last5 = same as global (rolling no tiene last5) --
    df["h_win_rate_last5"] = df["h_win_rate_global"]
    df["a_win_rate_last5"] = df["a_win_rate_global"]
    df["diff_win_rate_last5"] = df["diff_win_rate_global"]

    # -- Obtener sets_h, sets_a, pts_h, pts_a desde _aggregate_matches --
    am = _aggregate_matches(sp)
    # build_rolling_match_features ya llama a _aggregate_matches internamente,
    # pero no expone pts_h/pts_a/sets_h/sets_a en su salida. Los reincorporamos.
    df = df.merge(
        am[["partido_id", "local", "sets_h", "sets_a", "pts_h", "pts_a"]],
        on=["partido_id", "local"],
        how="left",
    )

    # -- pts_fav/con_exp: puntos por set en el partido --
    total_sets = np.maximum(df["sets_h"] + df["sets_a"], 1)
    df["h_pts_fav_exp"] = df["pts_h"] / total_sets
    df["h_pts_con_exp"] = df["pts_a"] / total_sets
    df["a_pts_fav_exp"] = df["pts_a"] / total_sets
    df["a_pts_con_exp"] = df["pts_h"] / total_sets
    df["diff_pts_fav_exp"] = df["h_pts_fav_exp"] - df["a_pts_fav_exp"]
    df["diff_pts_con_exp"] = df["h_pts_con_exp"] - df["a_pts_con_exp"]

    # -- forma_away = forma_home (rolling no separa localia en forma EWMA) --
    df["h_forma_away"] = df["h_forma_home"]
    df["a_forma_away"] = df["a_forma_home"]

    # -- ultimo_set_diff = 0 (no disponible en rolling) --
    df["h_ultimo_set_diff"] = 0
    df["a_ultimo_set_diff"] = 0
    df["diff_ultimo_set_diff"] = 0

    # -- descanso: constante asumida --
    df["h_descanso"] = 7
    df["a_descanso"] = 7
    df["diff_descanso"] = 0

    # -- rank_season: no disponible desde rolling --
    df["h_rank_season"] = 0
    df["a_rank_season"] = 0
    df["diff_rank_season"] = 0

    # -- Elo con localía --
    df["elo_h_home"] = df["elo_h"] + ELO_HOME_ADV
    df["elo_a_away"] = df["elo_a"]

    # -- set_ratio_h/a (alias de h_set_win_rate) --
    df["set_ratio_h"] = df["h_set_win_rate"]
    df["set_ratio_a"] = df["a_set_win_rate"]
    # diff_set_ratio ya se llamaba así en el rolling original (y se renombró a diff_set_win_rate)
    # El formato viejo espera diff_set_ratio como columna propia
    df["diff_set_ratio"] = df["diff_set_win_rate"]

    # -- dominancia = set_win_rate - 0.5 --
    df["dominancia_h"] = df["h_set_win_rate"] - 0.5
    df["dominancia_a"] = df["a_set_win_rate"] - 0.5
    df["diff_dominancia"] = df["diff_set_win_rate"]

    # -- SOS: constante --
    df["sos_h"] = 0.5
    df["sos_a"] = 0.5
    df["diff_sos"] = 0.0

    # -- Orden de columnas exacto del CSV original (69 columnas, +3 B5) --
    ordered = [
        "partido_id",
        "temporada",
        "jornada_num",
        "local",
        "visitante",
        "h_win_rate_global",
        "h_win_rate_last5",
        "h_win_rate_home",
        "h_win_rate_away",
        "h_pts_fav_exp",
        "h_pts_con_exp",
        "h_set_win_rate",
        "h_set_diff_exp",
        "h_forma_home",
        "h_forma_away",
        "h_ultimo_set_diff",
        "h_racha",
        "h_h2h_set_diff_exp",
        "h_h2h_win_rate",
        "a_win_rate_global",
        "a_win_rate_last5",
        "a_win_rate_home",
        "a_win_rate_away",
        "a_pts_fav_exp",
        "a_pts_con_exp",
        "a_set_win_rate",
        "a_set_diff_exp",
        "a_forma_home",
        "a_forma_away",
        "a_ultimo_set_diff",
        "a_racha",
        "a_h2h_set_diff_exp",
        "a_h2h_win_rate",
        "diff_win_rate_global",
        "diff_win_rate_last5",
        "diff_set_win_rate",
        "diff_set_diff_exp",
        "diff_pts_fav_exp",
        "diff_pts_con_exp",
        "diff_racha",
        "diff_ultimo_set_diff",
        "diff_rank_season",
        "diff_forma_efectiva",
        "h_descanso",
        "a_descanso",
        "diff_descanso",
        "h_rank_season",
        "a_rank_season",
        "elo_h",
        "elo_a",
        "elo_h_home",
        "elo_a_away",
        "elo_diff",
        "elo_win_prob_h",
        "set_ratio_h",
        "set_ratio_a",
        "point_ratio_h",
        "point_ratio_a",
        "dominancia_h",
        "dominancia_a",
        "diff_set_ratio",
        "diff_dominancia",
        "sos_h",
        "sos_a",
        "diff_sos",
        "h_roster_continuity",
        "a_roster_continuity",
        "diff_roster_continuity",
        "gana_local",
    ]

    available = [c for c in ordered if c in df.columns]
    missing = set(ordered) - set(df.columns)
    if missing:
        print(f"  [WARN] Columnas faltantes en el DataFrame (rellenadas con 0): {missing}")
        for c in missing:
            df[c] = 0
        available = [c for c in ordered if c in df.columns]

    return df[available]


def main():
    print("=" * 70)
    print("  REGENERACIÓN DE match_features.csv (B0) — sin colisión de partido_id")
    print("=" * 70)

    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    # Normalizar equipos (igual que data_pipeline.load_sets_partidos)
    from src.data.team_mapper import normalize_team_name

    sp["equipo_local"] = sp["equipo_local"].apply(normalize_team_name)
    sp["equipo_visitante"] = sp["equipo_visitante"].apply(normalize_team_name)

    df = build_match_features(sp)

    print(f"  Filas (partidos): {len(df)}")
    print(f"  Columnas: {len(df.columns)}")
    print(f"  Temporadas: {sorted(df['temporada'].unique())}")
    print(f"  Balance target gana_local: {df['gana_local'].mean():.3f}")

    # Backup del CSV colisionado (solo la primera vez)
    if not BACKUP_PATH.exists() and MATCH_FEATURES_PATH.exists():
        MATCH_FEATURES_PATH.replace(BACKUP_PATH)
        print(f"  Backup del CSV viejo -> {BACKUP_PATH.name}")

    # Atomic write: tmp + os.replace
    MATCH_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MATCH_FEATURES_PATH.with_suffix(MATCH_FEATURES_PATH.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, MATCH_FEATURES_PATH)
    print(f"  Escrito {MATCH_FEATURES_PATH}")


if __name__ == "__main__":
    main()
