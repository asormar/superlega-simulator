"""
feature_store.py — Gestión de features y splits temporales.

Divide los datos en train/validation/test por temporada (split temporal)
y prepara las matrices de features listas para entrenar los modelos ML.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from src.data.data_pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = BASE_DIR / "models" / "feature_cache"


# ─────────────────────────────────────────────────────────────
# Splits temporales
# ─────────────────────────────────────────────────────────────

# Split temporal: entrena con el pasado, valida/testea con el futuro
TEMPORAL_SPLITS = {
    "train": [2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "val": [2023],
    "test": [2024],
}


# ─────────────────────────────────────────────────────────────
# Features para predicción de PARTIDOS
# ─────────────────────────────────────────────────────────────

# Columnas de features a usar para el modelo de partido
MATCH_FEATURE_COLS = [
    # Win rates
    "h_win_rate_global",
    "h_win_rate_last5",
    "h_win_rate_home",
    "a_win_rate_global",
    "a_win_rate_last5",
    "a_win_rate_away",
    # Diferencias de win rate
    "diff_win_rate_global",
    "diff_win_rate_last5",
    # Set metrics
    "h_set_win_rate",
    "a_set_win_rate",
    "diff_set_win_rate",
    "h_set_diff_exp",
    "a_set_diff_exp",
    "diff_set_diff_exp",
    # Points
    "h_pts_fav_exp",
    "a_pts_fav_exp",
    "diff_pts_fav_exp",
    "h_pts_con_exp",
    "a_pts_con_exp",
    "diff_pts_con_exp",
    # Form
    "h_forma_home",
    "h_forma_away",
    "a_forma_home",
    "a_forma_away",
    "diff_forma_efectiva",
    # H2H
    "h_h2h_win_rate",
    "h_h2h_set_diff_exp",
    # Momentum
    "h_racha",
    "a_racha",
    "diff_racha",
    "h_ultimo_set_diff",
    "a_ultimo_set_diff",
    "diff_ultimo_set_diff",
    # Rest days
    "h_descanso",
    "a_descanso",
    "diff_descanso",
    # Rankings
    "h_rank_season",
    "a_rank_season",
    "diff_rank_season",
    # Elo
    "elo_h",
    "elo_a",
    "elo_diff",
    "elo_win_prob_h",
    "elo_h_home",
    "elo_a_away",
    # Ratios
    "set_ratio_h",
    "set_ratio_a",
    "diff_set_ratio",
    "point_ratio_h",
    "point_ratio_a",
    "dominancia_h",
    "dominancia_a",
    "diff_dominancia",
    # SOS (strength of schedule)
    "sos_h",
    "sos_a",
    "diff_sos",
    # Jornada
    "jornada_num",
]

MATCH_TARGET = "gana_local"


# ─────────────────────────────────────────────────────────────
# Features para predicción de SETS
# ─────────────────────────────────────────────────────────────

SET_FEATURE_COLS = [
    # Pre-match strength
    "strength_h",
    "strength_a",
    "strength_diff",
    "elo_diff",
    # Set-level metrics
    "set_wr_h",
    "set_wr_a",
    "diff_set_wr",
    # Form
    "forma_h",
    "forma_a",
    "diff_forma",
    # Points
    "pts_fav_h",
    "pts_fav_a",
    # H2H
    "h2h_diff",
    # Derived
    "diff_set_ratio",
    "diff_dominancia",
    # In-match state
    "set_num_norm",
    "sets_h_antes",
    "sets_a_antes",
    "diff_sets_antes",
    "momentum_h",
    "es_desempate",
]

SET_TARGET = "ganador_set_local"


# ─────────────────────────────────────────────────────────────
# Enriquecimiento con stats de equipo por temporada
# ─────────────────────────────────────────────────────────────


def enrich_with_team_stats(
    match_df: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enriquece match_features con stats agregadas de equipo por temporada.

    Añade diferencias de ataque, recepción, saque y bloqueo entre
    los dos equipos, cruzando con Comparacion_equipos_10_años.csv.
    """
    df = match_df.copy()

    # Preparar team_stats con clave (equipo, temporada)
    ts = team_stats.copy()

    # La temporada en match_features es '2024/2025', en team_stats es año entero
    # Convertir ambos a año de inicio
    if "temporada_year" in ts.columns:
        ts["temporada_inicio"] = ts["temporada_year"]
    elif "temporada_inicio" not in ts.columns:
        ts["temporada_inicio"] = ts["temporada"].apply(
            lambda x: int(str(x).split("/")[0]) if "/" in str(x) else int(x)
        )

    # Features a extraer por equipo
    stat_cols = {
        "puntos_por_set": "pts_set",
        "aces_por_set": "aces_set",
        "pct_ataque": "atq_pct",
        "ataque_eficacia": "atq_eff",
        "recepcion_eficacia": "rec_eff",
        "bloqueos_por_set": "bloq_set",
        "aces_ratio": "ace_ratio",
    }

    # Merge para equipo local
    for orig_col, short_name in stat_cols.items():
        if orig_col in ts.columns:
            merge_col = f"ts_{short_name}"
            ts_sub = ts[["equipo", "temporada_inicio", orig_col]].rename(
                columns={orig_col: merge_col}
            )

            # Local
            df = df.merge(
                ts_sub.rename(columns={"equipo": "local", merge_col: f"h_{short_name}"}),
                on=["local", "temporada_inicio"],
                how="left",
            )

            # Visitante
            df = df.merge(
                ts_sub.rename(columns={"equipo": "visitante", merge_col: f"a_{short_name}"}),
                on=["visitante", "temporada_inicio"],
                how="left",
            )

            # Diferencia
            df[f"diff_{short_name}"] = df[f"h_{short_name}"] - df[f"a_{short_name}"]

    # Rellenar NaN
    new_cols = [c for c in df.columns if c not in match_df.columns]
    df[new_cols] = df[new_cols].fillna(0)

    print(
        f"  [enrich] {len(new_cols)} features nuevas añadidas: "
        f"{[c for c in new_cols if c.startswith('diff_')]}"
    )

    return df


# Features de enriquecimiento (se añaden dinámicamente)
ENRICHED_MATCH_COLS = [
    "h_pts_set",
    "a_pts_set",
    "diff_pts_set",
    "h_aces_set",
    "a_aces_set",
    "diff_aces_set",
    "h_atq_pct",
    "a_atq_pct",
    "diff_atq_pct",
    "h_atq_eff",
    "a_atq_eff",
    "diff_atq_eff",
    "h_rec_eff",
    "a_rec_eff",
    "diff_rec_eff",
    "h_bloq_set",
    "a_bloq_set",
    "diff_bloq_set",
    "h_ace_ratio",
    "a_ace_ratio",
    "diff_ace_ratio",
]


# ─────────────────────────────────────────────────────────────
# Enriquecimiento con stats de jugadores (roster features)
# ─────────────────────────────────────────────────────────────


def compute_roster_features(
    match_df: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula features agregadas del roster de cada equipo.

    Para cada partido, genera métricas del roster del equipo local y visitante:
    - top_scorer_avg: puntos/set del mejor anotador
    - roster_depth: distribución de puntos (entropy → más repartido = más profundo)
    - ace_threat: aces/set del mejor sacador
    - block_power: bloqueos/set promedio de los top 3 bloqueadores
    - rec_quality: eficacia de recepción del mejor receptor

    Args:
        match_df: DataFrame de match_features con columnas 'local', 'visitante', 'temporada'
        player_stats: DataFrame de player_stats con stats individuales
    """
    from src.data.team_mapper import normalize_team_name
    from src.data.team_id_mapper import get_canonical_team
    from scipy.stats import entropy

    df = match_df.copy()
    ps = player_stats.copy()

    # Normalizar nombres de equipo en player_stats
    # Use team_id_mapper first (handles opaque ID_EQUIPO codes), then
    # normalize_team_name as fallback for other naming variants.
    ps["equipo"] = ps["equipo_id"].apply(lambda x: get_canonical_team(x) or normalize_team_name(x))

    # Filtrar solo jugadores individuales (no totales de equipo)
    # noqa E712 a proposito: esto es una Series de pandas, no un bool. La
    # correccion que sugiere ruff (`not ps[...]`) lanzaria "truth value of a
    # Series is ambiguous". Ademas `!= True` es defensivo frente a NaN, que
    # `~col` no maneja.
    ps = ps[ps["es_total_equipo"] != True].copy()  # noqa: E712

    # Calcular features por equipo-temporada
    roster_agg = {}
    for (equipo, temp), group in ps.groupby(["equipo", "temporada"]):
        sets_total = group["sets"].sum()
        if sets_total <= 0 or len(group) < 3:
            continue

        # Top scorer: pts/set del mejor
        pts_per_set = group["puntos"] / group["sets"].clip(lower=1)
        top_scorer = pts_per_set.max()

        # Roster depth: entropy de distribución de puntos
        total_pts = max(group["puntos"].sum(), 1)
        pts_shares = group["puntos"] / total_pts
        depth = float(entropy(pts_shares.clip(lower=1e-10)))

        # Ace threat: aces/set del mejor sacador
        aces_per_set = group["aces"] / group["sets"].clip(lower=1)
        ace_threat = aces_per_set.max()

        # Block power: bloqueos/set promedio de top 3 bloqueadores
        blk_per_set = group["bloqueos"] / group["sets"].clip(lower=1)
        block_power = blk_per_set.nlargest(3).mean()

        # Reception quality: mejor % recepción (de jugadores con >10 rec)
        receivers = group[group["recepciones"] > 10]
        if len(receivers) > 0:
            rec_quality = receivers["pct_recepcion"].max()
        else:
            rec_quality = 0.0

        roster_agg[(equipo, temp)] = {
            "top_scorer_avg": round(float(top_scorer), 3),
            "roster_depth": round(float(depth), 3),
            "ace_threat": round(float(ace_threat), 3),
            "block_power": round(float(block_power), 3),
            "rec_quality": round(float(rec_quality), 3),
        }

    # Añadir al match_df
    for prefix, team_col in [("h", "local"), ("a", "visitante")]:
        for feat in ["top_scorer_avg", "roster_depth", "ace_threat", "block_power", "rec_quality"]:
            col_name = f"{prefix}_{feat}"
            df[col_name] = df.apply(
                lambda row: roster_agg.get((row[team_col], row.get("temporada", "")), {}).get(
                    feat, 0.0
                ),
                axis=1,
            )

    # Diffs
    df["diff_top_scorer"] = df["h_top_scorer_avg"] - df["a_top_scorer_avg"]
    df["diff_roster_depth"] = df["h_roster_depth"] - df["a_roster_depth"]
    df["diff_ace_threat"] = df["h_ace_threat"] - df["a_ace_threat"]
    df["diff_block_power"] = df["h_block_power"] - df["a_block_power"]
    df["diff_rec_quality"] = df["h_rec_quality"] - df["a_rec_quality"]

    n_new = sum(1 for c in df.columns if c not in match_df.columns)
    print(f"  [roster] {n_new} features de roster añadidas")

    return df


# Roster features básicas (puntos/aces)
ROSTER_BASIC_COLS = [
    "h_top_scorer_avg",
    "a_top_scorer_avg",
    "diff_top_scorer",
    "h_roster_depth",
    "a_roster_depth",
    "diff_roster_depth",
    "h_ace_threat",
    "a_ace_threat",
    "diff_ace_threat",
]

# Roster features completas (+ bloqueos/recepción)
ROSTER_FULL_COLS = ROSTER_BASIC_COLS + [
    "h_block_power",
    "a_block_power",
    "diff_block_power",
    "h_rec_quality",
    "a_rec_quality",
    "diff_rec_quality",
]


# ─────────────────────────────────────────────────────────────
# Funciones de preparación de datos
# ─────────────────────────────────────────────────────────────


def prepare_match_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Tuple[dict, dict]:
    """
    Prepara los datos de match_features en splits train/val/test.

    Returns:
        X_splits: dict con keys 'train', 'val', 'test' → DataFrames de features
        y_splits: dict con keys 'train', 'val', 'test' → Series de target
    """
    if feature_cols is None:
        feature_cols = MATCH_FEATURE_COLS

    # Filtrar columnas que existen
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"  [WARN] Columnas no encontradas en match_features: {missing}")

    X_splits = {}
    y_splits = {}

    for split_name, years in TEMPORAL_SPLITS.items():
        mask = df["temporada_inicio"].isin(years)
        split_df = df[mask].copy()

        X = split_df[available_cols].copy()
        y = split_df[MATCH_TARGET].copy()

        # Rellenar NaN con 0 (valores faltantes en primeras jornadas)
        X = X.fillna(0)

        X_splits[split_name] = X
        y_splits[split_name] = y

        print(f"  [{split_name:5s}] {len(X):>4} partidos, " f"gana_local={y.mean():.3f}")

    return X_splits, y_splits


def prepare_set_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Tuple[dict, dict]:
    """
    Prepara los datos de set_features en splits train/val/test.
    """
    if feature_cols is None:
        feature_cols = SET_FEATURE_COLS

    # Extraer temporada de partido_id
    if "temporada_inicio" not in df.columns:
        df["temporada"] = df["partido_id"].apply(
            lambda x: str(x).split("_")[0] if "_" in str(x) else ""
        )
        df["temporada_inicio"] = df["temporada"].apply(
            lambda x: int(str(x).split("/")[0]) if "/" in str(x) else 0
        )

    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"  [WARN] Columnas no encontradas en set_features: {missing}")

    X_splits = {}
    y_splits = {}

    for split_name, years in TEMPORAL_SPLITS.items():
        mask = df["temporada_inicio"].isin(years)
        split_df = df[mask].copy()

        X = split_df[available_cols].copy()
        y = split_df[SET_TARGET].copy()

        X = X.fillna(0)

        X_splits[split_name] = X
        y_splits[split_name] = y

        print(f"  [{split_name:5s}] {len(X):>5} sets, " f"gana_local={y.mean():.3f}")

    return X_splits, y_splits


# ─────────────────────────────────────────────────────────────
# Cache / persistencia
# ─────────────────────────────────────────────────────────────


def save_splits(X_splits, y_splits, prefix: str):
    """Guarda los splits como CSV para reutilización rápida."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in X_splits:
        X_path = FEATURES_DIR / f"{prefix}_X_{split_name}.csv"
        y_path = FEATURES_DIR / f"{prefix}_y_{split_name}.csv"
        X_splits[split_name].to_csv(X_path, index=False)
        y_splits[split_name].to_csv(y_path, index=False)

    print(f"  Splits guardados en {FEATURES_DIR}")


def load_splits(prefix: str) -> Tuple[dict, dict]:
    """Carga splits previamente guardados."""
    X_splits = {}
    y_splits = {}

    for split_name in TEMPORAL_SPLITS:
        X_path = FEATURES_DIR / f"{prefix}_X_{split_name}.csv"
        y_path = FEATURES_DIR / f"{prefix}_y_{split_name}.csv"

        if not X_path.exists():
            raise FileNotFoundError(f"No se encontró {X_path}")

        X_splits[split_name] = pd.read_csv(X_path)
        y_splits[split_name] = pd.read_csv(y_path).squeeze()

    return X_splits, y_splits


# ─────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ejecutando pipeline de datos...\n")
    data = run_pipeline()

    print("\n" + "=" * 60)
    print("PREPARANDO SPLITS PARA MATCH FEATURES")
    print("=" * 60)
    X_match, y_match = prepare_match_data(data["match_features"])
    save_splits(X_match, y_match, prefix="match")

    print("\n" + "=" * 60)
    print("PREPARANDO SPLITS PARA SET FEATURES")
    print("=" * 60)
    X_set, y_set = prepare_set_data(data["set_features"])
    save_splits(X_set, y_set, prefix="set")

    print("\nFeature store completado.")
