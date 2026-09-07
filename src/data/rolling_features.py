"""
rolling_features.py — Features de partido rolling, sin leakage temporal.

Reconstruye las features a nivel de partido desde sets_partidos.csv (la
verdad set a set), recorriendo los partidos en orden cronológico y usando
SOLO información de partidos estrictamente anteriores. Esto elimina el
leakage de las features enriquecidas por temporada completa (Fase 1/2 del
plan) y mejora la señal base:

  - Elo con margen de victoria (T2.1)
  - Forma EWMA con half-life (T2.2)
  - H2H con decaimiento temporal (T2.3)
  - win rate / set ratio / point ratio expanding dentro de temporada

El resultado es un DataFrame de 1 fila por partido con features h_/a_/diff_
y target `gana_local`, coherente con lo que un modelo vería ANTES del partido.
"""

import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.team_mapper import normalize_team_name
from src.data.data_pipeline import load_player_stats
from src.data.roster_continuity import compute_roster_continuity
from src.data.team_id_mapper import get_canonical_team

# ─── Parámetros (tuneables; ver optimize_elo) ────────────────────────────────
# CANONICAL Elo constants — single source of truth. Other modules (e.g. the
# RuntimeFeatureBuilder) import these instead of redefining them. The values
# match what was used to compute get_historical_team_elo() and the API
# team strengths, so runtime updates stay coherent with the historical signal.
ELO_BASE = 1500.0
ELO_K = 28.0
ELO_HOME_ADV = 60.0
ELO_SEASON_REGRESS = 0.25  # al empezar temporada: elo = (1-r)*elo + r*BASE
FORM_HALFLIFE = 5.0  # partidos
H2H_HALFLIFE_SEASONS = 2.0

# Orden cronológico de las dos vueltas. El `partido_id` de sets_partidos.csv
# NO distingue ida de vuelta (abrevia nombres a 5 chars y omite `fase`), así que
# la columna `fase` es lo único que las separa temporalmente.
FASE_ORDER = {"1st half": 0, "2nd half": 1}
# Mantener alias privado por compatibilidad con callers internos que importan _FASE_ORDER
_FASE_ORDER = FASE_ORDER

# Peso del margen de victoria en el multiplicador de Elo: margen 3-0→(1+0.15*2)=1.30, 3-1→1.15, 3-2→1.0
ELO_MARGIN_WEIGHT = 0.15


def _jornada_num(j) -> int:
    """Extrae el número de jornada de strings tipo '11 Round'."""
    s = str(j)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def _aggregate_matches(sp: pd.DataFrame) -> pd.DataFrame:
    """Agrega sets_partidos (set a set) a nivel de partido.

    Agrupa por ``(partido_id, local)`` — NO solo por ``partido_id`` — porque en
    este dataset el ``partido_id`` COLISIONA la ida y la vuelta del mismo cruce
    (abrevia los nombres a 5 chars y omite la columna ``fase``), de modo que
    "A vs B" y "B vs A" comparten id. Agrupar solo por ``partido_id`` fundía los
    dos partidos, sumando sus sets y corrompiendo el target ``gana_local`` (el
    82% de los ids estaban afectados). El criterio ``(partido_id, local)`` los
    separa: cada encuentro tiene un local distinto. Con esto se pasa de 725
    "partidos" (rotos) a ~1322 partidos reales.

    Orden cronológico por ``(temporada_inicio, fase, jornada_num)``: la ida
    (1st half) va antes que la vuelta (2nd half), que comparten ``jornada_num``.
    """
    sp = sp.copy()
    sp["local"] = sp["equipo_local"].apply(normalize_team_name)
    sp["visitante"] = sp["equipo_visitante"].apply(normalize_team_name)
    sp["t"] = sp["temporada"].str.split("/").str[0].astype(int)
    sp["jnum"] = sp["jornada"].apply(_jornada_num)
    has_fase = "fase" in sp.columns
    sp["fnum"] = sp["fase"].map(lambda x: FASE_ORDER.get(x, 0)) if has_fase else 0

    g = sp.groupby(["partido_id", "local"])
    rows = []
    for (pid, _loc), grp in g:
        sets_h = int((grp["ganador_set_local"] == 1).sum())
        sets_a = int((grp["ganador_set_local"] == 0).sum())
        rows.append(
            {
                "partido_id": pid,
                "temporada_inicio": int(grp["t"].iloc[0]),
                "jornada_num": int(grp["jnum"].iloc[0]),
                "fnum": int(grp["fnum"].iloc[0]),
                "local": grp["local"].iloc[0],
                "visitante": grp["visitante"].iloc[0],
                "sets_h": sets_h,
                "sets_a": sets_a,
                "pts_h": int(grp["puntos_local"].sum()),
                "pts_a": int(grp["puntos_visitante"].sum()),
                "gana_local": 1 if sets_h > sets_a else 0,
            }
        )
    m = pd.DataFrame(rows)
    m = m.sort_values(
        ["temporada_inicio", "fnum", "jornada_num", "partido_id", "local"]
    ).reset_index(drop=True)
    return m.drop(columns=["fnum"])


def _elo_expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def elo_to_strength(elo: float) -> float:
    """Fuerza [0,1] desde el rating Elo, logística centrada en ELO_BASE.

    Mismo mapeo que ``get_historical_team_strengths`` y el cálculo inline que
    usan otros módulos. Factorizada aquí como fuente única de verdad.
    """
    return 1.0 / (1.0 + 10 ** (-(elo - ELO_BASE) / 400.0))


def margin_multiplier(mov: int) -> float:
    """Multiplicador de K de Elo según margen de sets (3-0, 3-1, 3-2).

    Args:
        mov: Diferencia absoluta de sets (1, 2 o 3).

    Returns:
        Factor multiplicador: 1.0 para 3-2, 1.15 para 3-1, 1.30 para 3-0.
    """
    return 1.0 + ELO_MARGIN_WEIGHT * (mov - 1)


def build_rolling_match_features(
    sp: pd.DataFrame,
    elo_k: float = ELO_K,
    elo_home_adv: float = ELO_HOME_ADV,
    elo_season_regress: float = ELO_SEASON_REGRESS,
    form_halflife: float = FORM_HALFLIFE,
    h2h_halflife: float = H2H_HALFLIFE_SEASONS,
) -> pd.DataFrame:
    """
    Construye features rolling pre-partido. Recorre los partidos en orden
    cronológico; para cada uno calcula features con el estado ACUMULADO
    hasta ese momento y LUEGO actualiza el estado con el resultado.
    """
    m = _aggregate_matches(sp)

    elo = defaultdict(lambda: ELO_BASE)
    last_season = {}  # team -> última temporada vista
    # historial dentro de temporada: team -> list de dicts
    season_hist = defaultdict(list)  # (team, season) -> [(win, is_home, sf, sa, pf, pa)]
    ewma_form = {}  # team -> forma EWMA (win)
    streak = defaultdict(int)
    h2h = defaultdict(list)  # (a,b) -> [(season, home_won)]

    alpha = 1 - 0.5 ** (1.0 / form_halflife)  # peso EWMA

    # ── Roster continuity (pre-season churn signal, B5) ──
    ps = load_player_stats()
    cont_df = compute_roster_continuity(ps)
    # Map equipo_id → canonical name and aggregate (safe for 1:many mappings like Cuneo)
    cont_df_flat = cont_df.reset_index()
    cont_df_flat["canonical"] = cont_df_flat["equipo_id"].map(get_canonical_team)
    cont_df_flat["temp_year"] = cont_df_flat["temporada"].str.split("/").str[0].astype(int)
    cont_agg = cont_df_flat.groupby(["canonical", "temp_year"])["continuity"].mean()
    cont_lookup = cont_agg.to_dict()
    cont_league_median = float(cont_agg.median())

    feat_rows = []
    for _, r in m.iterrows():
        h, a = r["local"], r["visitante"]
        season = r["temporada_inicio"]

        # Regresión a la media al cambiar de temporada
        for team in (h, a):
            if last_season.get(team) != season and team in last_season:
                elo[team] = (1 - elo_season_regress) * elo[team] + elo_season_regress * ELO_BASE
            last_season[team] = season

        elo_h, elo_a = elo[h], elo[a]

        # ── Features Elo ──
        p_home_elo = _elo_expected(elo_h + elo_home_adv, elo_a)
        feat = {
            "partido_id": r["partido_id"],
            "temporada_inicio": season,
            "jornada_num": r["jornada_num"],
            "local": h,
            "visitante": a,
            "gana_local": r["gana_local"],
            "elo_h": elo_h,
            "elo_a": elo_a,
            "elo_diff": elo_h - elo_a,
            "elo_win_prob_h": p_home_elo,
        }

        # ── Stats dentro de temporada (expanding, pre-partido) ──
        for prefix, team in [("h", h), ("a", a)]:
            hist = season_hist[(team, season)]
            n = len(hist)
            if n > 0:
                wins = sum(1 for x in hist if x[0])
                sf = sum(x[2] for x in hist)
                sa = sum(x[3] for x in hist)
                pf = sum(x[4] for x in hist)
                pa = sum(x[5] for x in hist)
                home_res = [x for x in hist if x[1]]
                away_res = [x for x in hist if not x[1]]
                feat[f"{prefix}_win_rate"] = wins / n
                feat[f"{prefix}_win_rate_home"] = (
                    sum(1 for x in home_res if x[0]) / len(home_res) if home_res else wins / n
                )
                feat[f"{prefix}_win_rate_away"] = (
                    sum(1 for x in away_res if x[0]) / len(away_res) if away_res else wins / n
                )
                feat[f"{prefix}_set_ratio"] = sf / max(sf + sa, 1)
                feat[f"{prefix}_point_ratio"] = pf / max(pf + pa, 1)
                feat[f"{prefix}_set_diff_exp"] = (sf - sa) / n
                feat[f"{prefix}_n_played"] = n
            else:
                feat[f"{prefix}_win_rate"] = 0.5
                feat[f"{prefix}_win_rate_home"] = 0.5
                feat[f"{prefix}_win_rate_away"] = 0.5
                feat[f"{prefix}_set_ratio"] = 0.5
                feat[f"{prefix}_point_ratio"] = 0.5
                feat[f"{prefix}_set_diff_exp"] = 0.0
                feat[f"{prefix}_n_played"] = 0
            feat[f"{prefix}_form_ewma"] = ewma_form.get(team, 0.5)
            feat[f"{prefix}_streak"] = streak[team]

        # ── H2H con decay ──
        pair_hist = h2h[(h, a)] + [(s, 1 - hw) for (s, hw) in h2h[(a, h)]]
        if pair_hist:
            num = den = 0.0
            for s, home_won in pair_hist:
                w = 0.5 ** ((season - s) / h2h_halflife)
                num += w * home_won
                den += w
            feat["h2h_win_rate_h"] = num / den if den > 0 else 0.5
        else:
            feat["h2h_win_rate_h"] = 0.5

        # ── Diffs ──
        feat["diff_win_rate"] = feat["h_win_rate"] - feat["a_win_rate"]
        feat["diff_set_ratio"] = feat["h_set_ratio"] - feat["a_set_ratio"]
        feat["diff_point_ratio"] = feat["h_point_ratio"] - feat["a_point_ratio"]
        feat["diff_form_ewma"] = feat["h_form_ewma"] - feat["a_form_ewma"]
        feat["diff_set_diff_exp"] = feat["h_set_diff_exp"] - feat["a_set_diff_exp"]
        feat["diff_streak"] = feat["h_streak"] - feat["a_streak"]
        feat["diff_win_rate_home_away"] = feat["h_win_rate_home"] - feat["a_win_rate_away"]

        # ── Roster continuity (B5) ──
        h_cont = cont_lookup.get((h, season))
        a_cont = cont_lookup.get((a, season))
        feat["h_roster_continuity"] = h_cont if h_cont is not None else cont_league_median
        feat["a_roster_continuity"] = a_cont if a_cont is not None else cont_league_median
        feat["diff_roster_continuity"] = feat["h_roster_continuity"] - feat["a_roster_continuity"]

        feat_rows.append(feat)

        # ══ ACTUALIZAR ESTADO (post-partido) ══
        home_won = r["gana_local"] == 1
        mov = abs(r["sets_h"] - r["sets_a"])  # 3,2,1
        margin_mult = margin_multiplier(mov)  # 3-0→1.30, 3-1→1.15, 3-2→1.0
        exp_h = _elo_expected(elo_h + elo_home_adv, elo_a)
        delta = elo_k * margin_mult * ((1.0 if home_won else 0.0) - exp_h)
        elo[h] = elo_h + delta
        elo[a] = elo_a - delta

        season_hist[(h, season)].append(
            (home_won, True, r["sets_h"], r["sets_a"], r["pts_h"], r["pts_a"])
        )
        season_hist[(a, season)].append(
            (not home_won, False, r["sets_a"], r["sets_h"], r["pts_a"], r["pts_h"])
        )

        ewma_form[h] = (1 - alpha) * ewma_form.get(h, 0.5) + alpha * (1.0 if home_won else 0.0)
        ewma_form[a] = (1 - alpha) * ewma_form.get(a, 0.5) + alpha * (0.0 if home_won else 1.0)

        streak[h] = (
            streak[h] + 1
            if home_won and streak[h] >= 0
            else (1 if home_won else (streak[h] - 1 if streak[h] <= 0 else -1))
        )
        streak[a] = (
            streak[a] + 1
            if (not home_won) and streak[a] >= 0
            else (1 if not home_won else (streak[a] - 1 if streak[a] <= 0 else -1))
        )

        h2h[(h, a)].append((season, 1 if home_won else 0))

    return pd.DataFrame(feat_rows)


def get_historical_team_elo(
    sp: Optional[pd.DataFrame] = None,
    elo_k: float = ELO_K,
    elo_home_adv: float = ELO_HOME_ADV,
    elo_season_regress: float = ELO_SEASON_REGRESS,
) -> dict:
    """
    Rating final de Elo con margen por equipo, replayeando todo el histórico
    (sets_partidos.csv) en orden cronológico. Ratings centrados en ELO_BASE.

    Returns:
        dict {nombre_canonico: rating_elo (float)}.
    """
    if sp is None:
        sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    m = _aggregate_matches(sp)

    elo = defaultdict(lambda: ELO_BASE)
    last_season = {}
    for _, r in m.iterrows():
        h, a, s = r["local"], r["visitante"], r["temporada_inicio"]
        for t in (h, a):
            if last_season.get(t) != s and t in last_season:
                elo[t] = (1 - elo_season_regress) * elo[t] + elo_season_regress * ELO_BASE
            last_season[t] = s
        eh, ea = elo[h], elo[a]
        exp = _elo_expected(eh + elo_home_adv, ea)
        mov = abs(r["sets_h"] - r["sets_a"])
        mm = margin_multiplier(mov)
        d = elo_k * mm * ((1.0 if r["gana_local"] == 1 else 0.0) - exp)
        elo[h] = eh + d
        elo[a] = ea - d

    return dict(elo)


def get_historical_team_strengths(sp: Optional[pd.DataFrame] = None) -> dict:
    """
    Fuerza [0,1] por equipo derivada del rating final de Elo con margen.

    Mapea el rating de `get_historical_team_elo` a [0,1] con una logística
    centrada en ELO_BASE (escala 400). Prior de fuerza mucho más fiel que el
    win-rate plano: recencia (regresión entre temporadas) + margen de victoria.

    Returns:
        dict {nombre_canonico: fuerza en [0,1]}.
    """
    elo = get_historical_team_elo(sp)
    return {t: elo_to_strength(e) for t, e in elo.items()}


ROLLING_MATCH_COLS = [
    "elo_h",
    "elo_a",
    "elo_diff",
    "elo_win_prob_h",
    "h_win_rate",
    "a_win_rate",
    "diff_win_rate",
    "h_win_rate_home",
    "a_win_rate_away",
    "diff_win_rate_home_away",
    "h_set_ratio",
    "a_set_ratio",
    "diff_set_ratio",
    "h_point_ratio",
    "a_point_ratio",
    "diff_point_ratio",
    "h_set_diff_exp",
    "a_set_diff_exp",
    "diff_set_diff_exp",
    "h_form_ewma",
    "a_form_ewma",
    "diff_form_ewma",
    "h_streak",
    "a_streak",
    "diff_streak",
    "h2h_win_rate_h",
    "jornada_num",
]


if __name__ == "__main__":
    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    feats = build_rolling_match_features(sp)
    print(f"Partidos con features rolling: {len(feats)}")
    print(f"Temporadas: {sorted(feats['temporada_inicio'].unique())}")
    print(f"Features: {len(ROLLING_MATCH_COLS)}")
    # sanity: correlación de elo_win_prob_h con resultado
    from sklearn.metrics import roc_auc_score, log_loss

    mask = feats["temporada_inicio"] >= 2020
    y = feats.loc[mask, "gana_local"]
    p = feats.loc[mask, "elo_win_prob_h"].clip(1e-6, 1 - 1e-6)
    print(f"\nElo puro (temp>=2020): AUC={roc_auc_score(y, p):.4f} logloss={log_loss(y, p):.4f}")
