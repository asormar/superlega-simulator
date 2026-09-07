"""
feature_builder.py — Constructor de features en runtime para la simulacion.

Mantiene el estado dinamico de la temporada (Elo, forma, rachas, H2H) y
combina con features estaticas (roster, team stats) para generar las 87
features que necesita el MatchPredictor en cada partido.
"""

import threading
from pathlib import Path
from collections import defaultdict
from typing import Optional
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

from src.data.team_mapper import normalize_team_name
from src.data.rolling_features import ELO_K, ELO_HOME_ADV, ELO_BASE
from src.simulation.constants import AVG_POINTS_PER_SET
from src.data.feature_store import (
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
)

# ─────────────────────────────────────────────────────────────
# Constantes generales (Elo canónico importado de rolling_features)
# ─────────────────────────────────────────────────────────────

ASSUMED_REST_DAYS = 7  # Descanso entre jornadas (simplificado)


def _elo_expected(elo_a: float, elo_b: float) -> float:
    """Probabilidad esperada de que A gane a B segun Elo."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _elo_update(elo: float, expected: float, actual: float) -> float:
    """Actualiza Elo tras un resultado."""
    return elo + ELO_K * (actual - expected)


# ─────────────────────────────────────────────────────────────
# Feature Builder
# ─────────────────────────────────────────────────────────────


class RuntimeFeatureBuilder:
    """
    Construye features de partido en runtime durante la simulacion.

    Carga perfiles estaticos desde match_features.csv y mantiene
    estado dinamico (Elo, resultados, H2H) actualizado tras cada
    partido simulado.
    """

    def __init__(self, csv_path: Optional[Path] = None, initial_elo: Optional[dict] = None):
        """
        Args:
            csv_path: CSV de perfiles estáticos (match_features.csv por defecto).
            initial_elo: ratings de Elo iniciales por equipo. Si se pasa, el
                Elo runtime NO arranca plano en 1500 sino desde el histórico
                (ver rolling_features.get_historical_team_elo). Esto hace que
                la señal de partido sea fiel desde la jornada 1 en lugar de
                calentar desde cero y diluir el prior de fuerzas. Si es None
                (default), se mantiene el arranque plano en ELO_BASE.
        """
        if csv_path is None:
            csv_path = BASE_DIR / "DB" / "features" / "match_features.csv"

        self._lock = threading.Lock()
        self._initial_elo = initial_elo
        self._load_static_profiles(csv_path)
        self._init_dynamic_state()

    # NOTE: build_features() and update() are called from FastAPI's threadpool
    # (sync endpoints run via run_in_executor).  The lock prevents concurrent
    # mutations to shared dynamic state (elo, results, streaks, h2h, etc.)
    # when multiple season simulations run in parallel.
    # Lock granularity: per-method, released between successive calls within
    # the same simulation loop.  No deadlock risk since callers never nest
    # both methods in a single lock acquisition.

    def _load_static_profiles(self, csv_path: Path):
        """Carga perfiles estaticos de equipo desde el CSV historico."""
        df = pd.read_csv(csv_path, encoding="utf-8")

        # Normalizar nombres
        if "local" in df.columns:
            df["local"] = df["local"].apply(normalize_team_name)
            df["visitante"] = df["visitante"].apply(normalize_team_name)

        # Perfiles estaticos: media de cada feature por equipo (todas las temporadas)
        self.static_profiles = {}

        # Features estaticas: todas las que no dependen del momento de la temporada
        static_feature_names = (
            MATCH_FEATURE_COLS
            + [c for c in ENRICHED_MATCH_COLS if c in df.columns]
            + [c for c in ROSTER_BASIC_COLS if c in df.columns]
        )

        for team_col, prefix in [("local", "h"), ("visitante", "a")]:
            for col in static_feature_names:
                if not col.startswith(f"{prefix}_") and not col.startswith("diff_"):
                    continue
                # Solo features de equipo (no diffs)
                if not col.startswith(f"{prefix}_"):
                    continue

                base_name = col[2:]  # quitar prefijo h_ o a_
                if base_name not in self.static_profiles:
                    self.static_profiles[base_name] = {}

                team_means = df.groupby(team_col)[col].mean()
                for team, val in team_means.items():
                    if team not in self.static_profiles[base_name]:
                        self.static_profiles[base_name][team] = float(val)

        # H2H historico
        self.historical_h2h = {}
        for _, row in df.iterrows():
            key = (row["local"], row["visitante"])
            if key not in self.historical_h2h:
                self.historical_h2h[key] = {"wins_h": 0, "total": 0}
            self.historical_h2h[key]["total"] += 1
            if row.get("gana_local", 0) == 1:
                self.historical_h2h[key]["wins_h"] += 1

        self.all_teams = set(df["local"].unique()) | set(df["visitante"].unique())
        print(f"  [FeatureBuilder] {len(self.all_teams)} equipos cargados desde CSV")

    def _init_dynamic_state(self):
        """Inicializa estado dinamico para una nueva temporada."""
        init = self._initial_elo or {}
        self.elo = {team: float(init.get(team, ELO_BASE)) for team in self.all_teams}
        self.results = defaultdict(
            list
        )  # team → [(win_bool, is_home, sets_favor, sets_contra, pts_favor, pts_contra)]
        self.h2h = {}  # (a,b) → (wins_a, total)
        self.streaks = defaultdict(int)  # team → racha (± consecutiva)
        self.standings_points = defaultdict(int)
        self.sets_won_total = defaultdict(int)
        self.sets_lost_total = defaultdict(int)
        self.current_jornada = 0

    def build_features(
        self,
        local: str,
        visitante: str,
        jornada: int,
    ) -> pd.DataFrame:
        """
        Construye las 87 features para el partido (local vs visitante).

        Returns:
            DataFrame de 1 fila listo para MatchPredictor.predict_proba()
        """
        with self._lock:
            self.current_jornada = jornada
            features = {}

            for prefix, team in [("h", local), ("a", visitante)]:
                # Features estaticas (media historica)
                for feat_name, team_vals in self.static_profiles.items():
                    col = f"{prefix}_{feat_name}"
                    features[col] = team_vals.get(team, self._default_for(feat_name))

                # Win rates dinamicos
                results = self.results.get(team, [])
                total = len(results)
                if total > 0:
                    wins = sum(1 for w, _, _, _, _, _ in results if w)
                    features[f"{prefix}_win_rate_global"] = wins / total
                    last5 = results[-5:]
                    features[f"{prefix}_win_rate_last5"] = sum(
                        1 for w, _, _, _, _, _ in last5 if w
                    ) / len(last5)
                    # Win rate home/away (con localia real)
                    home_results = [r for r in results if r[1]]  # is_home=True
                    away_results = [r for r in results if not r[1]]  # is_home=False
                    features[f"{prefix}_win_rate_home"] = sum(
                        1 for w, _, _, _, _, _ in home_results if w
                    ) / max(len(home_results), 1)
                    features[f"{prefix}_win_rate_away"] = sum(
                        1 for w, _, _, _, _, _ in away_results if w
                    ) / max(len(away_results), 1)
                    # Set win rate
                    sf = sum(sf for _, _, sf, _, _, _ in results)
                    sc = sum(sc for _, _, _, sc, _, _ in results)
                    total_sets = sf + sc
                    features[f"{prefix}_set_win_rate"] = sf / max(total_sets, 1)
                    # Set diff exp
                    features[f"{prefix}_set_diff_exp"] = (sf - sc) / max(total, 1)
                    # Puntos fav/contra (usando puntos reales, no sets)
                    pf = sum(pf for _, _, _, _, pf, _ in results)
                    pc = sum(pc for _, _, _, _, _, pc in results)
                    features[f"{prefix}_pts_fav_exp"] = pf / max(total, 1)
                    features[f"{prefix}_pts_con_exp"] = pc / max(total, 1)
                    # Forma (home/away real)
                    features[f"{prefix}_forma_home"] = (
                        (
                            sum(1 for w, _, _, _, _, _ in home_results if w)
                            / max(len(home_results), 1)
                        )
                        if home_results
                        else wins / total
                    )
                    features[f"{prefix}_forma_away"] = (
                        (
                            sum(1 for w, _, _, _, _, _ in away_results if w)
                            / max(len(away_results), 1)
                        )
                        if away_results
                        else wins / total
                    )
                    # Racha
                    features[f"{prefix}_racha"] = self.streaks.get(team, 0)
                    features[f"{prefix}_ultimo_set_diff"] = (
                        results[-1][2] - results[-1][3] if results else 0
                    )
                    # Ranking (por puntos SuperLega)
                    features[f"{prefix}_rank_season"] = self.standings_points.get(team, 0)
                else:
                    features[f"{prefix}_win_rate_global"] = 0.5
                    features[f"{prefix}_win_rate_last5"] = 0.5
                    features[f"{prefix}_win_rate_home"] = 0.5
                    features[f"{prefix}_win_rate_away"] = 0.5
                    features[f"{prefix}_set_win_rate"] = 0.5
                    features[f"{prefix}_set_diff_exp"] = 0.0
                    features[f"{prefix}_pts_fav_exp"] = AVG_POINTS_PER_SET
                    features[f"{prefix}_pts_con_exp"] = AVG_POINTS_PER_SET
                    features[f"{prefix}_forma_home"] = 0.5
                    features[f"{prefix}_forma_away"] = 0.5
                    features[f"{prefix}_racha"] = 0
                    features[f"{prefix}_ultimo_set_diff"] = 0
                    features[f"{prefix}_rank_season"] = 0

                features[f"{prefix}_descanso"] = ASSUMED_REST_DAYS
                # Elo
                features[f"elo_{prefix}"] = self.elo.get(team, ELO_BASE)

            # Diffs dinamicos
            for feat in [
                "win_rate_global",
                "win_rate_last5",
                "set_win_rate",
                "set_diff_exp",
                "pts_fav_exp",
                "pts_con_exp",
                "racha",
                "ultimo_set_diff",
                "descanso",
                "rank_season",
                "forma_efectiva",
            ]:
                h_val = features.get(f"h_{feat}", 0)
                a_val = features.get(f"a_{feat}", 0)
                features[f"diff_{feat}"] = h_val - a_val

            # Elo diffs
            features["elo_diff"] = features["elo_h"] - features["elo_a"]
            p_home = _elo_expected(features["elo_h"] + ELO_HOME_ADV, features["elo_a"])
            features["elo_win_prob_h"] = p_home
            features["elo_h_home"] = features["elo_h"] + ELO_HOME_ADV
            features["elo_a_away"] = features["elo_a"]

            # H2H (combinar historico con simulado)
            h2h_key = (local, visitante)
            hist = self.historical_h2h.get(h2h_key, {"wins_h": 0, "total": 0})
            sim = self.h2h.get(h2h_key, {"wins_h": 0, "total": 0})
            total_h2h = hist["total"] + sim["total"]
            wins_h2h = hist["wins_h"] + sim["wins_h"]
            h2h_rate = wins_h2h / max(total_h2h, 1)
            features["h_h2h_win_rate"] = h2h_rate if total_h2h > 0 else 0.5
            features["h_h2h_set_diff_exp"] = (h2h_rate - 0.5) * 2.0

            # Set ratios y dominancia (estaticos)
            features["set_ratio_h"] = features.get("h_set_win_rate", 0.5)
            features["set_ratio_a"] = features.get("a_set_win_rate", 0.5)
            features["diff_set_ratio"] = features["set_ratio_h"] - features["set_ratio_a"]

            features["point_ratio_h"] = features.get("h_point_ratio_h", 0.53)
            features["point_ratio_a"] = features.get("a_point_ratio_a", 0.52)
            features["dominancia_h"] = features.get("h_set_win_rate", 0.5) - 0.5
            features["dominancia_a"] = features.get("a_set_win_rate", 0.5) - 0.5
            features["diff_dominancia"] = features["dominancia_h"] - features["dominancia_a"]

            # SOS (estatico)
            features["sos_h"] = 0.5
            features["sos_a"] = 0.5
            features["diff_sos"] = 0.0
            # Jornada
            features["jornada_num"] = jornada

            # Rellenar cualquier feature faltante con 0
            all_needed = (
                [c for c in MATCH_FEATURE_COLS]
                + [c for c in ENRICHED_MATCH_COLS]
                + [c for c in ROSTER_BASIC_COLS]
            )
            for col in all_needed:
                if col not in features:
                    features[col] = 0.0

            return pd.DataFrame([features])

    def update(
        self,
        local: str,
        visitante: str,
        sets_local: int,
        sets_visitante: int,
        winner: str,
        points_local: int = 0,
        points_visitante: int = 0,
    ):
        """
        Actualiza el estado dinamico tras un partido simulado.

        Args:
            local: nombre del equipo local
            visitante: nombre del equipo visitante
            sets_local: sets ganados por el local
            sets_visitante: sets ganados por el visitante
            winner: "home" o "away"
            points_local: puntos de volleyball del local (si se conocen)
            points_visitante: puntos de volleyball del visitante
        """
        with self._lock:
            home_won = winner == "home"

            # Elo con margen de victoria (alineado con el modelo offline:
            # 3-0 mueve más rating que 3-2). margin_mult: 3-0→1.30,
            # 3-1→1.15, 3-2→1.00. Update zero-sum (el ganador gana lo que
            # el perdedor pierde).
            elo_h = self.elo.get(local, ELO_BASE)
            elo_a = self.elo.get(visitante, ELO_BASE)
            expected_h = _elo_expected(elo_h + ELO_HOME_ADV, elo_a)
            mov = abs(sets_local - sets_visitante)
            margin_mult = 1.0 + 0.15 * (mov - 1)
            delta = ELO_K * margin_mult * ((1.0 if home_won else 0.0) - expected_h)
            self.elo[local] = elo_h + delta
            self.elo[visitante] = elo_a - delta

            # Resultados: (win_bool, is_home, sets_favor, sets_contra, pts_favor, pts_contra)
            self.results[local].append(
                (home_won, True, sets_local, sets_visitante, points_local, points_visitante)
            )
            self.results[visitante].append(
                (not home_won, False, sets_visitante, sets_local, points_visitante, points_local)
            )

            # Rachas
            if home_won:
                if self.streaks[local] >= 0:
                    self.streaks[local] += 1
                else:
                    self.streaks[local] = 1
                if self.streaks[visitante] <= 0:
                    self.streaks[visitante] -= 1
                else:
                    self.streaks[visitante] = -1
            else:
                if self.streaks[visitante] >= 0:
                    self.streaks[visitante] += 1
                else:
                    self.streaks[visitante] = 1
                if self.streaks[local] <= 0:
                    self.streaks[local] -= 1
                else:
                    self.streaks[local] = -1

            # H2H
            h2h_key = (local, visitante)
            if h2h_key not in self.h2h:
                self.h2h[h2h_key] = {"wins_h": 0, "total": 0}
            self.h2h[h2h_key]["total"] += 1
            if home_won:
                self.h2h[h2h_key]["wins_h"] += 1

            # Puntos SuperLega (para ranking)
            if sets_local == 3 and sets_visitante <= 1:
                self.standings_points[local] += 3
            elif sets_local == 3 and sets_visitante == 2:
                self.standings_points[local] += 2
                self.standings_points[visitante] += 1
            elif sets_visitante == 3 and sets_local <= 1:
                self.standings_points[visitante] += 3
            else:
                self.standings_points[visitante] += 2
                self.standings_points[local] += 1

            # Sets totales
            self.sets_won_total[local] += sets_local
            self.sets_lost_total[local] += sets_visitante
            self.sets_won_total[visitante] += sets_visitante
            self.sets_lost_total[visitante] += sets_local

    @staticmethod
    def _default_for(feat_name: str) -> float:
        """Valor por defecto para una feature estatica desconocida."""
        defaults = {
            "point_ratio": 0.53,
            "top_scorer_avg": 3.5,
            "roster_depth": 1.5,
            "ace_threat": 0.35,
            "rec_quality": 0.35,
            "atq_pct": 0.45,
            "atq_eff": 0.28,
            "pts_set": 23.5,
            "aces_set": 1.5,
            "bloq_set": 2.0,
            "ace_ratio": 0.06,
            "rec_eff": 0.35,
        }
        for key, val in defaults.items():
            if key in feat_name:
                return val
        return 0.5
