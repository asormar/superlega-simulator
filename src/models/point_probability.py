"""
point_probability.py — Modelo de probabilidad de ganar un punto.

Estima P(equipo local gana un punto) dado que está sacando o recibiendo.
Esta probabilidad alimenta el simulador Markov Chain punto a punto.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.simulation.constants import (
    DEFAULT_SIDEOUT_RATE,
    POINT_PROB_CLIP,
    POINT_RATIO_CLIP,
)

# Feature keys esperados por el modelo entrenado
_FEATURE_KEYS = [
    "elo_diff",
    "diff_win_rate_global",
    "diff_set_win_rate",
    "diff_dominancia",
    "diff_set_ratio",
    "diff_forma_efectiva",
]

# Mapeo de las 6 `_FEATURE_KEYS` (nombres del RUNTIME) a sus equivalentes en
# `build_rolling_match_features` (nombres del ENTRENAMIENTO). Guardrail 3:
# la definicion debe ser la MISMA a ambos lados.
#
# NOTA — desviacion consciente respecto al plan (B3, paso 1). El plan mapea
# `diff_dominancia` -> `diff_set_diff_exp`. Eso introduciria train/serve skew,
# porque en el runtime `diff_dominancia` NO es una diferencia de sets
# esperados: `feature_builder.py:264-266` define
#     dominancia_x = x_set_win_rate - 0.5
# luego `diff_dominancia = (h_swr - 0.5) - (a_swr - 0.5) = h_swr - a_swr`,
# que es EXACTAMENTE `diff_set_win_rate` y `diff_set_ratio`. Es decir: en
# runtime esas tres features son algebraicamente identicas. Se mapean las tres
# a `diff_set_ratio` para reproducir esa identidad en el entrenamiento. La
# colinealidad resultante es justo lo que la regularizacion L2 de Ridge
# maneja bien, y es preferible a entrenar con una senal que en produccion
# nunca se sirve.
_ROLLING_FEATURE_MAP = {
    "elo_diff": "elo_diff",
    "diff_win_rate_global": "diff_win_rate",
    "diff_set_win_rate": "diff_set_ratio",
    "diff_dominancia": "diff_set_ratio",
    "diff_set_ratio": "diff_set_ratio",
    "diff_forma_efectiva": "diff_form_ewma",
}


def build_point_training_data(max_season: Optional[int] = None) -> pd.DataFrame:
    """Construye el dataset de entrenamiento del PointProbabilityModel (B3).

    Usa features rolling PRE-partido (sin leakage) y como target el ratio de
    puntos REAL del partido, que es un outcome y por tanto valido como target.

    Args:
        max_season: si se indica, se excluyen los partidos con
            `temporada_inicio >= max_season`. OBLIGATORIO para backtestear una
            temporada sin leakage temporal (Guardrail 1): el modelo de
            produccion se entrena con todo, pero para medir en la temporada T
            hay que entrenar solo con historia < T.

    Returns:
        DataFrame con las 6 columnas de `_FEATURE_KEYS` (renombradas desde sus
        equivalentes rolling) mas `point_ratio_h` / `point_ratio_a` y
        `temporada_inicio`.
    """
    # Import diferido: backtest_simulator importa este modulo de forma lazy.
    from src.data.rolling_features import build_rolling_match_features
    from src.models.backtest_simulator import load_real_matches

    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")

    feats = build_rolling_match_features(sp)
    matches = load_real_matches(sp)

    # Clave natural, unica en ambos lados (verificado: 0 duplicados). No se
    # usa `partido_id` porque colisiona entre ida y vuelta (bug B0) y ademas
    # `load_real_matches` no lo expone.
    join_key = ["temporada_inicio", "jornada_num", "local", "visitante"]
    merged = feats.merge(
        matches[join_key + ["pts_h", "pts_a"]],
        on=join_key,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(feats):
        raise ValueError(
            f"El join perdio filas: {len(feats)} features -> {len(merged)} tras "
            "unir con los resultados reales."
        )

    out = pd.DataFrame(index=merged.index)
    for key, rolling_col in _ROLLING_FEATURE_MAP.items():
        out[key] = merged[rolling_col]

    total = (merged["pts_h"] + merged["pts_a"]).replace(0, np.nan)
    out["point_ratio_h"] = merged["pts_h"] / total
    out["point_ratio_a"] = merged["pts_a"] / total
    out["temporada_inicio"] = merged["temporada_inicio"]

    out = out.dropna(subset=["point_ratio_h"])
    if max_season is not None:
        out = out[out["temporada_inicio"] < max_season]

    return out.reset_index(drop=True)


def build_features_from_strengths(home_strength: float, away_strength: float) -> dict:
    """
    Construye el dict de features minimo para PointProbabilityModel
    a partir de las fuerzas relativas de los equipos.

    Todas las features derivan de la misma diferencia de fuerza porque
    durante un partido individual no hay contexto adicional disponible
    (sin historial de sets, sin racha reciente, etc.).

    Args:
        home_strength: fuerza del local [0, 1]
        away_strength: fuerza del visitante [0, 1]

    Returns:
        dict con las 6 features que espera el modelo entrenado.
    """
    diff = home_strength - away_strength
    return {
        "elo_diff": diff * 200,
        "diff_win_rate_global": diff,
        "diff_set_win_rate": diff,
        "diff_dominancia": diff,
        "diff_set_ratio": diff,
        "diff_forma_efectiva": diff,
    }


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class PointProbabilityModel:
    """
    Modelo que estima la probabilidad de ganar un punto para cada equipo.

    En volleyball, la probabilidad de ganar un punto depende de:
    1. La fuerza relativa de los equipos (strength_diff, elo_diff, etc.)
    2. Quién está sacando (el equipo al saque tiene ventaja con el ace,
       pero el equipo en recepción tiene la iniciativa del ataque)

    Approach:
    - Usamos las ratios de puntos por set (point_ratio_h/a) de match_features
      para derivar la probabilidad base de ganar un punto.
    - Ajustamos por sideout rate (típicamente ~60-65% en volleyball masculino
      profesional — el equipo que recibe gana el punto más a menudo).
    """

    # DEFAULT_SIDEOUT_RATE y POINT_PROB_CLIP ahora viven en src.simulation.constants
    # (centralizados en Batch 3 para evitar duplicación).

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, match_features: pd.DataFrame):
        """
        Ajusta el modelo a partir de match_features.

        Calcula las probabilidades base de ganar un punto para cada
        partido usando las ratios de puntos.
        """
        df = match_features.copy()

        # Calcular probabilidad base de punto para el local
        # point_ratio_h = puntos ganados / puntos totales del local
        # Esto da una estimación directa de P(local gana un punto)
        if "point_ratio_h" in df.columns and "point_ratio_a" in df.columns:
            # La media de point_ratio da la probabilidad base
            self.base_home_point_prob = df["point_ratio_h"].mean()
            self.base_away_point_prob = df["point_ratio_a"].mean()
        else:
            self.base_home_point_prob = 0.50
            self.base_away_point_prob = 0.50

        # Entrenar un modelo simple para ajustar la probabilidad
        # según las features del partido
        feature_cols = []
        for col in [
            "elo_diff",
            "diff_win_rate_global",
            "diff_set_win_rate",
            "diff_dominancia",
            "diff_set_ratio",
            "diff_forma_efectiva",
        ]:
            if col in df.columns:
                feature_cols.append(col)

        if feature_cols and "point_ratio_h" in df.columns:
            X = df[feature_cols].fillna(0)
            # Target: la ratio real de puntos del local (CONTINUO).
            y = df["point_ratio_h"].fillna(0.5)

            X_scaled = self.scaler.fit_transform(X)
            # B3: regresion continua en vez de LogisticRegression sobre un
            # target binarizado (y > 0.5). La binarizacion tiraba toda la
            # informacion de MAGNITUD --- un 0.51 y un 0.58 eran la misma
            # clase --- y obligaba al mapping `0.45 + 0.10 * p_dominante`,
            # que aplastaba la salida y la sesgaba. Ridge predice el ratio
            # directamente; su L2 absorbe la colinealidad de las tres
            # features que en runtime son identicas (ver _ROLLING_FEATURE_MAP).
            self.model = Ridge(alpha=1.0, random_state=42)
            self.model.fit(X_scaled, y)
            self.feature_cols = feature_cols
            self.is_fitted = True

        print(f"  [PointProbability] Base home point prob: {self.base_home_point_prob:.4f}")
        print(f"  [PointProbability] Base away point prob: {self.base_away_point_prob:.4f}")
        print(f"  [PointProbability] Default sideout rate: {DEFAULT_SIDEOUT_RATE:.2f}")

    def get_point_probabilities(
        self,
        match_features: Optional[dict] = None,
        home_strength: float = 0.5,
        away_strength: float = 0.5,
        home_sideout: float = DEFAULT_SIDEOUT_RATE,
        away_sideout: float = DEFAULT_SIDEOUT_RATE,
    ) -> dict:
        """
        Calcula las probabilidades de ganar un punto para ambos equipos,
        distinguiendo entre sacar y recibir.

        Args:
            match_features: dict con features del partido (elo_diff, etc.)
            home_strength: fuerza relativa del local [0, 1]
            away_strength: fuerza relativa del visitante [0, 1]
            home_sideout: per-team sideout rate of the local team [0, 1].
                Falls back to DEFAULT_SIDEOUT_RATE if unknown.
            away_sideout: per-team sideout rate of the visitor team [0, 1].
                Falls back to DEFAULT_SIDEOUT_RATE if unknown.

        Returns:
            dict con:
                p_home_serving: P(local gana punto | local saca)
                p_home_receiving: P(local gana punto | visitante saca)
                p_away_serving: P(visitante gana punto | visitante saca)
                p_away_receiving: P(visitante gana punto | local saca)
        """
        # Probabilidad base de punto
        if match_features and self.is_fitted:
            # Usar el modelo para ajustar según las features
            X = pd.DataFrame([match_features])[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            # B3: el modelo predice DIRECTAMENTE el ratio de puntos del local.
            # El clip es solo un salvavidas (ver POINT_RATIO_CLIP); ya no hay
            # mapping `0.45 + 0.10 * p_dominante`.
            pred = float(self.model.predict(X_scaled)[0])
            p_home_point = float(np.clip(pred, POINT_RATIO_CLIP[0], POINT_RATIO_CLIP[1]))
        else:
            # Usar strength directamente
            total = home_strength + away_strength
            if total > 0:
                p_home_point = home_strength / total
            else:
                p_home_point = 0.5

        p_away_point = 1.0 - p_home_point

        # Ajustar por sideout rate PER-TEAM (Batch 3 mid-effort).
        # En volleyball, el equipo que recibe gana ~60-65% de los rallies
        # porque puede organizar un ataque combinado. Cuando un equipo
        # saca, su probabilidad de ganar el punto es menor.
        # - Cuando LOCAL saca, la probabilidad depende del AWAY sideout
        #   (qué tan bueno es el visitante recibiendo).
        # - Cuando VISITANTE saca, depende del HOME sideout.
        p_home_serving = (
            p_home_point
            * (1 - away_sideout)
            / (p_home_point * (1 - away_sideout) + p_away_point * away_sideout)
        )
        p_home_receiving = (
            p_home_point
            * home_sideout
            / (p_home_point * home_sideout + p_away_point * (1 - home_sideout))
        )

        # Clamp para evitar probabilidades extremas
        p_home_serving = np.clip(p_home_serving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])
        p_home_receiving = np.clip(p_home_receiving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])

        return {
            "p_home_serving": p_home_serving,
            "p_home_receiving": p_home_receiving,
            "p_away_serving": 1.0 - p_home_receiving,
            "p_away_receiving": 1.0 - p_home_serving,
        }

    def save(self, path: Optional[Path] = None):
        """Guarda el modelo."""
        if path is None:
            path = MODELS_DIR / "point_probability.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": getattr(self, "feature_cols", []),
            "base_home_point_prob": self.base_home_point_prob,
            "base_away_point_prob": self.base_away_point_prob,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(save_data, path)
        print(f"  Modelo guardado en {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PointProbabilityModel":
        """Carga un modelo previamente guardado."""
        if path is None:
            path = MODELS_DIR / "point_probability.joblib"

        save_data = joblib.load(path)
        model = cls()
        model.model = save_data["model"]
        model.scaler = save_data["scaler"]
        model.feature_cols = save_data["feature_cols"]
        model.base_home_point_prob = save_data["base_home_point_prob"]
        model.base_away_point_prob = save_data["base_away_point_prob"]
        model.is_fitted = save_data["is_fitted"]
        return model
