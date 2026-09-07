"""
set_predictor_v2.py — Adapter para el SetPredictor v2 (LogReg con recencia).

El modelo v2 se almacena como un dict joblib (no una instancia de la clase
SetPredictor) con claves: type, model, features, recency_halflife, train_seasons.

Este adapter envuelve ese dict y expone la interfaz duck-typed que esperan
MatchSimulator y SeasonSimulator:
  - .feature_names -> list[str]
  - .predict_proba(df) -> np.ndarray shape [n, 2]
  - .predict(df) -> np.ndarray int 0/1

Ver `memoria/set_predictor.md` (banner) y `memoria/mejora_precision_2026-07.md`
§6-§7.2 para el contexto completo (motivación, métricas honestas y caveats).
"""

import warnings

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Union


class LogRegSetPredictor:
    """Adapter para el SetPredictor v2 (LogisticRegression, sin scaler, sin calibración).

    El v2 fue entrenado sobre features crudas (21 columnas de SET_FEATURE_COLS)
    con pesos de recencia (half-life = 2.0 temporadas), C = 0.5, max_iter = 2000.
    No se aplicó feature scaling en entrenamiento, así que tampoco en inferencia.

    Test AUC 2025 = 0.71 (es 2025-específico; CV rolling-origin 2 folds = 0.63 ± 0.08).
    Ver `memoria/mejora_precision_2026-07.md` §7.2 para el análisis per-year.
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        recency_halflife: Optional[float] = None,
        train_seasons: Optional[list[int]] = None,
        type_: Optional[str] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.recency_halflife = recency_halflife
        self.train_seasons = train_seasons
        self.type_ = type_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities P(away), P(home).

        The input DataFrame may have columns in any order, or may have extra
        columns.  Only the columns listed in ``self.feature_names`` are used;
        any missing column is treated as 0.0.  This makes the adapter
        order-agnostic and robust at runtime.

        Returns:
            np.ndarray of shape (n, 2) — standard sklearn convention:
            column 0 = P(away), column 1 = P(home).
        """
        # Build a DataFrame with columns in the exact order the model expects
        row = {f: X.get(f, pd.Series(0.0, index=X.index)) for f in self.feature_names}
        X_reordered = pd.DataFrame(row, columns=self.feature_names)

        # Warn on schema drift: symmetric difference between the model's trained
        # feature set and the input columns.  Drift means either a newer contract
        # renamed/grew features without retraining the model, or the model was
        # retrained on a different feature set.
        input_cols_set = set(X.columns)
        model_cols_set = set(self.feature_names)
        missing = model_cols_set - input_cols_set
        extra = input_cols_set - model_cols_set
        if missing or extra:
            msg_parts = []
            if missing:
                msg_parts.append(f"missing (will fill 0.0): {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra (will ignore): {sorted(extra)}")
            warnings.warn(
                f"Schema drift between contract and v2 model: {'; '.join(msg_parts)}",
                UserWarning,
                stacklevel=2,
            )

        return self.model.predict_proba(X_reordered.fillna(0.0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class label: 0 = away wins, 1 = home wins."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:
        return (
            f"LogRegSetPredictor(type={self.type_}, "
            f"n_features={len(self.feature_names)}, "
            f"halflife={self.recency_halflife}, "
            f"train_seasons={self.train_seasons})"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogRegSetPredictor":
        """Carga el adapter v2 desde un archivo joblib (dict).

        Raises:
            ValueError: si el artefacto carece de las claves requeridas
                ``"model"`` o ``"features"``.
            Exception: cualquier error de joblib.load (archivo corrupto, etc.).
        """
        data = joblib.load(str(path))
        if "model" not in data:
            raise ValueError("v2 artifact missing required key 'model'")
        if "features" not in data:
            raise ValueError("v2 artifact missing required key 'features'")
        return cls(
            model=data["model"],
            feature_names=data["features"],
            recency_halflife=data.get("recency_halflife"),
            train_seasons=data.get("train_seasons"),
            type_=data.get("type"),
        )

    @classmethod
    def try_load_v2(
        cls,
        v2_path: Union[str, Path],
        legacy_path: Optional[Union[str, Path]] = None,
    ) -> tuple:
        """Intenta cargar el predictor v2; cae al legacy SetPredictor como fallback.

        Args:
            v2_path: ruta a ``set_predictor_v2.joblib``.
            legacy_path: ruta al legacy ``set_predictor.joblib`` (ExtraTrees).

        Returns:
            (predictor, source_label) donde source_label es uno de
            ``"logreg_v2"``, ``"extra_trees_v1"``, o ``"none"``.

        El predictor devuelto es duck-typed a la misma interfaz que el legacy:
        ``.feature_names``, ``.predict_proba(df)`` → ``[n, 2]``.
        """
        v2_path = Path(v2_path)
        if v2_path.exists():
            try:
                predictor = cls.load(v2_path)
            except Exception as exc:
                print(f"[WARN] v2 load failed, falling back to legacy: {exc}")
            else:
                return predictor, "logreg_v2"

        if legacy_path is not None:
            from src.models.set_predictor import SetPredictor as LegacySetPredictor

            legacy_path = Path(legacy_path)
            if legacy_path.exists():
                return LegacySetPredictor.load(legacy_path), "extra_trees_v1"

        return None, "none"
