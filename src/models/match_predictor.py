"""
match_predictor.py — Modelo de prediccion de ganador de partido.

Entrena y compara GradientBoosting, RandomForest, ExtraTrees y XGBoost
para predecir si el equipo local gana un partido completo.
Usa 87 features (base + team stats + roster basico).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


def get_match_candidate_models() -> dict:
    """Modelos candidatos para prediccion de partido."""
    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


class MatchPredictor:
    """
    Predictor del ganador de un partido de volleyball.

    Entrena multiples modelos, selecciona el mejor por AUC-ROC
    en el set de validacion, y lo calibra para obtener
    probabilidades bien calibradas.
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.calibrated_model = None
        self.feature_names = None
        self._test_metrics = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """
        Entrena todos los modelos candidatos y selecciona el mejor.
        """
        self.feature_names = list(X_train.columns)

        candidates = get_match_candidate_models()
        best_auc = -1

        print("\n  Entrenando modelos candidatos para MATCH:")
        print("  " + "-" * 65)
        print(f"  {'Modelo':<22} {'Acc':>6} {'AUC':>6} {'Brier':>7} {'Prec':>6} {'Rec':>6}")
        print("  " + "-" * 65)

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)
            brier = brier_score_loss(y_val, y_prob)

            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            precision = report["weighted avg"]["precision"]
            recall = report["weighted avg"]["recall"]

            self.models[name] = model
            self.results[name] = {
                "accuracy": acc,
                "auc_roc": auc,
                "brier_score": brier,
                "precision": precision,
                "recall": recall,
                "confusion_matrix": confusion_matrix(y_val, y_pred),
            }

            print(
                f"  {name:<22} {acc:>6.3f} {auc:>6.3f} {brier:>7.4f} "
                f"{precision:>6.3f} {recall:>6.3f}"
            )

            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name

        print("  " + "-" * 65)
        print(f"  >> Mejor modelo: {self.best_model_name} (AUC={best_auc:.4f})")

        self.best_model = self.models[self.best_model_name]

        print(f"\n  Calibrando {self.best_model_name}...")
        self.calibrated_model = CalibratedClassifierCV(self.best_model, cv=3, method="isotonic")
        self.calibrated_model.fit(X_train, y_train)

        print("  Calibracion completada.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice la probabilidad de que el equipo local gane el partido.
        Devuelve array [P(away), P(home)].
        """
        return self.calibrated_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice el ganador del partido (0 = visitante, 1 = local)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evalua el modelo calibrado en el set de test."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "brier_score": brier_score_loss(y_test, y_prob),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Visitante", "Local"],
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        print(f"\n  Evaluacion en TEST ({self.best_model_name} calibrado):")
        print(f"    Accuracy:    {results['accuracy']:.4f}")
        print(f"    AUC-ROC:     {results['auc_roc']:.4f}")
        print(f"    Brier Score: {results['brier_score']:.4f}")
        print(f"\n{results['classification_report']}")

        self._test_metrics = results
        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Devuelve la importancia de las features del mejor modelo."""
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        return df

    def save(self, path: Optional[Path] = None):
        """Guarda el predictor completo."""
        if path is None:
            path = MODELS_DIR / "match_predictor.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "results": self.results,
            "test_metrics": getattr(self, "_test_metrics", None),
        }
        joblib.dump(save_data, path)
        print(f"  Modelo guardado en {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MatchPredictor":
        """Carga un predictor previamente guardado."""
        if path is None:
            path = MODELS_DIR / "match_predictor.joblib"

        save_data = joblib.load(path)

        predictor = cls()
        predictor.best_model_name = save_data["best_model_name"]
        predictor.best_model = save_data["best_model"]
        predictor.calibrated_model = save_data["calibrated_model"]
        predictor.feature_names = save_data["feature_names"]
        predictor.results = save_data["results"]
        predictor._test_metrics = save_data.get("test_metrics")

        return predictor
