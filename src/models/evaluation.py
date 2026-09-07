"""
evaluation.py — Protocolo de evaluación rolling-origin (Fase 0 del plan).

Reemplaza el val único de 81 partidos (ruido puro, error estándar ~±0.06)
por validación expanding-window sobre múltiples temporadas. La selección
de modelo campeón y de hiperparámetros se hace por MEDIA de log-loss sobre
los folds, no por AUC sobre un año suelto.

Split:
  - Folds (expanding): train ≤ T-1  →  validar en T
  - Test final intocable: la última temporada disponible.

Métrica primaria: log-loss (calidad de probabilidad, que es lo que alimenta
el simulador Monte Carlo). Secundaria: AUC. Se reporta media ± std entre folds.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


# Modelos que necesitan escalado (lineales / SVM / redes)
_SCALED_MODELS = {"LogisticRegression", "SVM_RBF", "MLP"}


def rolling_origin_folds(
    seasons: list[int],
    n_val_folds: int = 4,
    min_train_seasons: int = 3,
) -> tuple[list[tuple[list[int], int]], int]:
    """
    Construye los folds expanding-window y separa la temporada de test.

    Args:
        seasons: lista ordenada de temporadas (años de inicio) disponibles.
        n_val_folds: cuántas temporadas usar como folds de validación.
        min_train_seasons: mínimo de temporadas en el primer train.

    Returns:
        (folds, test_season) donde:
          folds = [(train_years, val_year), ...]
          test_season = última temporada (held-out).
    """
    seasons = sorted(seasons)
    test_season = seasons[-1]
    pool = seasons[:-1]  # todo menos el test

    # Las últimas n_val_folds temporadas del pool son folds de validación
    val_years = pool[-n_val_folds:]
    folds = []
    for vy in val_years:
        train_years = [s for s in pool if s < vy]
        if len(train_years) >= min_train_seasons:
            folds.append((train_years, vy))
    return folds, test_season


def _fit_predict(model, X_tr, y_tr, X_ev, name, sample_weight=None):
    """Entrena y devuelve probabilidades P(clase=1) sobre X_ev."""
    from sklearn.base import clone

    m = clone(model)
    if name in _SCALED_MODELS:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_ev = scaler.transform(X_ev)
    else:
        X_tr = X_tr.values if hasattr(X_tr, "values") else X_tr
        X_ev = X_ev.values if hasattr(X_ev, "values") else X_ev
    if sample_weight is not None:
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        m.fit(X_tr, y_tr)
    return m.predict_proba(X_ev)[:, 1]


def evaluate_model_rolling(
    model,
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    season_col: str = "temporada_inicio",
    n_val_folds: int = 4,
    sample_weight_fn: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None,
) -> dict:
    """
    Evalúa un modelo con rolling-origin sobre los folds de validación.

    Args:
        sample_weight_fn: opcional, (train_df, val_year) -> pesos de muestra.

    Returns:
        dict con métricas media±std sobre folds: logloss, auc, brier, acc.
    """
    seasons = sorted(df[season_col].dropna().unique().tolist())
    folds, _ = rolling_origin_folds(seasons, n_val_folds=n_val_folds)

    per_fold = {"logloss": [], "auc": [], "brier": [], "acc": []}
    for train_years, val_year in folds:
        tr = df[df[season_col].isin(train_years)]
        va = df[df[season_col] == val_year]
        if len(va) == 0 or len(tr) == 0:
            continue
        X_tr, y_tr = tr[feature_cols].fillna(0), tr[target]
        X_va, y_va = va[feature_cols].fillna(0), va[target]
        if y_va.nunique() < 2:
            continue

        sw = sample_weight_fn(tr, val_year) if sample_weight_fn else None
        p = _fit_predict(model, X_tr, y_tr, X_va, name, sample_weight=sw)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        per_fold["logloss"].append(log_loss(y_va, p))
        per_fold["auc"].append(roc_auc_score(y_va, p))
        per_fold["brier"].append(brier_score_loss(y_va, p))
        per_fold["acc"].append(accuracy_score(y_va, (p >= 0.5).astype(int)))

    out = {"n_folds": len(per_fold["logloss"])}
    for k, v in per_fold.items():
        out[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
        out[f"{k}_std"] = float(np.std(v)) if v else float("nan")
    return out


def select_champion(
    candidates: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    season_col: str = "temporada_inicio",
    n_val_folds: int = 4,
    sample_weight_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> tuple[str, pd.DataFrame]:
    """
    Evalúa todos los candidatos con rolling-origin y elige el campeón por
    MENOR log-loss medio (métrica primaria).

    Returns:
        (best_name, tabla_ordenada)
    """
    rows = []
    for name, model in candidates.items():
        res = evaluate_model_rolling(
            model,
            name,
            df,
            feature_cols,
            target,
            season_col=season_col,
            n_val_folds=n_val_folds,
            sample_weight_fn=sample_weight_fn,
        )
        res["modelo"] = name
        rows.append(res)

    tabla = pd.DataFrame(rows).sort_values("logloss_mean").reset_index(drop=True)
    cols = [
        "modelo",
        "logloss_mean",
        "logloss_std",
        "auc_mean",
        "auc_std",
        "brier_mean",
        "acc_mean",
        "n_folds",
    ]
    tabla = tabla[cols]

    if verbose:
        print("\n  Rolling-origin (media ± std sobre folds):")
        print("  " + "-" * 78)
        print(f"  {'Modelo':<20} {'LogLoss':>14} {'AUC':>14} {'Brier':>8} {'Acc':>7}")
        print("  " + "-" * 78)
        for _, r in tabla.iterrows():
            print(
                f"  {r['modelo']:<20} "
                f"{r['logloss_mean']:.4f}±{r['logloss_std']:.3f}   "
                f"{r['auc_mean']:.4f}±{r['auc_std']:.3f}   "
                f"{r['brier_mean']:.4f}  {r['acc_mean']:.3f}"
            )
        print("  " + "-" * 78)

    best_name = tabla.iloc[0]["modelo"]
    if verbose:
        print(f"  >> Campeón (menor logloss): {best_name}")
    return best_name, tabla


def evaluate_on_test(
    model,
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    season_col: str = "temporada_inicio",
    sample_weight_fn: Optional[Callable] = None,
) -> dict:
    """
    Entrena en todas las temporadas menos la última y evalúa en la última
    (test held-out). Se llama UNA sola vez por fase.
    """
    seasons = sorted(df[season_col].dropna().unique().tolist())
    _, test_season = rolling_origin_folds(seasons)
    tr = df[df[season_col] < test_season]
    te = df[df[season_col] == test_season]

    X_tr, y_tr = tr[feature_cols].fillna(0), tr[target]
    X_te, y_te = te[feature_cols].fillna(0), te[target]

    sw = sample_weight_fn(tr, test_season) if sample_weight_fn else None
    p = _fit_predict(model, X_tr, y_tr, X_te, name, sample_weight=sw)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    return {
        "test_season": int(test_season),
        "n_test": int(len(te)),
        "logloss": float(log_loss(y_te, p)),
        "auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
        "acc": float(accuracy_score(y_te, (p >= 0.5).astype(int))),
    }
