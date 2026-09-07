"""
hyperparameter_search.py — Optuna-based hyperparameter search for SET and MATCH champions.

Quick Win 2 (Batch 3): explores if Optuna can improve the AUC of the current
champion models (ExtraTrees for SET, XGBoost for MATCH) over the defaults
in benchmark.py.

Method:
- TPE sampler, seed=42, max 30 trials per study, 10 min timeout
- Train split: 2016-2022 (per feature_store.TEMPORAL_SPLITS)
- Val split: 2023 (Optuna's objective)
- Test split: 2024 (NEVER touched during search — held out for follow-up eval)
- Best params saved to models/best_params.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

# Ensure project root is on the path (per AGENTS.md convention)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_set_data,
    prepare_match_data,
    enrich_with_team_stats,
    compute_roster_features,
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
)

MODELS_DIR = BASE_DIR / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"


# ─────────────────────────────────────────────────────────────
# Default params (mirror of benchmark.py:36-65)
# ─────────────────────────────────────────────────────────────

DEFAULT_SET_EXTRATREES = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_MATCH_XGBOOST = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}


# ─────────────────────────────────────────────────────────────
# Default-baseline evaluation
# ─────────────────────────────────────────────────────────────


def _evaluate_default_set(X_train, y_train, X_val, y_val):
    model = ExtraTreesClassifier(**DEFAULT_SET_EXTRATREES)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, probs))


def _evaluate_default_match(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(**DEFAULT_MATCH_XGBOOST)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, probs))


# ─────────────────────────────────────────────────────────────
# Optuna objectives
# ─────────────────────────────────────────────────────────────


def _objective_set_extratrees(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, probs))


def _objective_match_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, probs))


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────


def run_search(n_trials: int = 30, timeout_sec: int = 600) -> dict:
    """
    Run Optuna search for SET (ExtraTrees) and MATCH (XGBoost) champion models.

    Returns dict with best params, default AUC, Optuna AUC, and delta for each model.
    """
    # Public loader used by set_predictor / match_predictor at train time
    # (re-exported via BEST_PARAMS_PATH + load_best_params below).

    print("=" * 70)
    print("  OPTUNA HYPERPARAMETER SEARCH — SET + MATCH")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    data = run_pipeline()

    print("\n[2/4] Preparing splits (train 2016-2022, val 2023)...")
    X_set, y_set = prepare_set_data(data["set_features"])

    # MATCH: enrich with team stats + roster, same as train.py (87 features total).
    # Earlier runs used basic features only; we re-run with production-grade data
    # so Optuna's params match what the actual MatchPredictor sees at inference time.
    match_df = data["match_features"].copy()
    match_df = enrich_with_team_stats(match_df, data["team_stats"])
    match_df = compute_roster_features(match_df, data["player_stats"])
    match_cols = [
        c
        for c in MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS + ROSTER_BASIC_COLS
        if c in match_df.columns
    ]
    print(f"  [match] {len(match_cols)} features (enriched = base + team stats + roster)")
    X_match, y_match = prepare_match_data(match_df, feature_cols=match_cols)

    # Optuna should not print per-trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = {}

    # ─── SET: ExtraTrees ───
    print("\n[3/4] Searching ExtraTrees for SET...")
    t0 = time.time()
    default_auc_set = _evaluate_default_set(
        X_set["train"],
        y_set["train"],
        X_set["val"],
        y_set["val"],
    )
    print(f"  Default AUC (val 2023): {default_auc_set:.4f}")

    study_set = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_set.optimize(
        lambda trial: _objective_set_extratrees(
            trial,
            X_set["train"],
            y_set["train"],
            X_set["val"],
            y_set["val"],
        ),
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=False,
    )
    elapsed_set = time.time() - t0
    print(f"  Optuna AUC  (val 2023): {study_set.best_value:.4f}")
    print(
        f"  Delta: {study_set.best_value - default_auc_set:+.4f}  "
        f"({len(study_set.trials)} trials, {elapsed_set:.1f}s)"
    )
    results["set_extratrees"] = {
        "default_auc": default_auc_set,
        "optuna_auc": float(study_set.best_value),
        "delta": float(study_set.best_value - default_auc_set),
        "best_params": study_set.best_params,
    }

    # ─── MATCH: XGBoost ───
    print("\n[4/4] Searching XGBoost for MATCH...")
    t0 = time.time()
    default_auc_match = _evaluate_default_match(
        X_match["train"],
        y_match["train"],
        X_match["val"],
        y_match["val"],
    )
    print(f"  Default AUC (val 2023): {default_auc_match:.4f}")

    study_match = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_match.optimize(
        lambda trial: _objective_match_xgboost(
            trial,
            X_match["train"],
            y_match["train"],
            X_match["val"],
            y_match["val"],
        ),
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=False,
    )
    elapsed_match = time.time() - t0
    print(f"  Optuna AUC  (val 2023): {study_match.best_value:.4f}")
    print(
        f"  Delta: {study_match.best_value - default_auc_match:+.4f}  "
        f"({len(study_match.trials)} trials, {elapsed_match:.1f}s)"
    )
    results["match_xgboost"] = {
        "default_auc": default_auc_match,
        "optuna_auc": float(study_match.best_value),
        "delta": float(study_match.best_value - default_auc_match),
        "best_params": study_match.best_params,
    }

    # Persist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Best params saved to {BEST_PARAMS_PATH}")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    for name, r in results.items():
        d = r["delta"]
        if d > 0.01:
            verdict = "IMPROVED (>+0.01 AUC)"
        elif d > -0.01:
            verdict = "marginal (|delta| <= 0.01)"
        else:
            verdict = "DEGRADED (<-0.01 AUC)"
        print(
            f"  {name:<22} {r['default_auc']:.4f} -> {r['optuna_auc']:.4f}  "
            f"({d:+.4f})  {verdict}"
        )

    return results


# ─────────────────────────────────────────────────────────────
# Public loader (used by set_predictor / match_predictor at train time)
# ─────────────────────────────────────────────────────────────


def load_best_params(model_key: Optional[str] = None):
    """
    Load Optuna best params from models/best_params.json.

    Args:
        model_key: e.g. "set_extratrees" or "match_xgboost".
                   If None, returns the full results dict.

    Returns:
        The best_params dict for that model, or None if the JSON is missing
        or the key isn't present. Callers should fall back to defaults.
    """
    if not BEST_PARAMS_PATH.exists():
        return None
    try:
        with open(BEST_PARAMS_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if model_key is None:
        return data
    entry = data.get(model_key)
    if entry is None:
        return None
    return entry.get("best_params")


if __name__ == "__main__":
    run_search()
