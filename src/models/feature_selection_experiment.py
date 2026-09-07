"""
feature_selection_experiment.py — A/B test: 87 features vs top-N (Batch 3 mid-effort #2).

The MatchPredictor in production uses 87 features (base + team stats + roster basico).
memoria/match_predictor.md notes that 87 features for 319 training samples is a
lot — there is overfitting risk. This script tests whether dropping to the
top-N most-important features (by tree-based importance) helps or hurts.

Method
------
- Same enriched pipeline as train.py: 87 features (base + team stats + roster).
- Same temporal split: train 2016-2022, val 2023, test 2024.
- Train Variant A with all 87 features → baseline (production parity).
- Get feature importance from Variant A's best model.
- Select top-N features by importance.
- Train Variant B with the top-N features.
- Compare val 2023 AUC AND test 2024 AUC.
- Save results to models/feature_selection_results.json.

Decision rule (mirrors the Optuna validation pattern in memoria/benchmark.md §7.1):
- If Variant B beats Variant A on test 2024 by >= 0.01 AUC → "improved"
- If A and B are within 0.01 AUC → "marginal"
- If B is worse by >= 0.01 AUC → "degraded"
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_match_data,
    enrich_with_team_stats,
    compute_roster_features,
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
)
from src.models.match_predictor import MatchPredictor

MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "feature_selection_results.json"


def _prepare_enriched_match_data() -> tuple:
    """Replicate the train.py pipeline: enrich with team stats + roster (87 features)."""
    data = run_pipeline()
    match_df = data["match_features"].copy()
    match_df = enrich_with_team_stats(match_df, data["team_stats"])
    match_df = compute_roster_features(match_df, data["player_stats"])
    match_cols = [
        c
        for c in MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS + ROSTER_BASIC_COLS
        if c in match_df.columns
    ]
    X_match, y_match = prepare_match_data(match_df, feature_cols=match_cols)
    return X_match, y_match, match_cols


def _train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Train a MatchPredictor on the given features; return val + test metrics."""
    predictor = MatchPredictor()
    predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    val_metrics = predictor.evaluate(X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test)
    return {
        "best_model_name": predictor.best_model_name,
        "val_auc": float(val_metrics["auc_roc"]),
        "val_acc": float(val_metrics["accuracy"]),
        "val_brier": float(val_metrics["brier_score"]),
        "test_auc": float(test_metrics["auc_roc"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_brier": float(test_metrics["brier_score"]),
        "n_features": len(predictor.feature_names),
    }


def run_experiment(n_top: int = 30) -> dict:
    """Run the A/B experiment: all 87 features vs top-N by importance."""
    print("=" * 70)
    print(f"  FEATURE SELECTION EXPERIMENT — all 87 vs top-{n_top}")
    print("=" * 70)

    print("\n[1/5] Loading enriched match data (87 features)...")
    X, y, all_cols = _prepare_enriched_match_data()
    print(
        f"  {len(all_cols)} features, {len(X['train'])} train / {len(X['val'])} val / {len(X['test'])} test"
    )

    # ─── Variant A: all 87 features (baseline) ───
    print(f"\n[2/5] Training Variant A: all {len(all_cols)} features...")
    t0 = time.time()
    variant_a = _train_and_evaluate(
        X["train"],
        y["train"],
        X["val"],
        y["val"],
        X["test"],
        y["test"],
    )
    elapsed_a = time.time() - t0
    print(f"  Champion: {variant_a['best_model_name']}")
    print(
        f"  Val  AUC: {variant_a['val_auc']:.4f}   |   Test AUC: {variant_a['test_auc']:.4f}   "
        f"({elapsed_a:.1f}s)"
    )

    # Get feature importance from Variant A
    importance_df = None
    for model_name, model_obj in [
        ("XGBoost", None),  # placeholder; we need the actual fit
    ]:
        pass
    # Re-fit just to get the importance (MatchPredictor doesn't expose the model dict)
    importance_predictor = MatchPredictor()
    importance_predictor.train(
        X_train=X["train"],
        y_train=y["train"],
        X_val=X["val"],
        y_val=y["val"],
    )
    importance_df = importance_predictor.get_feature_importance()
    top_features = list(importance_df.head(n_top)["feature"])
    print(f"\n  Top-{n_top} features (by importance):")
    for i, row in importance_df.head(n_top).iterrows():
        print(f"    {i+1:>2}. {row['feature']:<35} {row['importance']:.4f}")

    # ─── Variant B: top-N features ───
    print(f"\n[3/5] Training Variant B: top-{n_top} features...")
    X_train_top = X["train"][top_features]
    X_val_top = X["val"][top_features]
    X_test_top = X["test"][top_features]
    t0 = time.time()
    variant_b = _train_and_evaluate(
        X_train_top,
        y["train"],
        X_val_top,
        y["val"],
        X_test_top,
        y["test"],
    )
    elapsed_b = time.time() - t0
    print(f"  Champion: {variant_b['best_model_name']}")
    print(
        f"  Val  AUC: {variant_b['val_auc']:.4f}   |   Test AUC: {variant_b['test_auc']:.4f}   "
        f"({elapsed_b:.1f}s)"
    )

    # ─── Compare ───
    print("\n[4/5] Comparing variants...")
    delta_val = variant_b["val_auc"] - variant_a["val_auc"]
    delta_test = variant_b["test_auc"] - variant_a["test_auc"]
    print(f"  Delta val:  {delta_val:+.4f}")
    print(f"  Delta test: {delta_test:+.4f}")

    if delta_test > 0.01:
        verdict = "improved"
        recommendation = "apply-top-N"
    elif delta_test > -0.01:
        verdict = "marginal"
        recommendation = "keep-defaults"
    else:
        verdict = "degraded"
        recommendation = "keep-defaults"
    print(f"  Verdict: {verdict} -> {recommendation}")

    # ─── Save ───
    results = {
        "n_top": n_top,
        "all_features": all_cols,
        "top_features": top_features,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "delta_val": float(delta_val),
        "delta_test": float(delta_test),
        "verdict": verdict,
        "recommendation": recommendation,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[5/5] Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-top", type=int, default=30, help="Number of top features to keep (default 30)"
    )
    args = p.parse_args()
    run_experiment(n_top=args.n_top)
