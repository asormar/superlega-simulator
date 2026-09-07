"""
benchmark.py — Benchmarking completo de modelos ML.

Compara múltiples algoritmos con cross-validation y split temporal.
Genera tabla comparativa con todas las métricas.
"""

import sys
import time
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
)
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


def get_all_models() -> dict:
    """Devuelve todos los modelos candidatos para benchmarking."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
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
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
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
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    }


def get_stacking_model() -> StackingClassifier:
    """Crea un modelo Stacking con los mejores estimadores base."""
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            ),
        ),
        (
            "lgbm",
            lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1,
    )


def run_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    include_stacking: bool = True,
    run_cv: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta benchmark completo de todos los modelos.

    Returns:
        DataFrame con métricas por modelo
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    models = get_all_models()
    if include_stacking:
        models["Stacking"] = get_stacking_model()

    # Modelos que necesitan datos escalados
    scaled_models = {"LogisticRegression", "SVM_RBF", "MLP"}

    results = []
    print("\n" + "=" * 90)
    print(
        f"  {'Modelo':<22} {'Acc_val':>7} {'AUC_val':>7} {'Acc_test':>8} "
        f"{'AUC_test':>8} {'Brier':>7} {'F1':>6} {'CV_5f':>7} {'Tiempo':>7}"
    )
    print("  " + "-" * 86)

    for name, model in models.items():
        t0 = time.time()

        # Seleccionar datos (escalados o no)
        if name in scaled_models:
            Xtr, Xv, Xte = X_train_sc, X_val_sc, X_test_sc
        else:
            Xtr, Xv, Xte = X_train.values, X_val.values, X_test.values

        # Entrenar
        model.fit(Xtr, y_train)

        # Predicciones
        y_pred_val = model.predict(Xv)
        y_prob_val = model.predict_proba(Xv)[:, 1]
        y_pred_test = model.predict(Xte)
        y_prob_test = model.predict_proba(Xte)[:, 1]

        # Métricas validación
        acc_val = accuracy_score(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_prob_val)

        # Métricas test
        acc_test = accuracy_score(y_test, y_pred_test)
        auc_test = roc_auc_score(y_test, y_prob_test)
        brier = brier_score_loss(y_test, y_prob_test)
        f1 = f1_score(y_test, y_pred_test, average="weighted")

        # Cross-validation (5-fold en train)
        cv_score = 0.0
        if run_cv:
            try:
                cv_scores = cross_val_score(
                    model,
                    Xtr,
                    y_train,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                cv_score = cv_scores.mean()
            except (Exception, KeyboardInterrupt):
                cv_score = 0.0

        elapsed = time.time() - t0

        results.append(
            {
                "modelo": name,
                "acc_val": acc_val,
                "auc_val": auc_val,
                "acc_test": acc_test,
                "auc_test": auc_test,
                "brier_test": brier,
                "f1_test": f1,
                "cv_5fold": cv_score,
                "tiempo_s": elapsed,
            }
        )

        print(
            f"  {name:<22} {acc_val:>7.4f} {auc_val:>7.4f} {acc_test:>8.4f} "
            f"{auc_test:>8.4f} {brier:>7.4f} {f1:>6.3f} {cv_score:>7.4f} {elapsed:>6.1f}s"
        )

    print("  " + "-" * 86)

    df = pd.DataFrame(results).sort_values("auc_test", ascending=False)

    # Marcar el mejor
    best = df.iloc[0]
    print(
        f"\n  >> MEJOR MODELO: {best['modelo']} "
        f"(Test AUC={best['auc_test']:.4f}, Acc={best['acc_test']:.4f})"
    )

    return df


def run_full_benchmark():
    """Ejecuta benchmark completo cargando datos desde el pipeline."""
    from src.data.data_pipeline import run_pipeline
    from src.data.feature_store import prepare_set_data, prepare_match_data

    print("=" * 90)
    print("  BENCHMARK COMPLETO - SuperLega Volleyball Simulator")
    print("=" * 90)

    data = run_pipeline()

    # ─── Benchmark en SET features ───
    print("\n" + "=" * 90)
    print("  BENCHMARK: SET PREDICTOR (set_features)")
    print("=" * 90)

    X_set, y_set = prepare_set_data(data["set_features"])
    df_set = run_benchmark(
        X_set["train"],
        y_set["train"],
        X_set["val"],
        y_set["val"],
        X_set["test"],
        y_set["test"],
    )

    # ─── Benchmark en MATCH features ───
    print("\n" + "=" * 90)
    print("  BENCHMARK: MATCH PREDICTOR (match_features)")
    print("=" * 90)

    X_match, y_match = prepare_match_data(data["match_features"])
    df_match = run_benchmark(
        X_match["train"],
        y_match["train"],
        X_match["val"],
        y_match["val"],
        X_match["test"],
        y_match["test"],
    )

    # Guardar resultados
    results_dir = BASE_DIR / "models" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_set.to_csv(results_dir / "set_benchmark.csv", index=False)
    df_match.to_csv(results_dir / "match_benchmark.csv", index=False)

    print(f"\n  Resultados guardados en {results_dir}")

    return df_set, df_match


if __name__ == "__main__":
    run_full_benchmark()
