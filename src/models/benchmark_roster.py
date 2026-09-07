"""
benchmark_roster.py — Compara precision del modelo con/sin features de roster.

Compara 3 configuraciones:
1. Base: solo features de equipo (actual)
2. +Roster Basico: + top_scorer, depth, ace_threat
3. +Roster Completo: + block_power, rec_quality

Así determinamos si bloqueos/recepciones mejoran la predicción.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_match_data,
    compute_roster_features,
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
    ROSTER_FULL_COLS,
    enrich_with_team_stats,
)
from src.models.benchmark import run_benchmark


def run_roster_comparison():
    print("=" * 90)
    print("  BENCHMARK: FEATURES DE ROSTER (JUGADORES)")
    print("=" * 90)

    data = run_pipeline()
    mf = data["match_features"]
    ps = data["player_stats"]
    ts = data["team_stats"]

    # Enriquecer con team stats
    mf_enriched = enrich_with_team_stats(mf, ts)

    # Enriquecer con roster features
    mf_roster = compute_roster_features(mf_enriched, ps)

    # Base features (sin roster)
    base_cols = [c for c in MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS if c in mf_enriched.columns]

    # Base + roster basico (puntos/aces)
    basic_cols = base_cols + [c for c in ROSTER_BASIC_COLS if c in mf_roster.columns]

    # Base + roster completo (+ bloqueos/recepcion)
    full_cols = base_cols + [c for c in ROSTER_FULL_COLS if c in mf_roster.columns]

    configs = [
        ("BASE (sin roster)", mf_enriched, base_cols),
        ("+ROSTER BASICO (pts/aces)", mf_roster, basic_cols),
        ("+ROSTER COMPLETO (+bloq/rec)", mf_roster, full_cols),
    ]

    results_summary = []
    for name, df, cols in configs:
        print(f"\n{'=' * 90}")
        print(f"  MATCH PREDICTOR - {name}")
        print(f"  Features: {len(cols)}")
        print(f"{'=' * 90}")

        X, y = prepare_match_data(df, feature_cols=cols)
        result_df = run_benchmark(
            X["train"],
            y["train"],
            X["val"],
            y["val"],
            X["test"],
            y["test"],
        )

        best = result_df.iloc[0]  # sorted by auc_test desc
        results_summary.append(
            {
                "config": name,
                "n_features": len(cols),
                "best_model": best["modelo"],
                "auc_test": best["auc_test"],
                "acc_test": best["acc_test"],
                "brier": best["brier_test"],
            }
        )

    # ─── RESUMEN ───
    print("\n" + "=" * 90)
    print("  RESUMEN - IMPACTO DE FEATURES DE ROSTER")
    print("=" * 90)

    print(
        f"\n  {'Configuracion':<32s} {'Features':>8} {'Modelo':<18} {'AUC':>8} {'Acc':>8} {'Brier':>8}"
    )
    print("  " + "-" * 95)

    for r in results_summary:
        print(
            f"  {r['config']:<32s} {r['n_features']:>8} {r['best_model']:<18} "
            f"{r['auc_test']:>8.4f} {r['acc_test']:>8.4f} {r['brier']:>8.4f}"
        )

    print("  " + "-" * 95)

    base_auc = results_summary[0]["auc_test"]
    basic_auc = results_summary[1]["auc_test"]
    full_auc = results_summary[2]["auc_test"]

    print(f"\n  Delta BASE -> +ROSTER BASICO:    AUC {basic_auc - base_auc:>+.4f}")
    print(f"  Delta BASE -> +ROSTER COMPLETO:  AUC {full_auc - base_auc:>+.4f}")
    print(f"  Delta BASICO -> COMPLETO:        AUC {full_auc - basic_auc:>+.4f}")

    if full_auc > basic_auc:
        print("\n  [OK] Bloqueos/recepciones MEJORAN el modelo")
    elif full_auc == basic_auc:
        print("\n  - Bloqueos/recepciones no tienen efecto significativo")
    else:
        print("\n  [BAD] Bloqueos/recepciones NO mejoran (posible overfitting)")

    # Guardar
    out_dir = BASE_DIR / "models" / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_summary).to_csv(out_dir / "roster_comparison.csv", index=False)
    print(f"\n  Resultados guardados en {out_dir}")


if __name__ == "__main__":
    run_roster_comparison()
