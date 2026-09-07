"""
benchmark_teams.py — Compara la precision del modelo con 12 vs 16 equipos.

Tambien compara el efecto de anadir features de bloqueos/recepciones.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import prepare_set_data, prepare_match_data
from src.data.team_mapper import get_superliga_teams, _ALL_VIABLE_TEAMS
from src.models.benchmark import run_benchmark

CURRENT_12 = set(get_superliga_teams("2024/2025"))
ALL_16 = set(_ALL_VIABLE_TEAMS)


def filter_match_features(df, teams_set):
    """Filtra match_features para incluir solo partidos con ambos equipos en el set."""
    mask = df["local"].isin(teams_set) & df["visitante"].isin(teams_set)
    return df[mask].copy()


def filter_set_features(sf, mf, teams_set):
    """Filtra set_features usando los partido_ids de match_features filtrados."""
    valid_ids = set(filter_match_features(mf, teams_set)["partido_id"])
    return sf[sf["partido_id"].isin(valid_ids)].copy()


def run_comparison():
    print("=" * 90)
    print("  BENCHMARK: COMPARACION 12 EQUIPOS vs 16 EQUIPOS")
    print("=" * 90)

    data = run_pipeline()
    mf = data["match_features"]
    sf = data["set_features"]

    # ─── SET PREDICTOR ───
    print("\n" + "=" * 90)
    print("  SET PREDICTOR - 12 EQUIPOS (baseline)")
    print("=" * 90)

    sf_12 = filter_set_features(sf, mf, CURRENT_12)
    print(f"  Sets con 12 equipos: {len(sf_12)}")
    X_12, y_12 = prepare_set_data(sf_12)
    df_set_12 = run_benchmark(
        X_12["train"],
        y_12["train"],
        X_12["val"],
        y_12["val"],
        X_12["test"],
        y_12["test"],
    )

    print("\n" + "=" * 90)
    print("  SET PREDICTOR - 16 EQUIPOS (expandido)")
    print("=" * 90)

    sf_16 = filter_set_features(sf, mf, ALL_16)
    print(f"  Sets con 16 equipos: {len(sf_16)}")
    X_16, y_16 = prepare_set_data(sf_16)
    df_set_16 = run_benchmark(
        X_16["train"],
        y_16["train"],
        X_16["val"],
        y_16["val"],
        X_16["test"],
        y_16["test"],
    )

    # ─── MATCH PREDICTOR ───
    print("\n" + "=" * 90)
    print("  MATCH PREDICTOR - 12 EQUIPOS (baseline)")
    print("=" * 90)

    mf_12 = filter_match_features(mf, CURRENT_12)
    print(f"  Partidos con 12 equipos: {len(mf_12)}")
    X_m12, y_m12 = prepare_match_data(mf_12)
    df_match_12 = run_benchmark(
        X_m12["train"],
        y_m12["train"],
        X_m12["val"],
        y_m12["val"],
        X_m12["test"],
        y_m12["test"],
    )

    print("\n" + "=" * 90)
    print("  MATCH PREDICTOR - 16 EQUIPOS (expandido)")
    print("=" * 90)

    mf_16 = filter_match_features(mf, ALL_16)
    print(f"  Partidos con 16 equipos: {len(mf_16)}")
    X_m16, y_m16 = prepare_match_data(mf_16)
    df_match_16 = run_benchmark(
        X_m16["train"],
        y_m16["train"],
        X_m16["val"],
        y_m16["val"],
        X_m16["test"],
        y_m16["test"],
    )

    # ─── RESUMEN ───
    print("\n" + "=" * 90)
    print("  RESUMEN COMPARATIVO")
    print("=" * 90)

    best_12_set = df_set_12.loc[df_set_12["auc_test"].idxmax()]
    best_16_set = df_set_16.loc[df_set_16["auc_test"].idxmax()]
    best_12_match = df_match_12.loc[df_match_12["auc_test"].idxmax()]
    best_16_match = df_match_16.loc[df_match_16["auc_test"].idxmax()]

    print(f"\n  {'':30s} {'12 equipos':>15} {'16 equipos':>15} {'Delta':>8}")
    print("  " + "-" * 70)

    print(
        f"  {'SET - Mejor AUC':30s} "
        f"{best_12_set['auc_test']:>15.4f} "
        f"{best_16_set['auc_test']:>15.4f} "
        f"{best_16_set['auc_test'] - best_12_set['auc_test']:>+8.4f}"
    )

    print(
        f"  {'SET - Mejor Acc':30s} "
        f"{best_12_set['acc_test']:>15.4f} "
        f"{best_16_set['acc_test']:>15.4f} "
        f"{best_16_set['acc_test'] - best_12_set['acc_test']:>+8.4f}"
    )

    print(
        f"  {'MATCH - Mejor AUC':30s} "
        f"{best_12_match['auc_test']:>15.4f} "
        f"{best_16_match['auc_test']:>15.4f} "
        f"{best_16_match['auc_test'] - best_12_match['auc_test']:>+8.4f}"
    )

    print(
        f"  {'MATCH - Mejor Acc':30s} "
        f"{best_12_match['acc_test']:>15.4f} "
        f"{best_16_match['acc_test']:>15.4f} "
        f"{best_16_match['acc_test'] - best_12_match['acc_test']:>+8.4f}"
    )

    print("  " + "-" * 70)
    print(f"  {'SET - Mejor modelo (12)':30s} {best_12_set['modelo']:>15}")
    print(f"  {'SET - Mejor modelo (16)':30s} {best_16_set['modelo']:>15}")
    print(f"  {'MATCH - Mejor modelo (12)':30s} {best_12_match['modelo']:>15}")
    print(f"  {'MATCH - Mejor modelo (16)':30s} {best_16_match['modelo']:>15}")

    # Guardar
    out_dir = BASE_DIR / "models" / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_set_12.to_csv(out_dir / "set_12_teams.csv", index=False)
    df_set_16.to_csv(out_dir / "set_16_teams.csv", index=False)
    df_match_12.to_csv(out_dir / "match_12_teams.csv", index=False)
    df_match_16.to_csv(out_dir / "match_16_teams.csv", index=False)
    print(f"\n  Resultados guardados en {out_dir}")


if __name__ == "__main__":
    run_comparison()
