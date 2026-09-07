"""
measure_precision.py — Medición unificada de precisión (rolling-origin).

Mide SetPredictor y MatchPredictor con el protocolo de Fase 0 usando el
estado ACTUAL del código (features, enriquecimiento, candidatos). Se corre
antes y después de cada fase para comparar de forma consistente sobre el
mismo test held-out (última temporada).

Uso:
    python -m src.models.measure_precision            # imprime tabla
    python -m src.models.measure_precision --save X   # guarda snapshot json
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_set_data,  # noqa: F401 (kept for parity)
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
    SET_FEATURE_COLS,
    enrich_with_team_stats,
    compute_roster_features,
)
from src.models.benchmark import get_all_models
from src.models.evaluation import select_champion, evaluate_on_test


def _prep_match_df(data):
    """Construye el DataFrame de match con features actuales (base+enrich+roster)."""
    df = data["match_features"].copy()
    df = enrich_with_team_stats(df, data["team_stats"])
    df = compute_roster_features(df, data["player_stats"])
    cols = [
        c for c in MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS + ROSTER_BASIC_COLS if c in df.columns
    ]
    return df, cols


def _prep_set_df(data):
    df = data["set_features"].copy()
    if "temporada_inicio" not in df.columns:
        df["temporada"] = df["partido_id"].apply(
            lambda x: str(x).split("_")[0] if "_" in str(x) else ""
        )
        df["temporada_inicio"] = df["temporada"].apply(
            lambda x: int(str(x).split("/")[0]) if "/" in str(x) else 0
        )
    cols = [c for c in SET_FEATURE_COLS if c in df.columns]
    return df, cols


def measure() -> dict:
    data = run_pipeline()
    candidates = get_all_models()  # sin Stacking para velocidad de folds
    candidates.pop("SVM_RBF", None)  # lento en CV; opcional

    snapshot = {}

    # ─── MATCH ───
    print("\n" + "=" * 80)
    print("  MATCH PREDICTOR — rolling-origin")
    print("=" * 80)
    mdf, mcols = _prep_match_df(data)
    print(
        f"  features={len(mcols)}, filas={len(mdf)}, "
        f"temporadas={sorted(mdf['temporada_inicio'].unique())}"
    )
    best_m, tabla_m = select_champion(candidates, mdf, mcols, "gana_local")
    test_m = evaluate_on_test(candidates[best_m], best_m, mdf, mcols, "gana_local")
    print(
        f"\n  TEST (temp {test_m['test_season']}, n={test_m['n_test']}): "
        f"logloss={test_m['logloss']:.4f} auc={test_m['auc']:.4f} "
        f"brier={test_m['brier']:.4f} acc={test_m['acc']:.4f}"
    )
    snapshot["match"] = {
        "champion": best_m,
        "cv": tabla_m[tabla_m["modelo"] == best_m].iloc[0].to_dict(),
        "test": test_m,
        "n_features": len(mcols),
    }

    # ─── SET ───
    print("\n" + "=" * 80)
    print("  SET PREDICTOR — rolling-origin")
    print("=" * 80)
    sdf, scols = _prep_set_df(data)
    print(
        f"  features={len(scols)}, filas={len(sdf)}, "
        f"temporadas={sorted(sdf['temporada_inicio'].unique())}"
    )
    best_s, tabla_s = select_champion(candidates, sdf, scols, "ganador_set_local")
    test_s = evaluate_on_test(candidates[best_s], best_s, sdf, scols, "ganador_set_local")
    print(
        f"\n  TEST (temp {test_s['test_season']}, n={test_s['n_test']}): "
        f"logloss={test_s['logloss']:.4f} auc={test_s['auc']:.4f} "
        f"brier={test_s['brier']:.4f} acc={test_s['acc']:.4f}"
    )
    snapshot["set"] = {
        "champion": best_s,
        "cv": tabla_s[tabla_s["modelo"] == best_s].iloc[0].to_dict(),
        "test": test_s,
        "n_features": len(scols),
    }

    return snapshot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Etiqueta del snapshot (se guarda en models/precision_<label>.json)",
    )
    args = ap.parse_args()

    snap = measure()

    if args.save:
        out = BASE_DIR / "models" / f"precision_{args.save}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        print(f"\n  Snapshot guardado en {out}")
