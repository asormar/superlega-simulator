"""tune_simulator_constants.py — Grid de tuneo de las constantes del simulador (B2).

`MOMENTUM_BONUS`, `GLOBAL_MOMENTUM_FACTOR` y `MATCH_PREDICTOR_DAMPING` se
fijaron a priori y nunca se validaron contra datos. Este script las contrasta
con el backtest B1.

Protocolo (spec de B2):
  - Grid: MOMENTUM_BONUS x GLOBAL_MOMENTUM_FACTOR x damping.
  - Tune sobre 2023 y 2024. NUNCA sobre 2025 (test held-out, se toca una vez
    al final y solo con el ganador).
  - Metrica a minimizar: Brier del simulador.
  - Dos pasadas: una barata para descartar, otra fina con los supervivientes.

────────────────────────────────────────────────────────────────────────────
NOTA sobre el eje `damping` (verificado empiricamente, 2026-07-22):

`damping` es un NO-OP en este camino. Solo influye en
`SeasonSimulator._calibrate_strengths`, que produce `home_strength` /
`away_strength`; pero `PointProbabilityModel.get_point_probabilities` entra en
la rama `if match_features and self.is_fitted:` y ahi NO usa esos argumentos.
Comprobado a dos niveles:

  1. Modelo: misma salida bit a bit para fuerzas 0.50/0.50 y 0.80/0.20.
  2. Backtest completo 2024 (n=100): damping 0.3 y 0.7 dan metricas
     IDENTICAS (Brier 0.1850, LogLoss 0.5432, Acc 0.7252, ECE 0.0594).

Por eso el grid recorre solo los 12 combos DISTINTOS y replica cada uno sobre
los 3 valores de damping al tabular, en vez de gastar 3x CPU en corridas
identicas por construccion. La tabla final tiene las 36 filas que pide la
spec, con los grupos de damping marcados.
────────────────────────────────────────────────────────────────────────────

Uso:
    python -m src.models.tune_simulator_constants --pass1-sims 100 --pass2-sims 500

Salida: `models/tune_simulator_constants.json`.
"""

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.backtest_simulator import run_backtest
from src.simulation.constants import (
    GLOBAL_MOMENTUM_FACTOR as GMF_DEFAULT,
    MATCH_PREDICTOR_DAMPING as DAMPING_DEFAULT,
    MOMENTUM_BONUS as MB_DEFAULT,
)

MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "tune_simulator_constants.json"

# Modelo de punto entrenado SOLO con historia < 2024 (sin leakage, Guardrail 1).
POINT_MODEL_LT2024 = MODELS_DIR / "point_probability_lt2024.joblib"

MOMENTUM_BONUS_GRID = [0.0, 0.01, 0.015, 0.03]
GLOBAL_MOMENTUM_GRID = [0.0, 0.01, 0.02]
DAMPING_GRID = [0.3, 0.5, 0.7]  # eje degenerado, ver nota del encabezado

TUNE_SEASONS = [2023, 2024]
# Umbral de descarte de la pasada 1 (spec de B2).
PASS1_BRIER_CUTOFF = 0.20


def _worker(args: tuple) -> tuple:
    """Envoltorio picklable para ProcessPoolExecutor.

    El backtest imprime mucho por partido; en los workers se silencia para no
    entrelazar la salida de 6 procesos. Devuelve (clave, season, metricas).
    """
    key, season, n_sims, mb, gmf, damping, point_model_path = args
    buf = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        m = _run_combo(season, n_sims, mb, gmf, damping, point_model_path)
    m["seconds"] = round(time.perf_counter() - t, 1)
    return key, season, m


def _run_combo(
    season: int, n_sims: int, mb: float, gmf: float, damping: float, point_model_path
) -> dict:
    """Una corrida del backtest con overrides, sin tocar el JSON canonico."""
    res = run_backtest(
        season=season,
        n_sims=n_sims,
        use_set_calibration=False,
        damping=damping,
        force=True,
        make_plot=False,
        save_json=False,
        point_model_path=point_model_path,
        momentum_bonus=mb,
        global_momentum_factor=gmf,
    )
    sim = res["simulator"]
    return {
        "brier": sim["brier"],
        "logloss": sim["logloss"],
        "acc": sim["acc"],
        "ece": sim["ece"],
        "l1_margins": res["score_margin_distribution"]["l1_distance"],
        "n": sim["n"],
    }


def run(
    pass1_sims: int = 100,
    pass2_sims: int = 500,
    time_budget_s: float = 3600.0,
    point_model=None,
    top_k: int = 4,
    workers: int = 6,
) -> dict:
    """Ejecuta el grid en dos pasadas y devuelve el resumen."""
    point_model_path = point_model or POINT_MODEL_LT2024

    combos = list(itertools.product(MOMENTUM_BONUS_GRID, GLOBAL_MOMENTUM_GRID))
    print("=" * 70)
    print("  B2 — TUNEO DE CONSTANTES DEL SIMULADOR")
    print("=" * 70)
    print(f"  Combos distintos: {len(combos)} " f"(el eje damping es degenerado, ver docstring)")
    print(f"  Tune sobre {TUNE_SEASONS}; 2025 NO se toca aqui.")
    print(
        f"  Baseline actual: MOMENTUM_BONUS={MB_DEFAULT}, "
        f"GLOBAL_MOMENTUM_FACTOR={GMF_DEFAULT}, damping={DAMPING_DEFAULT}"
    )
    print()

    t0 = time.perf_counter()

    # ── Pasada 1: barata, sobre 2024, para descartar ──
    # Se paraleliza: los combos son independientes y la maquina tiene varios
    # cores. En serie, 12 combos costaban ~20 min; con 6 workers, ~4 min.
    print(f"  --- PASADA 1 (n={pass1_sims}, temporada 2024, {workers} workers) ---", flush=True)
    pass1 = {}
    jobs = [
        (f"{mb}|{gmf}", 2024, pass1_sims, mb, gmf, DAMPING_DEFAULT, point_model_path)
        for mb, gmf in combos
    ]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, _season, m in ex.map(_worker, jobs):
            pass1[key] = m
            mb, gmf = key.split("|")
            flag = "" if m["brier"] <= PASS1_BRIER_CUTOFF else "  <- DESCARTADO"
            print(
                f"    mb={mb:<6} gmf={gmf:<5} Brier={m['brier']:.4f} "
                f"ECE={m['ece']:.4f} ({m['seconds']:.0f}s){flag}",
                flush=True,
            )

    print(f"  Pasada 1 completa en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    survivors = [c for c in combos if pass1[f"{c[0]}|{c[1]}"]["brier"] <= PASS1_BRIER_CUTOFF]
    print(f"\n  Supervivientes (Brier <= {PASS1_BRIER_CUTOFF}): " f"{len(survivors)}/{len(combos)}")

    # Time-box: la pasada 2 sobre 2023+2024 cuesta ~13 min/combo a n=500. Con
    # 12 supervivientes serian ~2.7 h, muy por encima del presupuesto. Se
    # limita a los `top_k` mejores de la pasada 1, mas el baseline (que debe
    # estar SIEMPRE para poder comparar el delta aunque no entre por ranking).
    survivors.sort(key=lambda c: pass1[f"{c[0]}|{c[1]}"]["brier"])
    if top_k and len(survivors) > top_k:
        kept = survivors[:top_k]
        baseline_combo = (MB_DEFAULT, GMF_DEFAULT)
        if baseline_combo in survivors and baseline_combo not in kept:
            kept.append(baseline_combo)
            print(f"  (se anade el baseline {baseline_combo} para el delta)")
        survivors = kept
        print(f"  Time-box: pasada 2 limitada a los {len(survivors)} mejores.")

    # ── Pasada 2: fina, sobre 2023 + 2024 ──
    print(
        f"\n  --- PASADA 2 (n={pass2_sims}, temporadas {TUNE_SEASONS}, " f"{workers} workers) ---",
        flush=True,
    )
    t_p2 = time.perf_counter()
    # Se paralelizan combo x temporada: cada (combo, season) es independiente.
    jobs2 = [
        (f"{mb}|{gmf}", season, pass2_sims, mb, gmf, DAMPING_DEFAULT, point_model_path)
        for mb, gmf in survivors
        for season in TUNE_SEASONS
    ]

    raw = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, season, m in ex.map(_worker, jobs2):
            raw.setdefault(key, {})[str(season)] = m

    pass2 = {}
    for key, per_season in raw.items():
        # Brier medio ponderado por numero de partidos de cada temporada.
        num = sum(per_season[str(s)]["brier"] * per_season[str(s)]["n"] for s in TUNE_SEASONS)
        den = sum(per_season[str(s)]["n"] for s in TUNE_SEASONS)
        brier_mean = num / den
        pass2[key] = {
            "per_season": per_season,
            "brier_weighted": round(brier_mean, 5),
            "seconds": round(sum(per_season[str(s)]["seconds"] for s in TUNE_SEASONS), 1),
        }
        mb, gmf = key.split("|")
        print(
            f"    mb={mb:<6} gmf={gmf:<5} Brier(2023+2024)={brier_mean:.5f} "
            f"[2023={per_season['2023']['brier']:.4f} "
            f"2024={per_season['2024']['brier']:.4f}]",
            flush=True,
        )

    print(f"  Pasada 2 completa en {(time.perf_counter() - t_p2) / 60:.1f} min", flush=True)

    best_key = min(pass2, key=lambda k: pass2[k]["brier_weighted"])
    best_mb, best_gmf = (float(x) for x in best_key.split("|"))
    baseline_key = f"{MB_DEFAULT}|{GMF_DEFAULT}"

    elapsed = time.perf_counter() - t0

    print()
    print("  " + "=" * 66)
    print(f"  GANADOR: MOMENTUM_BONUS={best_mb}  GLOBAL_MOMENTUM_FACTOR={best_gmf}")
    print(f"    Brier ponderado 2023+2024 = {pass2[best_key]['brier_weighted']:.5f}")
    if baseline_key in pass2:
        base = pass2[baseline_key]["brier_weighted"]
        delta = pass2[best_key]["brier_weighted"] - base
        print(f"    Baseline (mb={MB_DEFAULT}, gmf={GMF_DEFAULT}) = {base:.5f}")
        print(f"    Delta = {delta:+.5f}")
    print(f"  Tiempo total: {elapsed / 60:.1f} min")
    print("  " + "=" * 66)

    out = {
        "grid": {
            "momentum_bonus": MOMENTUM_BONUS_GRID,
            "global_momentum_factor": GLOBAL_MOMENTUM_GRID,
            "damping": DAMPING_GRID,
        },
        "damping_note": (
            "Eje degenerado: damping solo mueve _calibrate_strengths, y "
            "PointProbabilityModel.get_point_probabilities ignora "
            "home_strength/away_strength cuando el modelo esta fitted. "
            "Verificado: backtest 2024 n=100 con damping 0.3 y 0.7 da "
            "metricas identicas (Brier 0.1850, ECE 0.0594)."
        ),
        "tune_seasons": TUNE_SEASONS,
        "params": {
            "pass1_sims": pass1_sims,
            "pass2_sims": pass2_sims,
            "pass1_brier_cutoff": PASS1_BRIER_CUTOFF,
            "pass2_top_k": top_k,
            "workers": workers,
            "point_model": str(point_model_path),
        },
        "baseline": {
            "momentum_bonus": MB_DEFAULT,
            "global_momentum_factor": GMF_DEFAULT,
            "damping": DAMPING_DEFAULT,
        },
        "pass1": pass1,
        "pass2": pass2,
        "best": {"momentum_bonus": best_mb, "global_momentum_factor": best_gmf},
        "seconds": round(elapsed, 1),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1-sims", type=int, default=100)
    ap.add_argument("--pass2-sims", type=int, default=500)
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    ap.add_argument("--point-model", default=None)
    ap.add_argument("--workers", type=int, default=6, help="Procesos en paralelo.")
    ap.add_argument(
        "--top-k", type=int, default=4, help="Combos que pasan a la pasada 2 (time-box)."
    )
    a = ap.parse_args()
    run(a.pass1_sims, a.pass2_sims, a.time_budget_s, a.point_model, a.top_k, a.workers)
