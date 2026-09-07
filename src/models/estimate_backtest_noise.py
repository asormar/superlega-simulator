"""estimate_backtest_noise.py — Suelo de ruido del backtest (apoyo a B2).

El grid de B2 encontro diferencias de Brier de ~0.0016 entre combos. Para
saber si eso es senal o ruido de Monte Carlo hace falta el SUELO DE RUIDO:
cuanto varia la metrica al cambiar solo las semillas, con la configuracion
fija.

Se corre el mismo combo (el baseline) sobre la misma temporada variando
`seed_base`. Toda diferencia menor que ~2 sigma de esta distribucion es
indistinguible de ruido.

Uso:
    python -m src.models.estimate_backtest_noise --season 2024 --n-sims 500

Salida: `models/backtest_noise_floor.json`.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.backtest_simulator import run_backtest
from src.simulation.constants import (
    GLOBAL_MOMENTUM_FACTOR as GMF_DEFAULT,
    MATCH_PREDICTOR_DAMPING as DAMPING_DEFAULT,
    MOMENTUM_BONUS as MB_DEFAULT,
)

MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "backtest_noise_floor.json"
POINT_MODEL_LT2024 = MODELS_DIR / "point_probability_lt2024.joblib"


def _worker(args: tuple) -> tuple:
    season, n_sims, seed_base = args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_backtest(
            season=season,
            n_sims=n_sims,
            use_set_calibration=False,
            damping=DAMPING_DEFAULT,
            force=True,
            make_plot=False,
            save_json=False,
            point_model_path=POINT_MODEL_LT2024,
            momentum_bonus=MB_DEFAULT,
            global_momentum_factor=GMF_DEFAULT,
            seed_base=seed_base,
        )
    sim = res["simulator"]
    return seed_base, sim["brier"], sim["ece"], sim["acc"]


def run(season: int = 2024, n_sims: int = 500, n_seeds: int = 6, workers: int = 6) -> dict:
    seed_bases = [1000 + 7919 * k for k in range(n_seeds)]
    print("=" * 70)
    print("  SUELO DE RUIDO DEL BACKTEST (apoyo a B2)")
    print("=" * 70)
    print(f"  Temporada {season}, n_sims={n_sims}, {n_seeds} semillas base")
    print(f"  Config FIJA: mb={MB_DEFAULT}, gmf={GMF_DEFAULT}")
    print()

    t0 = time.perf_counter()
    briers, eces, accs = [], [], []
    jobs = [(season, n_sims, sb) for sb in seed_bases]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sb, brier, ece, acc in ex.map(_worker, jobs):
            briers.append(brier)
            eces.append(ece)
            accs.append(acc)
            print(
                f"    seed_base={sb:<8} Brier={brier:.5f} ECE={ece:.5f} " f"Acc={acc:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    out = {
        "season": season,
        "n_sims": n_sims,
        "seed_bases": seed_bases,
        "brier": {
            "values": [round(b, 5) for b in briers],
            "mean": round(float(np.mean(briers)), 5),
            "std": round(float(np.std(briers, ddof=1)), 5),
            "range": round(float(max(briers) - min(briers)), 5),
        },
        "ece": {
            "mean": round(float(np.mean(eces)), 5),
            "std": round(float(np.std(eces, ddof=1)), 5),
        },
        "acc": {
            "mean": round(float(np.mean(accs)), 5),
            "std": round(float(np.std(accs, ddof=1)), 5),
        },
        "seconds": round(elapsed, 1),
    }

    print()
    print(
        f"  Brier: media {out['brier']['mean']:.5f}  "
        f"std {out['brier']['std']:.5f}  rango {out['brier']['range']:.5f}"
    )
    print(f"  ECE:   media {out['ece']['mean']:.5f}  std {out['ece']['std']:.5f}")
    print(f"  Tiempo: {elapsed / 60:.1f} min")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.season, a.n_sims, a.n_seeds, a.workers)
