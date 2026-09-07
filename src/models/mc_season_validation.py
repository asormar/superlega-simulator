"""mc_season_validation.py — Validacion Monte Carlo de N temporadas (A6).

Regenera la tabla de posicion media por equipo que cita
`memoria/mejora_precision_2026-07.md` §7.1. La corrida original quedo
invalidada dos veces: primero por el bug `Optional` (sembrado de Elo plano) y
despues por el clamp mal escalado que A2/A4 han corregido.

Protocolo (el que pide el plan consolidado, A6 paso 2):
  - 12 equipos, temporada completa (ida y vuelta), seeds 0..N-1.
  - Elo sembrado desde el historico real (`get_historical_team_elo`).
  - Clamp segun la config final de A2/A4 (SET_BLEND_WEIGHT_ELO=1.0).

IMPORTANTE: cada temporada estrena `RuntimeFeatureBuilder`. El builder acumula
estado Elo y `_init_dynamic_state` solo corre en el constructor, asi que
reutilizarlo haria que las primeras temporadas contaminasen a las ultimas
(gotcha documentado en el plan, E1).

Uso:
    python -m src.models.mc_season_validation --n-seeds 20

Salida: `models/mc_season_validation.json`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.rolling_features import ELO_BASE, elo_to_strength, get_historical_team_elo
from src.models.backtest_clamp import (
    TEAMS_12,
    _get_points,
    _load_match_predictor,
    _load_set_predictor_v2,
)
from src.models.backtest_simulator import _load_point_model
from src.simulation.constants import CLAMP_MARGIN_POINT, SET_BLEND_WEIGHT_ELO
from src.simulation.feature_builder import RuntimeFeatureBuilder
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.simulator import MatchSimulator

MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "mc_season_validation.json"


def run(n_seeds: int = 20, use_set_calibration: bool = True, max_seconds: float = 1800.0) -> dict:
    """Corre `n_seeds` temporadas completas y agrega posicion media por equipo."""
    print("=" * 70)
    print(f"  A6 — VALIDACION MONTE CARLO: {n_seeds} TEMPORADAS COMPLETAS")
    print("=" * 70)
    print(f"  Equipos: {len(TEAMS_12)}  |  ida y vuelta  |  seeds 0..{n_seeds - 1}")
    print(
        f"  Config clamp: SET_BLEND_WEIGHT_ELO={SET_BLEND_WEIGHT_ELO} "
        f"CLAMP_MARGIN_POINT={CLAMP_MARGIN_POINT} "
        f"set_calibration={'ON' if use_set_calibration else 'OFF'}"
    )
    print()

    set_predictor, sp_source = _load_set_predictor_v2()
    print(f"  SetPredictor v2: {sp_source}")
    match_predictor = _load_match_predictor()
    print(f"  MatchPredictor:  {'OK' if match_predictor else 'NO DISPONIBLE'}")

    # El PointProbabilityModel es parte del camino de PRODUCCION (el API lo
    # inyecta en el MatchSimulator, main.py:112). La primera version de este
    # script pasaba point_model=None y por tanto medía `_default_point_probs`,
    # no el modelo entrenado: la medida no era fiel a produccion y era ciega a
    # cambios como B3. Se carga aqui.
    point_model = _load_point_model()
    print(f"  PointProbability: {'OK' if point_model else 'NO DISPONIBLE (fallback)'}")

    elo_dict = get_historical_team_elo()
    strengths = {t: elo_to_strength(elo_dict.get(t, ELO_BASE)) for t in TEAMS_12}
    print(f"  Elo historico: {len(elo_dict)} equipos")
    print()

    positions = {t: [] for t in TEAMS_12}
    points = {t: [] for t in TEAMS_12}

    t_start = time.perf_counter()
    for s in range(n_seeds):
        # Estado LIMPIO por temporada (ver docstring).
        season_sim = SeasonSimulator(
            simulator=MatchSimulator(point_model=point_model, player_stats_gen=None),
            team_strengths=strengths,
            set_predictor=set_predictor,
            feature_builder=RuntimeFeatureBuilder(initial_elo=elo_dict),
            match_predictor=match_predictor,
        )
        result = season_sim.simulate_season(
            teams=TEAMS_12,
            half=None,  # temporada completa: ida y vuelta
            seed=s,
            use_set_calibration=use_set_calibration,
            use_match_predictor=True,
        )
        for pos, entry in enumerate(result["standings"], 1):
            team = entry.team if hasattr(entry, "team") else entry["equipo"]
            positions[team].append(pos)
            points[team].append(_get_points(entry))

        # Time-box (Guardrail 8): proyectar tras la primera temporada.
        if s == 0:
            dt = time.perf_counter() - t_start
            projected = dt * n_seeds
            print(
                f"  [time-box] 1 temporada: {dt:.1f}s -> proyeccion "
                f"~{projected:.0f}s ({projected / 60:.1f} min)"
            )
            if projected > max_seconds:
                print(f"  ABORTADO: proyeccion supera {max_seconds:.0f}s.")
                return {"aborted": True, "projected_seconds": round(projected, 1)}

    elapsed = time.perf_counter() - t_start

    rows = []
    for t in TEAMS_12:
        rows.append(
            {
                "equipo": t,
                "posicion_media": round(float(np.mean(positions[t])), 2),
                "posicion_std": round(float(np.std(positions[t])), 2),
                "puntos_medios": round(float(np.mean(points[t])), 1),
                "fuerza_margin_elo": round(strengths[t], 3),
            }
        )
    rows.sort(key=lambda r: r["posicion_media"])

    rho, pval = spearmanr(
        [r["fuerza_margin_elo"] for r in rows],
        [r["posicion_media"] for r in rows],
    )
    mean_std_pos = float(np.mean([r["posicion_std"] for r in rows]))

    print()
    print("  " + "=" * 62)
    print(f"  {'#':<4}{'Equipo':<16}{'Pos media':>10}{'Pos std':>10}{'Fuerza':>10}")
    print("  " + "-" * 62)
    for i, r in enumerate(rows, 1):
        print(
            f"  {i:<4}{r['equipo']:<16}{r['posicion_media']:>10.1f}"
            f"{r['posicion_std']:>10.2f}{r['fuerza_margin_elo']:>10.3f}"
        )
    print("  " + "=" * 62)
    print(f"  Spearman fuerza->posicion: {rho:.4f} (p={pval:.2e})")
    print(f"  Std de posicion media:     {mean_std_pos:.3f}")
    print(f"  Tiempo: {elapsed:.0f}s ({elapsed / n_seeds:.1f}s/temporada)")

    out = {
        "n_seeds": n_seeds,
        "n_teams": len(TEAMS_12),
        "half": "full (ida y vuelta)",
        "config": {
            "set_blend_weight_elo": SET_BLEND_WEIGHT_ELO,
            "clamp_margin_point": CLAMP_MARGIN_POINT,
            "use_set_calibration": use_set_calibration,
            "set_predictor_source": sp_source,
            "point_model": point_model is not None,
        },
        "spearman_fuerza_posicion": round(float(rho), 5),
        "spearman_pvalue": float(pval),
        "mean_std_position": round(mean_std_pos, 4),
        "seconds": round(elapsed, 1),
        "standings": rows,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--no-set-calibration", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    a = ap.parse_args()
    run(a.n_seeds, not a.no_set_calibration, a.max_seconds)
