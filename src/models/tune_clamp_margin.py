"""tune_clamp_margin.py — Tuneo de CLAMP_MARGIN_POINT (A2, paso 3) y de
SET_BLEND_WEIGHT_ELO (A4).

El plan consolidado pide tunear el margen del clamp adaptativo en escala de
PUNTO sobre {0.05, 0.08, 0.10} usando el nivel-temporada de A5, y despues el
peso del blend sobre {0.5, 0.7, 0.9, 1.0}.

Reutiliza las funciones de `backtest_clamp` (A5) para no duplicar el
protocolo de medida: mismos equipos, mismas seeds, mismas metricas.

Uso:
    python -m src.models.tune_clamp_margin --what margin
    python -m src.models.tune_clamp_margin --what blend

Salida: `models/tune_clamp_margin_results.json` (o `..._blend_results.json`).

Metricas (identicas a A5):
  - spearman: fuerza margin-Elo vs posicion media. MAS NEGATIVO ES MEJOR
    (fuerza alta -> posicion baja). Ver backtest_clamp.py:218.
  - mean_std_total_points: estabilidad entre seeds (metrica primaria de
    varianza segun A5). Menor es mejor.
  - mean_abs_diff: |P_MC - p_elo| a nivel de par. Menor es mejor.
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.rolling_features import ELO_BASE, get_historical_team_elo
from src.models.backtest_clamp import (
    TEAMS_12,
    _load_match_predictor,
    _load_set_predictor_v2,
    _run_pair_level,
    _run_season_level,
)
from src.simulation import constants as C
from src.simulation import simulator as sim_mod
from src.simulation.feature_builder import RuntimeFeatureBuilder
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.simulator import MatchSimulator

MODELS_DIR = BASE_DIR / "models"

MARGIN_GRID = [0.05, 0.08, 0.10]
BLEND_GRID = [0.5, 0.7, 0.9, 1.0]


def _elo_to_strength(elo: float) -> float:
    """Misma conversion que usa backtest_clamp."""
    from src.data.rolling_features import elo_to_strength

    return elo_to_strength(elo)


def _setup():
    """Carga modelos, Elo y fuerzas (estado INMUTABLE, compartido)."""
    set_predictor_v2, sp_source = _load_set_predictor_v2()
    print(f"  SetPredictor v2: {sp_source}")
    match_predictor = _load_match_predictor()
    print(f"  MatchPredictor:  {'OK' if match_predictor else 'NO DISPONIBLE'}")

    elo_dict = get_historical_team_elo()
    print(f"  Elo disponible: {len(elo_dict)} equipos")

    strengths = {t: _elo_to_strength(elo_dict.get(t, ELO_BASE)) for t in TEAMS_12}
    return elo_dict, strengths, set_predictor_v2, match_predictor


def _measure(
    elo_dict, strengths, set_predictor, match_predictor, n_sims: int, n_seeds: int
) -> dict:
    """Mide nivel-par y nivel-temporada con la config actual de constantes.

    IMPORTANTE (gotcha documentado en el plan, E1): `RuntimeFeatureBuilder`
    acumula estado Elo cada vez que se simula una temporada, y
    `_init_dynamic_state` solo corre en el constructor. Si se comparte un
    unico builder entre configs, las temporadas simuladas de una config
    contaminan el Elo que ve la siguiente y las metricas dejan de ser
    comparables (sintoma observado: el nivel-par, que usa seeds FIJAS y
    deberia ser determinista, cambiaba entre corridas).

    Por eso se construye un builder y un SeasonSimulator NUEVOS por cada
    medicion, siempre sembrados desde el mismo `elo_dict` historico.
    """
    feature_builder = RuntimeFeatureBuilder(initial_elo=elo_dict)
    season_sim = SeasonSimulator(
        simulator=MatchSimulator(point_model=None, player_stats_gen=None),
        team_strengths=strengths,
        set_predictor=set_predictor,
        feature_builder=feature_builder,
        match_predictor=match_predictor,
    )

    pair = _run_pair_level(
        TEAMS_12,
        elo_dict,
        strengths,
        n_sims,
        set_predictor=set_predictor,
        feature_builder=feature_builder,
        label="NEW",
    )
    # El nivel-temporada se mide con un builder limpio aparte, para que las
    # temporadas simuladas no contaminen tampoco al nivel-par de arriba.
    fb_season = RuntimeFeatureBuilder(initial_elo=elo_dict)
    season_sim.feature_builder = fb_season
    season = _run_season_level(
        TEAMS_12,
        strengths,
        elo_dict,
        n_seeds,
        season_sim,
        use_set_calibration=True,
    )
    return {"pair_level": pair, "season_level": season}


def run(
    what: str = "margin", n_sims: int = 100, n_seeds: int = 5, time_budget_s: float = 1800.0
) -> dict:
    """Barre el grid pedido y devuelve las metricas por valor."""
    grid = MARGIN_GRID if what == "margin" else BLEND_GRID
    label = "CLAMP_MARGIN_POINT" if what == "margin" else "SET_BLEND_WEIGHT_ELO"

    print("=" * 70)
    print(f"  A2/A4 — TUNEO DE {label}")
    print("=" * 70)
    print(f"  Grid: {grid}   n_sims={n_sims}  n_seeds={n_seeds}")
    print()

    elo_dict, strengths, set_predictor, match_predictor = _setup()
    print()

    # Time-box (Guardrail 8): medir UNA config antes de lanzar el grid.
    t0 = time.perf_counter()
    original = getattr(C, label)
    # El resultado se descarta: esta corrida existe solo para CRONOMETRAR una
    # config antes de comprometerse al grid entero.
    _measure(elo_dict, strengths, set_predictor, match_predictor, n_sims, n_seeds)
    t_one = time.perf_counter() - t0
    projected = t_one * len(grid)
    print(
        f"  [time-box] 1 config: {t_one:.0f}s -> proyeccion grid "
        f"~{projected:.0f}s ({projected/60:.1f} min)"
    )
    if projected > time_budget_s:
        print(f"  ABORTADO: proyeccion supera el presupuesto {time_budget_s:.0f}s.")
        return {"aborted": True, "projected_seconds": round(projected, 1)}
    print()

    results = {}
    for val in grid:
        # Sobrescribir la constante en AMBOS namespaces: el modulo de
        # constantes y el de simulator (que la importo por valor).
        setattr(C, label, val)
        setattr(sim_mod, label, val)
        print(f"  --- {label} = {val} ---")
        t = time.perf_counter()
        m = _measure(elo_dict, strengths, set_predictor, match_predictor, n_sims, n_seeds)
        dt = time.perf_counter() - t
        sl, pl = m["season_level"], m["pair_level"]
        print(
            f"      spearman={sl['spearman']:.4f}  "
            f"std_pts={sl['mean_std_total_points']:.4f}  "
            f"|P_MC-p_elo|={pl['mean_abs_diff']:.4f}  ({dt:.0f}s)"
        )
        results[str(val)] = {**m, "time_seconds": round(dt, 1)}

    # Restaurar
    setattr(C, label, original)
    setattr(sim_mod, label, original)

    # Ganador: Spearman mas negativo (mejor fidelidad de ranking); desempate
    # por menor |P_MC - p_elo|.
    best = min(
        results.items(),
        key=lambda kv: (kv[1]["season_level"]["spearman"], kv[1]["pair_level"]["mean_abs_diff"]),
    )
    print()
    print(
        f"  GANADOR: {label} = {best[0]}  " f"(spearman={best[1]['season_level']['spearman']:.4f})"
    )

    out = {
        "parameter": label,
        "grid": grid,
        "results": results,
        "best": best[0],
        "params": {"n_sims": n_sims, "n_seeds": n_seeds},
        "criterion": "spearman mas negativo (mejor ranking); desempate por |P_MC - p_elo|",
    }
    suffix = "margin" if what == "margin" else "blend"
    path = MODELS_DIR / f"tune_clamp_{suffix}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  Resultados guardados en {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["margin", "blend"], default="margin")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--time-budget-s", type=float, default=1800.0)
    a = ap.parse_args()
    run(a.what, a.n_sims, a.n_seeds, a.time_budget_s)
