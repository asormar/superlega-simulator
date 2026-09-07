"""
backtest_clamp.py — A5: Backtest reproducible del clamp [0.20, 0.80].

Mide el impacto del clamp adaptativo del SetPredictor en el comportamiento
del simulador a nivel de partido (132 pares ordenados) y de temporada
(>= 10 seeds de ida simple), para tres configuraciones:
  - OFF: clamp estatico DEFAULT_CLAMP_RANGE (sin SetPredictor).
  - ON:  clamp adaptativo con SetPredictor v2 (LogReg con recencia).
  - NEW: placeholder para A3 (train/serve skew fix), null hasta que exista.

Metricas de estabilidad a nivel temporada:
  - mean_std_position: metrica secundaria. Esperada 0.0 para ON debido a la
    saturacion del clamp adaptativo per-point (clamp=[0.674, 0.900] para
    predicciones confiadas), que hace el ganador del set ~99% determinista.
    El seed SI se honra (los scores de cada set varian), pero los ganadores
    no, por lo que la posicion en tabla es 100% determinista.
  - mean_std_total_points: metrica primaria de estabilidad. Captura la
    variacion en puntos de clasificacion entre seeds (resultados 3-0/3-1/3-2),
    que SI varian bajo el clamp.

Time-box de 15 minutos: si la proyeccion excede, reduce parametros
automaticamente (n_sims 200->50, n_seeds 10->3). Si aun asi excede,
aborta con mensaje claro.

Uso:
    python -m src.models.backtest_clamp
    python -m src.models.backtest_clamp --n-sims 50 --n-seeds 3
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.rolling_features import (
    get_historical_team_elo,
    _elo_expected,
    elo_to_strength,
    ELO_BASE,
    ELO_HOME_ADV,
)
from src.simulation.simulator import MatchSimulator
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.feature_builder import RuntimeFeatureBuilder
from src.simulation.constants import DEFAULT_CLAMP_RANGE
from src.models.match_predictor import MatchPredictor
from src.models.set_predictor_v2 import LogRegSetPredictor

# ─── Rutas ──────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "backtest_clamp_results.json"

# Primeros 12 equipos de _STRENGTH_DEFAULTS (main.py:156-159)
TEAMS_12 = [
    "Trento",
    "Perugia",
    "Verona",
    "Piacenza",
    "Lube",
    "Milano",
    "Modena",
    "Monza",
    "Padova",
    "Cisterna",
    "Taranto",
    "Grottazzolina",
]
# Fuerza nominal asociada (fallback si el Elo no tiene datos)
STRENGTH_DEFAULTS_12 = {
    "Trento": 0.68,
    "Perugia": 0.65,
    "Verona": 0.60,
    "Piacenza": 0.58,
    "Lube": 0.56,
    "Milano": 0.53,
    "Modena": 0.52,
    "Monza": 0.48,
    "Padova": 0.47,
    "Cisterna": 0.45,
    "Taranto": 0.40,
    "Grottazzolina": 0.35,
}

# Config labels
CONFIGS = ["OFF", "ON", "NEW"]


def _load_set_predictor_v2() -> tuple:
    """Carga el SetPredictor v2; devuelve (predictor, source)."""
    try:
        pred, source = LogRegSetPredictor.try_load_v2(
            MODELS_DIR / "set_predictor_v2.joblib",
            MODELS_DIR / "set_predictor.joblib",
        )
        return pred, source
    except Exception as e:
        print(f"  [WARN] No se pudo cargar SetPredictor: {e}")
        return None, "none"


def _load_match_predictor():
    """Carga el MatchPredictor (necesario para season-level ON)."""
    try:
        return MatchPredictor.load(MODELS_DIR / "match_predictor.joblib")
    except Exception as e:
        print(f"  [WARN] No se pudo cargar MatchPredictor: {e}")
        return None


def _compute_pair_elo(home: str, away: str, elo_dict: dict) -> float:
    """Probabilidad de que gane el local segun Elo con margen."""
    elo_h = elo_dict.get(home, ELO_BASE)
    elo_a = elo_dict.get(away, ELO_BASE)
    return _elo_expected(elo_h + ELO_HOME_ADV, elo_a)


def _get_points(std_entry):
    """Extrae puntos del entry de standings (duck-typed)."""
    for attr in ("puntos", "points", "pts"):
        if hasattr(std_entry, attr):
            return getattr(std_entry, attr)
    if isinstance(std_entry, dict):
        for key in ("puntos", "points", "pts"):
            if key in std_entry:
                return std_entry[key]
    return 0


# ─── Nivel partido ──────────────────────────────────────────


def _run_pair_level(
    teams: list[str],
    elo_dict: dict,
    strengths: dict,
    n_sims: int,
    set_predictor: object,
    feature_builder: RuntimeFeatureBuilder,
    label: str,
) -> dict:
    """Mide |P_MC - p_elo| para los 132 pares ordenados de una config."""
    pairs = [(h, a) for h in teams for a in teams if h != a]
    abs_diffs = []

    simulator = MatchSimulator(point_model=None, player_stats_gen=None)

    for idx, (home, away) in enumerate(pairs):
        p_elo = _compute_pair_elo(home, away, elo_dict)

        # Para la config ON, construir team_features desde el feature builder
        team_feats = None
        if label in ("ON", "NEW") and set_predictor is not None and feature_builder is not None:
            try:
                feat_df = feature_builder.build_features(home, away, jornada=11)
                team_feats = SeasonSimulator._extract_set_team_features(feat_df)
            except Exception:
                pass  # si falla, corre sin team_features

        mc = simulator.monte_carlo_simulate(
            home_team=home,
            away_team=away,
            home_strength=strengths.get(home, 0.5),
            away_strength=strengths.get(away, 0.5),
            n_simulations=n_sims,
            seed=42 + idx,
            set_predictor=set_predictor,
            team_features=team_feats,
        )
        p_mc = mc["home_win_prob"]
        abs_diffs.append(abs(p_mc - p_elo))

    mean_ad = float(np.mean(abs_diffs))
    p95_ad = float(np.percentile(abs_diffs, 95))
    return {
        "mean_abs_diff": round(mean_ad, 5),
        "p95_abs_diff": round(p95_ad, 5),
        "n_pairs": len(pairs),
    }


# ─── Nivel temporada ────────────────────────────────────────


def _run_season_level(
    teams: list[str],
    strengths: dict,
    elo_dict: dict,
    n_seeds: int,
    season_sim: SeasonSimulator,
    use_set_calibration: bool,
) -> dict:
    """Mide Spearman, estabilidad de posicion y variacion de puntos para una config.

    Nota: mean_std_position es una metrica secundaria (esperada 0.0 para ON
    debido a la saturacion del clamp adaptativo per-point, que hace el ganador
    del set ~99% determinista). mean_std_total_points es la metrica primaria
    de estabilidad — captura la variacion en resultados 3-0/3-1/3-2, que
    siguen variando bajo el clamp.
    """
    all_positions = {t: [] for t in teams}
    all_points = {t: [] for t in teams}

    for s in range(n_seeds):
        result = season_sim.simulate_season(
            teams=teams,
            half="first",
            seed=s,
            use_set_calibration=use_set_calibration,
            use_match_predictor=True,
        )
        standings = result["standings"]
        for pos, std_entry in enumerate(standings, 1):
            team = std_entry.team if hasattr(std_entry, "team") else std_entry["equipo"]
            all_positions[team].append(pos)
            all_points[team].append(_get_points(std_entry))

    # Posicion media por equipo
    mean_pos = {t: float(np.mean(all_positions[t])) for t in teams}

    # Desviacion tipica por equipo (a traves de seeds) -> media
    std_pos = [float(np.std(all_positions[t])) for t in teams]
    mean_std = float(np.mean(std_pos))

    # Puntos totales por equipo (a traves de seeds) -> std -> media
    std_pts = [float(np.std(all_points[t])) for t in teams]
    mean_std_pts = float(np.mean(std_pts))

    # Spearman: fuerza margin-Elo descendente vs posicion media ascendente
    # (fuerza alta -> posicion baja, por eso spearman negativo esperado)
    forces = [elo_to_strength(elo_dict.get(t, ELO_BASE)) for t in teams]
    order_pos = [mean_pos[t] for t in teams]
    rho, pval = spearmanr(forces, order_pos)

    return {
        "spearman": round(float(rho), 5),
        "spearman_pvalue": round(float(pval), 6),
        "mean_std_position": round(mean_std, 5),
        "mean_std_total_points": round(mean_std_pts, 5),
        "n_seeds": n_seeds,
    }


# ─── Time-box ──────────────────────────────────────────────


def _project_time(
    n_sims: int,
    n_seeds: int,
) -> tuple:
    """Mide el coste de 1 partido y 1 temporada, proyecta el total.

    Returns:
        (pair_sec, season_sec, projected_total, accepted)
    """
    teams_short = TEAMS_12[:4]  # 4 equipos para la medicion rapida
    sim = MatchSimulator(point_model=None, player_stats_gen=None)

    # Coste de 1 partido MC
    t0 = time.perf_counter()
    sim.monte_carlo_simulate(
        home_team=teams_short[0],
        away_team=teams_short[1],
        home_strength=0.6,
        away_strength=0.5,
        n_simulations=n_sims,
        seed=0,
    )
    t_pair = time.perf_counter() - t0

    # Coste de 1 temporada
    dummy_strengths = {t: 0.5 for t in teams_short}
    ss_dummy = SeasonSimulator(
        simulator=MatchSimulator(point_model=None, player_stats_gen=None),
        team_strengths=dummy_strengths,
    )
    t0 = time.perf_counter()
    ss_dummy.simulate_season(
        teams=teams_short,
        half="first",
        seed=0,
        use_match_predictor=False,
    )
    t_season = time.perf_counter() - t0

    # Proyectar
    n_pairs = 132 * 2  # OFF + ON (NEW es placeholder)
    n_sim_seasons = n_seeds * 2  # OFF + ON
    # ON lleva sobrecarga de feature building (~40%)
    pair_projected = n_pairs * t_pair * 1.4
    season_projected = n_sim_seasons * t_season * 1.4
    projected = pair_projected + season_projected

    return t_pair, t_season, projected


def _print_projection(t_pair, t_season, projected, budget, n_sims_val, n_seeds_val):
    """Imprime la proyeccion de tiempo."""
    print("\n  === TIME-BOX ===")
    print(f"  Coste 1 partido MC (n_sims={n_sims_val}): {t_pair:.3f}s")
    print(f"  Coste 1 temporada ida:                {t_season:.3f}s")
    print(f"  Proyeccion total (OFF+ON):             {projected:.0f}s " f"({projected/60:.1f} min)")
    print(f"  Presupuesto:                           {budget:.0f}s ({budget/60:.1f} min)")


# ─── Tabla de resultados ───────────────────────────────────


def _print_summary_table(results: dict, params: dict):
    """Imprime la tabla resumen en espanol."""
    print()
    print("  " + "=" * 80)
    print("  RESULTADOS BACKTEST DEL CLAMP")
    print("  " + "=" * 80)
    header = (
        f"  {'Config':<8} {'|P_MC-p_elo|':<16} {'p95':<10} "
        f"{'Spearman':<11} {'Std pos':<10} {'Std pts':<10} "
        f"{'n_pairs':<9} {'n_seeds':<9} {'T(s)':<8}"
    )
    print(header)
    print("  " + "-" * 90)
    for cfg in CONFIGS:
        r = results["config"].get(cfg)
        if r is None:
            print(f"  {cfg:<8} {'(pendiente A3)':<70}")
            continue
        pl = r["pair_level"]
        sl = r["season_level"]
        t = r["time_seconds"]
        print(
            f"  {cfg:<8} {pl['mean_abs_diff']:<16.5f} {pl['p95_abs_diff']:<10.5f} "
            f"{sl['spearman']:<11.4f} {sl['mean_std_position']:<10.4f} "
            f"{sl['mean_std_total_points']:<10.4f} "
            f"{pl['n_pairs']:<9} {sl['n_seeds']:<9} {t:<8.1f}"
        )
    print("  " + "=" * 90)
    print(
        f"  Parametros: n_sims={params['n_sims']}, "
        f"n_seeds={params['n_seeds']}, "
        f"budget={params['time_budget_s']}s"
    )
    print(f"  Decision: {results['decision']}")
    print()


# ─── Main ──────────────────────────────────────────────────


def main(
    n_sims: int = 200,
    n_seeds: int = 10,
    time_budget_s: float = 900.0,
):
    """Ejecuta el backtest completo del clamp."""
    print("=" * 70)
    print("  A5 — BACKTEST REPRODUCIBLE DEL CLAMP [0.20, 0.80]")
    print("=" * 70)
    print(f"  n_sims={n_sims}  n_seeds={n_seeds}  " f"time_budget={time_budget_s:.0f}s")
    print(f"  Clamp estatico por defecto: {DEFAULT_CLAMP_RANGE}")
    print()

    # ── Cargar modelos ──
    print("  Cargando modelos...")
    set_predictor_v2, sp_source = _load_set_predictor_v2()
    print(f"  SetPredictor v2: {sp_source}")
    match_predictor = _load_match_predictor()
    mp_status = "OK" if match_predictor is not None else "NO DISPONIBLE"
    print(f"  MatchPredictor:  {mp_status}")

    # ── Elo historico ──
    print("  Cargando Elo historico...")
    elo_dict = get_historical_team_elo()
    print(f"  Elo disponible: {len(elo_dict)} equipos")

    # Fuerzas: usar las del Elo historico, con fallback a _STRENGTH_DEFAULTS
    strengths = {}
    for t in TEAMS_12:
        strengths[t] = elo_to_strength(elo_dict.get(t, ELO_BASE))

    # ── Feature builder: solo se comprueba que se puede construir ──
    # Tras el aislamiento por config (ver `_fresh_builder` mas abajo), cada
    # config estrena el suyo, asi que aqui no se guarda ninguna instancia:
    # esto es un smoke check de arranque para fallar pronto y con mensaje
    # claro si el Elo historico no carga.
    try:
        RuntimeFeatureBuilder(initial_elo=elo_dict)
        print("  FeatureBuilder:  OK")
    except Exception as e:
        print(f"  FeatureBuilder:  NO DISPONIBLE ({e})")

    # ── Time-box ──
    t_pair, t_season, projected = _project_time(n_sims, n_seeds)
    decision = "FULL"
    actual_n_sims = n_sims
    actual_n_seeds = n_seeds

    if projected > time_budget_s:
        print(
            f"\n  *** PROYECCION EXCEDE PRESUPUESTO ({projected:.0f}s > "
            f"{time_budget_s:.0f}s) ***"
        )
        # Primer intento: reducir parametros
        ns_new = max(50, n_sims // 4)
        nd_new = max(3, n_seeds // 3)
        print(f"  Reduciendo: n_sims {n_sims}->{ns_new}, " f"n_seeds {n_seeds}->{nd_new}")

        # Reproyectar con parametros reducidos
        t_pair2, t_season2, projected2 = _project_time(ns_new, nd_new)
        if projected2 > time_budget_s:
            print(f"  *** AUN EXCEDE ({projected2:.0f}s > {time_budget_s:.0f}s) ***")
            print(
                "  ABORTADO: la proyeccion supera el presupuesto incluso con " "parametros minimos."
            )
            print("  Usa --time-budget-s mayor o reduce manualmente n_sims/n_seeds.")
            abort_result = {
                "aborted": True,
                "projected_seconds": round(projected2, 1),
                "time_budget_s": time_budget_s,
                "reason": "Time-box exceeded even at reduced params",
            }
            return abort_result

        actual_n_sims = ns_new
        actual_n_seeds = nd_new
        t_pair, t_season = t_pair2, t_season2
        decision = "REDUCED_PARAMS"

    print()
    _print_projection(t_pair, t_season, projected, time_budget_s, n_sims, n_seeds)
    print(f"  Decision: {decision}")
    print(f"  Parametros efectivos: n_sims={actual_n_sims}, n_seeds={actual_n_seeds}")
    print()

    # ── Fabricas de estado LIMPIO por config ──
    # Gotcha (documentado en el plan, E1): RuntimeFeatureBuilder acumula
    # estado Elo en cada temporada simulada y `_init_dynamic_state` solo corre
    # en el constructor. Compartir un builder entre OFF/ON/NEW hace que las
    # temporadas de una config contaminen el Elo que ve la siguiente, y las
    # metricas dejan de ser comparables. Sintoma que lo delato: el nivel-par,
    # que usa seeds FIJAS y deberia ser determinista, cambiaba entre corridas
    # al variar n_seeds. Por eso cada config estrena builder sembrado desde el
    # mismo `elo_dict` historico.
    def _fresh_builder():
        return RuntimeFeatureBuilder(initial_elo=elo_dict)

    def _fresh_season_sim(with_predictor: bool):
        return SeasonSimulator(
            simulator=MatchSimulator(point_model=None, player_stats_gen=None),
            team_strengths=strengths,
            set_predictor=set_predictor_v2 if with_predictor else None,
            feature_builder=_fresh_builder(),
            match_predictor=match_predictor,
        )

    # ── Resultados ──
    results = {
        "config": {},
        "params": {
            "n_sims": actual_n_sims,
            "n_seeds": actual_n_seeds,
            "time_budget_s": time_budget_s,
            "clamp_range_default": list(DEFAULT_CLAMP_RANGE),
        },
        "decision": decision,
    }

    # ── OFF ──
    print("  --- CONFIG: OFF (clamp estatico) ---")
    t0 = time.perf_counter()
    pair_off = _run_pair_level(
        TEAMS_12,
        elo_dict,
        strengths,
        n_sims,
        set_predictor=None,
        feature_builder=None,
        label="OFF",
    )
    season_off = _run_season_level(
        TEAMS_12,
        strengths,
        elo_dict,
        n_seeds,
        _fresh_season_sim(with_predictor=False),
        use_set_calibration=False,
    )
    t_off = time.perf_counter() - t0
    print(
        f"  OFF: |P_MC-p_elo| media={pair_off['mean_abs_diff']:.5f} "
        f"p95={pair_off['p95_abs_diff']:.5f} | "
        f"spearman={season_off['spearman']:.4f} "
        f"std_pos={season_off['mean_std_position']:.4f} "
        f"std_pts={season_off['mean_std_total_points']:.4f} | "
        f"tiempo={t_off:.1f}s"
    )
    results["config"]["OFF"] = {
        "pair_level": pair_off,
        "season_level": season_off,
        "time_seconds": round(t_off, 1),
    }

    # ── ON ──
    print("  --- CONFIG: ON (clamp adaptativo v2) ---")
    t0 = time.perf_counter()
    pair_on = _run_pair_level(
        TEAMS_12,
        elo_dict,
        strengths,
        n_sims,
        set_predictor=set_predictor_v2,
        feature_builder=_fresh_builder(),
        label="ON",
    )
    season_on = _run_season_level(
        TEAMS_12,
        strengths,
        elo_dict,
        n_seeds,
        _fresh_season_sim(with_predictor=True),
        use_set_calibration=True,
    )
    t_on = time.perf_counter() - t0
    print(
        f"  ON:  |P_MC-p_elo| media={pair_on['mean_abs_diff']:.5f} "
        f"p95={pair_on['p95_abs_diff']:.5f} | "
        f"spearman={season_on['spearman']:.4f} "
        f"std_pos={season_on['mean_std_position']:.4f} "
        f"std_pts={season_on['mean_std_total_points']:.4f} | "
        f"tiempo={t_on:.1f}s"
    )
    results["config"]["ON"] = {
        "pair_level": pair_on,
        "season_level": season_on,
        "time_seconds": round(t_on, 1),
    }

    # ── NEW (contract-based, same as ON after A3) ──
    # After A3 (task T-005/T-006), _extract_set_team_features delegates to the
    # contract and _eval_set_predictor no longer overrides with live score.
    # ON is now effectively the same as NEW.  We record a separate run here
    # so the comparison table is complete.
    print("  --- CONFIG: NEW (contract path, A3) ---")
    t0 = time.perf_counter()
    pair_new = _run_pair_level(
        TEAMS_12,
        elo_dict,
        strengths,
        n_sims,
        set_predictor=set_predictor_v2,
        feature_builder=_fresh_builder(),
        label="NEW",
    )
    season_new = _run_season_level(
        TEAMS_12,
        strengths,
        elo_dict,
        n_seeds,
        _fresh_season_sim(with_predictor=True),
        use_set_calibration=True,
    )
    t_new = time.perf_counter() - t0
    print(
        f"  NEW: |P_MC-p_elo| media={pair_new['mean_abs_diff']:.5f} "
        f"p95={pair_new['p95_abs_diff']:.5f} | "
        f"spearman={season_new['spearman']:.4f} "
        f"std_pos={season_new['mean_std_position']:.4f} "
        f"std_pts={season_new['mean_std_total_points']:.4f} | "
        f"tiempo={t_new:.1f}s"
    )
    results["config"]["NEW"] = {
        "pair_level": pair_new,
        "season_level": season_new,
        "time_seconds": round(t_new, 1),
        "note": "A3 contract path — _extract_set_team_features via build_set_features, no live-score override",
    }

    # ── Imprimir tabla ──
    _print_summary_table(results, results["params"])

    # ── Guardar ──
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A5 — Backtest reproducible del clamp del simulador",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=200,
        help="Simulaciones Monte Carlo por partido (default: 200)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10, help="Semillas para nivel temporada (default: 10)"
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=900.0,
        help="Presupuesto de tiempo en segundos (default: 900)",
    )
    args = parser.parse_args()

    result = main(
        n_sims=args.n_sims,
        n_seeds=args.n_seeds,
        time_budget_s=args.time_budget_s,
    )

    if result.get("aborted"):
        sys.exit(1)
