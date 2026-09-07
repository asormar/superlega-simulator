"""
train.py — Orquestador de entrenamiento de todos los modelos.

Ejecuta el pipeline de datos, entrena los modelos de set, punto y
estadisticas de jugadores, y guarda los resultados.
"""

import sys
from pathlib import Path

# Asegurar que el directorio raiz esta en el path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_set_data,
    prepare_match_data,
    MATCH_FEATURE_COLS,
    ENRICHED_MATCH_COLS,
    ROSTER_BASIC_COLS,
    enrich_with_team_stats,
    compute_roster_features,
)
from src.models.set_predictor import SetPredictor
from src.models.point_probability import PointProbabilityModel, build_point_training_data
from src.models.player_stats_generator import PlayerStatsGenerator
from src.models.match_predictor import MatchPredictor


def train_all():
    """Entrena todos los modelos del sistema."""

    print("=" * 70)
    print("  ENTRENAMIENTO DE MODELOS - SuperLega Volleyball Simulator")
    print("=" * 70)

    # ─── 1. Cargar datos ───
    print("\n[PASO 1] Cargando y limpiando datos...")
    data = run_pipeline()

    # ─── 2. Preparar splits ───
    print("\n[PASO 2] Preparando splits temporales...")

    print("\n  >> Set features:")
    X_set, y_set = prepare_set_data(data["set_features"])

    print("\n  >> Match features:")
    X_match, y_match = prepare_match_data(data["match_features"])

    # ─── 3. Entrenar Set Predictor ───
    print("\n" + "=" * 70)
    print("  [PASO 3] ENTRENANDO SET PREDICTOR")
    print("=" * 70)

    set_predictor = SetPredictor()
    set_predictor.train(
        X_train=X_set["train"],
        y_train=y_set["train"],
        X_val=X_set["val"],
        y_val=y_set["val"],
    )

    # Evaluar en test
    print("\n  Evaluacion en TEST set:")
    test_results = set_predictor.evaluate(X_set["test"], y_set["test"])

    # Feature importance
    fi = set_predictor.get_feature_importance()
    print("\n  Top 10 features mas importantes:")
    print(fi.head(10).to_string(index=False))

    # Guardar
    set_predictor.save()

    # ─── 4. Entrenar Point Probability Model ───
    print("\n" + "=" * 70)
    print("  [PASO 4] ENTRENANDO POINT PROBABILITY MODEL")
    print("=" * 70)

    # B3: se entrena sobre features ROLLING pre-partido (sin leakage) con el
    # ratio de puntos real como target continuo, NO sobre `match_features`
    # (que incluye stats de temporada completa == leakage, ver B0b del plan).
    point_model = PointProbabilityModel()
    point_train = build_point_training_data()
    print(f"  Dataset rolling para el punto: {len(point_train)} partidos")
    point_model.fit(point_train)

    # Test rapido
    probs = point_model.get_point_probabilities(
        home_strength=0.55,
        away_strength=0.45,
    )
    print("\n  Test (home=0.55, away=0.45):")
    for k, v in probs.items():
        print(f"    {k}: {v:.4f}")

    point_model.save()

    # ─── 5. Entrenar Player Stats Generator ───
    print("\n" + "=" * 70)
    print("  [PASO 5] AJUSTANDO PLAYER STATS GENERATOR")
    print("=" * 70)

    player_gen = PlayerStatsGenerator()
    player_gen.fit(data["player_stats"], data["team_stats"])

    # Test rapido: generar stats para un set de Trento
    test_team = "Trento"
    if test_team in player_gen.team_profiles:
        test_stats = player_gen.generate_set_stats(test_team, 25, 20)
        print(f"\n  Test: stats simuladas para {test_team} (25-20):")
        for ps in test_stats[:5]:
            print(
                f"    {ps['jugador'][:20]:<20} pts={ps.get('puntos',0):>2} "
                f"ace={ps.get('aces',0):.0f} atq={ps.get('ataques_ganados',0):.0f} "
                f"blq={ps.get('bloqueos',0):.0f}"
            )

    player_gen.save()

    # ─── 6. Entrenar Match Predictor ───
    print("\n" + "=" * 70)
    print("  [PASO 6] ENTRENANDO MATCH PREDICTOR")
    print("=" * 70)

    # Enriquecer match_features con team stats y roster
    match_df = data["match_features"].copy()
    match_df = enrich_with_team_stats(match_df, data["team_stats"])
    match_df = compute_roster_features(match_df, data["player_stats"])

    # Features: base + enriquecidas + roster basico (87 total)
    match_cols = [
        c
        for c in MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS + ROSTER_BASIC_COLS
        if c in match_df.columns
    ]
    print(f"  [match] {len(match_cols)} features totales disponibles")

    X_match, y_match = prepare_match_data(match_df, feature_cols=match_cols)

    match_predictor = MatchPredictor()
    match_predictor.train(
        X_train=X_match["train"],
        y_train=y_match["train"],
        X_val=X_match["val"],
        y_val=y_match["val"],
    )

    match_test = match_predictor.evaluate(X_match["test"], y_match["test"])

    fi = match_predictor.get_feature_importance()
    print("\n  Top 10 features mas importantes:")
    print(fi.head(10).to_string(index=False))

    match_predictor.save()

    # ─── Resumen final ───
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(
        f"  Set Predictor:    {set_predictor.best_model_name} "
        f"(Test ACC={test_results['accuracy']:.3f}, "
        f"AUC={test_results['auc_roc']:.3f})"
    )
    print(
        f"  Match Predictor:  {match_predictor.best_model_name} "
        f"(Test ACC={match_test['accuracy']:.3f}, "
        f"AUC={match_test['auc_roc']:.3f})"
    )
    print("  Point Probability: Modelo calibrado")
    print(f"  Player Stats:      {len(player_gen.team_profiles)} equipos")
    print(f"\n  Modelos guardados en: {BASE_DIR / 'models'}")


if __name__ == "__main__":
    train_all()
