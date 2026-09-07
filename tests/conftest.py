"""Shared test fixtures for the predictor2 test suite.

All synthetic model fixtures produce real model instances fit on tiny
synthetic dataframes so they work on a fresh clone without models/*.joblib.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────
# Autouse: seed everything before every test
# ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed random and numpy.random for deterministic tests."""
    random.seed(42)
    np.random.seed(42)
    yield


# ─────────────────────────────────────────────────────────────
# Synthetic model helpers (tiny data, real classes)
# ─────────────────────────────────────────────────────────────

_N_SYNTHETIC = 50


def _make_synthetic_binary_df(
    feature_cols: List[str], n: int = _N_SYNTHETIC
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a tiny synthetic DataFrame + binary target for quick model fitting."""
    X = pd.DataFrame({col: np.random.uniform(-1, 1, n) for col in feature_cols})
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


# ─────────────────────────────────────────────────────────────
# Synthetic fixture: SetPredictor
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def synthetic_set_predictor():
    """A real SetPredictor instance fit on ~50-row synthetic data.

    The calibrated model provides bounded [0,1] probabilities suitable for
    smoke-testing the simulation / API code paths that consume ``predict_proba``.
    """
    from src.models.set_predictor import SetPredictor

    feat_names = [
        "set_num_norm",
        "sets_h_antes",
        "sets_a_antes",
        "diff_sets_antes",
        "es_desempate",
        "momentum_h",
        "pts_fav_h",
        "pts_fav_a",
    ]
    X, y = _make_synthetic_binary_df(feat_names)

    predictor = SetPredictor()
    # Quick-fit: use a tiny LogisticRegression + calibration so predict_proba works
    lr = LogisticRegression(max_iter=500, random_state=42)
    cal = CalibratedClassifierCV(lr, cv=2, method="isotonic")
    cal.fit(X.values, y.values)

    predictor.best_model_name = "LogisticRegression"
    predictor.best_model = lr
    predictor.calibrated_model = cal
    predictor.feature_names = feat_names
    predictor.scaler.fit(X)  # fit the scaler so transform works
    predictor.results = {
        "LogisticRegression": {"accuracy": 0.5, "auc_roc": 0.5, "brier_score": 0.25}
    }
    return predictor


# ─────────────────────────────────────────────────────────────
# Synthetic fixture: MatchPredictor
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def synthetic_match_predictor():
    """Real MatchPredictor instance fit on synthetic data."""
    from src.models.match_predictor import MatchPredictor
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.calibration import CalibratedClassifierCV

    feat_names = [
        "h_win_rate_global",
        "a_win_rate_global",
        "diff_win_rate_global",
        "elo_diff",
        "h_descanso",
        "a_descanso",
    ]
    X, y = _make_synthetic_binary_df(feat_names)

    et = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
    cal = CalibratedClassifierCV(et, cv=2, method="isotonic")
    cal.fit(X.values, y.values)

    predictor = MatchPredictor()
    predictor.best_model_name = "ExtraTrees"
    predictor.best_model = et
    predictor.calibrated_model = cal
    predictor.feature_names = feat_names
    predictor.results = {"ExtraTrees": {"accuracy": 0.5, "auc_roc": 0.5, "brier_score": 0.25}}
    return predictor


# ─────────────────────────────────────────────────────────────
# Synthetic fixture: PointProbabilityModel
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def synthetic_point_model():
    """Real PointProbabilityModel fit on tiny synthetic match_features."""
    from src.models.point_probability import PointProbabilityModel

    n = 30
    df = pd.DataFrame(
        {
            "point_ratio_h": np.random.uniform(0.45, 0.55, n),
            "point_ratio_a": np.random.uniform(0.35, 0.45, n),
            "elo_diff": np.random.uniform(-50, 50, n),
            "diff_win_rate_global": np.random.uniform(-0.2, 0.2, n),
            "diff_set_win_rate": np.random.uniform(-0.2, 0.2, n),
            "diff_dominancia": np.random.uniform(-0.2, 0.2, n),
            "diff_set_ratio": np.random.uniform(-0.2, 0.2, n),
            "diff_forma_efectiva": np.random.uniform(-0.2, 0.2, n),
        }
    )

    model = PointProbabilityModel()
    model.fit(df)
    return model


# ─────────────────────────────────────────────────────────────
# Synthetic fixture: PlayerStatsGenerator
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def synthetic_player_gen():
    """Real PlayerStatsGenerator fit on synthetic player/team stats."""
    from src.models.player_stats_generator import PlayerStatsGenerator

    players = ["PlayerA", "PlayerB", "PlayerC"]
    n = len(players)
    player_df = pd.DataFrame(
        {
            "equipo_id": ["TEAM01"] * n,
            "jugador": players,
            "sets": [50, 45, 40],
            "temporada": ["2024/2025"] * n,
            "puntos": [200, 180, 150],
            "aces": [15, 10, 8],
            "ataques_ganados": [150, 120, 100],
            "bloqueos": [35, 50, 42],
            "recepciones_exc": [80, 70, 60],
            "errores_saque": [10, 8, 5],
        }
    )
    team_df = pd.DataFrame({"equipo": ["Trento"]})

    gen = PlayerStatsGenerator()
    gen.fit(player_df, team_df)
    return gen


# ─────────────────────────────────────────────────────────────
# Synthetic fixture: RuntimeFeatureBuilder (tmp_path CSV)
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def synthetic_feature_builder(tmp_path: Path):
    """RuntimeFeatureBuilder backed by a tiny tmp_path CSV (no real DB needed).

    The CSV must include all ``h_*`` / ``a_*`` columns from
    ``MATCH_FEATURE_COLS`` because ``RuntimeFeatureBuilder._load_static_profiles``
    iterates over them unconditionally.  Non-``h_/a_`` columns are skipped by
    the static-profiles loop so they do NOT need to be present.
    """
    from src.simulation.feature_builder import RuntimeFeatureBuilder
    from src.data.feature_store import MATCH_FEATURE_COLS

    all_h_a_cols = [c for c in MATCH_FEATURE_COLS if c.startswith("h_") or c.startswith("a_")]

    csv_path = tmp_path / "match_features.csv"
    rows = []
    for local in ("Trento", "Perugia"):
        for visitante in ("Trento", "Perugia"):
            if local == visitante:
                continue
            row = {
                "local": local,
                "visitante": visitante,
                "gana_local": 1,
                "temporada": "2024/2025",
            }
            # Fill every required h_/a_ column with a dummy value
            for col in all_h_a_cols:
                row[col] = 0.5
            rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    builder = RuntimeFeatureBuilder(csv_path=csv_path)
    return builder


# ─────────────────────────────────────────────────────────────
# API override fixture: swap src.api.main singletons in + teardown
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def app_with_synthetic(
    monkeypatch,
    synthetic_set_predictor,
    synthetic_match_predictor,
    synthetic_point_model,
    synthetic_player_gen,
    synthetic_feature_builder,
):
    """Swap all five src.api.main singletons with synthetic equivalents.

    Uses monkeypatch.setattr so the swap is reverted on teardown.
    Tests that call the app via TestClient must use this fixture (or
    monkeypatch individual singletons themselves).
    """
    import src.api.main as api

    monkeypatch.setattr(api, "set_predictor", synthetic_set_predictor)
    monkeypatch.setattr(api, "match_predictor", synthetic_match_predictor)
    monkeypatch.setattr(api, "point_model", synthetic_point_model)
    monkeypatch.setattr(api, "player_gen", synthetic_player_gen)
    monkeypatch.setattr(api, "feature_builder", synthetic_feature_builder)

    # Rebuild simulator with synthetic dependencies
    from src.simulation.simulator import MatchSimulator

    monkeypatch.setattr(
        api,
        "simulator",
        MatchSimulator(
            point_model=synthetic_point_model,
            player_stats_gen=synthetic_player_gen,
        ),
    )

    # Recompute TEAM_STRENGTHS with synthetic data so sample teams resolve
    monkeypatch.setattr(
        api,
        "TEAM_STRENGTHS",
        {
            "Trento": 0.55,
            "Perugia": 0.52,
        },
    )

    yield

    # monkeypatch.undo() is automatic on fixture teardown


# ─────────────────────────────────────────────────────────────
# Sample-team fixture
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def sample_teams():
    """Two canonical team keys present in _STRENGTH_DEFAULTS."""
    return ["Trento", "Perugia"]


# ─────────────────────────────────────────────────────────────
# tmp_path CSV helper
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def csv_helper(tmp_path: Path):
    """Write a minimal CSV to tmp_path and return its path."""

    def _write(df: pd.DataFrame, filename: str = "test.csv") -> Path:
        path = tmp_path / filename
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    return _write


# ─────────────────────────────────────────────────────────────
# Normalize vectors fixture data
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def normalize_vectors() -> List[Tuple[str, str]]:
    """Known (raw → canonical) normalization pairs from the real team_mapper."""
    return [
        ("MonzaMonza", "Monza"),
        ("Diatec Trentino", "Trento"),
        ("Sir Safety Conad Perugia", "Perugia"),
        ("Azimut Modena", "Modena"),
        ("  Kioene Padova  ", "Padova"),
        ("Gi Group Monza", "Monza"),
        ("VeronaVerona", "Verona"),
        ("GrottazzolinaGrottazzolina", "Grottazzolina"),
        ("Vibo ValentiaVibo Valentia", "Vibo Valentia"),
        ("Emma Villas Siena", "Siena"),
        ("Videx Grottazzolina", "Grottazzolina"),
        ("Lube CivitanovaLube Civitanova", "Lube"),
    ]
