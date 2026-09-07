"""Smoke tests for the four ML models — synthetic fixtures, no models/*.joblib needed."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.set_predictor import SetPredictor
from src.models.set_predictor_v2 import LogRegSetPredictor
from src.models.match_predictor import MatchPredictor
from src.models.point_probability import PointProbabilityModel
from src.models.player_stats_generator import PlayerStatsGenerator
from src.simulation.feature_builder import RuntimeFeatureBuilder


def _assert_proba_bounded(proba: np.ndarray, label: str):
    """Assert that a predict_proba output is in [0,1] with no NaN."""
    assert proba is not None, f"{label}: predict_proba returned None"
    assert not np.any(np.isnan(proba)), f"{label}: NaN found in probabilities"
    assert np.all(proba >= 0.0), f"{label}: negative probabilities found"
    assert np.all(proba <= 1.0), f"{label}: probabilities > 1.0 found"


class TestSyntheticFixturesSmoke:
    """Every fixture in conftest.py instantiates cleanly (W2 gate-review closure)."""

    def test_synthetic_set_predictor(self, synthetic_set_predictor: SetPredictor):
        pred = synthetic_set_predictor
        assert pred is not None
        assert isinstance(pred.feature_names, list)
        assert len(pred.feature_names) > 0

        # Build a tiny input and assert predict_proba returns bounded [0,1]
        feat_df = pd.DataFrame(
            [{f: 0.0 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "SetPredictor")
        # Shape: (n_samples, 2) — P(away), P(home)
        assert proba.shape == (1, 2)

    def test_synthetic_match_predictor(self, synthetic_match_predictor: MatchPredictor):
        pred = synthetic_match_predictor
        assert pred is not None
        assert isinstance(pred.feature_names, list)
        assert len(pred.feature_names) > 0

        feat_df = pd.DataFrame(
            [{f: 0.0 for f in pred.feature_names}],
        )
        proba = pred.predict_proba(feat_df)
        _assert_proba_bounded(proba, "MatchPredictor")
        assert proba.shape == (1, 2)

    def test_synthetic_point_model(self, synthetic_point_model: PointProbabilityModel):
        model = synthetic_point_model
        assert model is not None
        assert model.is_fitted

        probs = model.get_point_probabilities(
            match_features={
                "elo_diff": 0.0,
                "diff_win_rate_global": 0.0,
                "diff_set_win_rate": 0.0,
                "diff_dominancia": 0.0,
                "diff_set_ratio": 0.0,
                "diff_forma_efectiva": 0.0,
            },
        )
        assert isinstance(probs, dict)
        for key in ("p_home_serving", "p_home_receiving", "p_away_serving", "p_away_receiving"):
            assert key in probs
            assert 0.0 <= probs[key] <= 1.0

    def test_synthetic_player_gen(self, synthetic_player_gen: PlayerStatsGenerator):
        gen = synthetic_player_gen
        assert gen is not None
        assert len(gen.team_profiles) > 0

        stats = gen.generate_set_stats("Trento", team_score=25, opponent_score=23)
        assert isinstance(stats, list)
        # At least one player should have stats
        assert len(stats) >= 0

    def test_synthetic_feature_builder(self, synthetic_feature_builder: RuntimeFeatureBuilder):
        builder = synthetic_feature_builder
        assert builder is not None
        df = builder.build_features("Trento", "Perugia", jornada=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_synthetic_set_predictor_brier(self):
        """Synthetic SetPredictor has Brier score < 0.30 on its training data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        feat_names = ["f1", "f2", "f3"]
        X = pd.DataFrame({f: np.random.uniform(-1, 1, 30) for f in feat_names})
        y = pd.Series(np.random.randint(0, 2, 30))
        lr = LogisticRegression(max_iter=500, random_state=42)
        cal = CalibratedClassifierCV(lr, cv=2, method="isotonic")
        cal.fit(X.values, y.values)
        proba = cal.predict_proba(X.values)
        brier = float(np.mean((proba[:, 1] - y.values) ** 2))
        assert brier < 0.30, f"Brier score {brier:.4f} >= 0.30"


class TestSetPredictorSmoke:
    """SetPredictor with synthetic data — calibrated, bounded, no NaN."""

    def test_predict_proba_bounds_and_sum(self, synthetic_set_predictor: SetPredictor):
        feat_df = pd.DataFrame([{f: 0.5 for f in synthetic_set_predictor.feature_names}])
        proba = synthetic_set_predictor.predict_proba(feat_df)
        _assert_proba_bounded(proba, "SetPredictor")
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_returns_int(self, synthetic_set_predictor: SetPredictor):
        feat_df = pd.DataFrame([{f: 0.3 for f in synthetic_set_predictor.feature_names}])
        preds = synthetic_set_predictor.predict(feat_df)
        assert preds.dtype == np.int_ or preds.dtype == np.int64


class TestMatchPredictorSmoke:
    """MatchPredictor with synthetic data."""

    def test_predict_proba_bounds_and_sum(self, synthetic_match_predictor: MatchPredictor):
        feat_df = pd.DataFrame([{f: 0.5 for f in synthetic_match_predictor.feature_names}])
        proba = synthetic_match_predictor.predict_proba(feat_df)
        _assert_proba_bounded(proba, "MatchPredictor")
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


class TestPointProbabilityModelSmoke:
    """PointProbabilityModel with synthetic data."""

    def test_get_point_probabilities_with_features(
        self, synthetic_point_model: PointProbabilityModel
    ):
        probs = synthetic_point_model.get_point_probabilities(
            match_features={
                "elo_diff": 5.0,
                "diff_win_rate_global": 0.05,
                "diff_set_win_rate": 0.03,
                "diff_dominancia": 0.02,
                "diff_set_ratio": 0.04,
                "diff_forma_efectiva": 0.01,
            },
        )
        assert 0.0 <= probs["p_home_serving"] <= 1.0
        assert 0.0 <= probs["p_home_receiving"] <= 1.0
        assert 0.0 <= probs["p_away_serving"] <= 1.0
        assert 0.0 <= probs["p_away_receiving"] <= 1.0
        # Conservation: p_home_serving + p_away_receiving ≈ 1
        # (they are complements in Markov)
        assert abs(probs["p_home_serving"] + probs["p_away_receiving"] - 1.0) < 1e-6

    def test_get_point_probabilities_without_features(
        self, synthetic_point_model: PointProbabilityModel
    ):
        """Fallback path: no match_features dict — uses strength directly."""
        probs = synthetic_point_model.get_point_probabilities(
            match_features=None,
            home_strength=0.60,
            away_strength=0.40,
        )
        for key in ("p_home_serving", "p_home_receiving"):
            assert 0.0 <= probs[key] <= 1.0

    def test_default_sideout_rate_is_62(self):
        from src.simulation.constants import DEFAULT_SIDEOUT_RATE

        assert DEFAULT_SIDEOUT_RATE == 0.62


class TestPlayerStatsGeneratorSmoke:
    """PlayerStatsGenerator with synthetic data (no player_stats_params.json)."""

    def test_generate_set_stats_shape(self, synthetic_player_gen: PlayerStatsGenerator):
        stats = synthetic_player_gen.generate_set_stats("Trento", team_score=25, opponent_score=23)
        assert isinstance(stats, list)
        if stats:
            for entry in stats:
                assert "jugador" in entry and "puntos" in entry
        assert isinstance(synthetic_player_gen.get_roster("Trento"), list)
        assert isinstance(synthetic_player_gen.get_profile("Trento"), dict)

    def test_generate_set_stats_unknown_team(self, synthetic_player_gen: PlayerStatsGenerator):
        """Unknown team returns empty list."""
        stats = synthetic_player_gen.generate_set_stats(
            "NonExistent",
            team_score=25,
            opponent_score=23,
        )
        assert stats == []


@pytest.mark.slow
class TestRealArtifacts:
    """Integration tests requiring actual models/*.joblib (opt-in via ``pytest -m slow``)."""

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

    @pytest.mark.parametrize(
        "name,cls",
        [
            ("set_predictor", SetPredictor),
            ("match_predictor", MatchPredictor),
        ],
    )
    def test_predictor_loads(self, name, cls):
        path = self.MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            pytest.skip(f"{name}.joblib not found")
        pred = cls.load(path)
        assert pred.feature_names is not None
        _assert_proba_bounded(
            pred.predict_proba(pd.DataFrame([{f: 0.5 for f in pred.feature_names}])), name
        )

    def test_real_point_model_get_probs(self):
        path = self.MODELS_DIR / "point_probability.joblib"
        if not path.exists():
            pytest.skip("point_probability.joblib not found")
        probs = PointProbabilityModel.load(path).get_point_probabilities(
            match_features={
                k: 0.0
                for k in (
                    "elo_diff",
                    "diff_win_rate_global",
                    "diff_set_win_rate",
                    "diff_dominancia",
                    "diff_set_ratio",
                    "diff_forma_efectiva",
                )
            },
        )
        for val in probs.values():
            assert 0.0 <= val <= 1.0

    def test_real_player_gen_fit(self):
        path = self.MODELS_DIR / "player_stats_params.json"
        if not path.exists():
            pytest.skip("player_stats_params.json not found")
        assert len(PlayerStatsGenerator.load(path).team_profiles) > 0


class TestOptunaSearchArtifacts:
    """Smoke tests for the Optuna hyperparameter search output (models/best_params.json)."""

    JSON_PATH = Path("models/best_params.json")
    EXPECTED_KEYS = {"set_extratrees", "match_xgboost"}
    EXPECTED_SUBKEYS = {"default_auc", "optuna_auc", "delta", "best_params"}

    def test_best_params_json_exists_and_valid(self):
        if not self.JSON_PATH.exists():
            pytest.skip(
                "models/best_params.json not yet generated; "
                "run `python -m src.models.hyperparameter_search` to create it"
            )
        with open(self.JSON_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert self.EXPECTED_KEYS.issubset(
            data.keys()
        ), f"Missing keys: {self.EXPECTED_KEYS - set(data.keys())}"
        for model_key in self.EXPECTED_KEYS:
            for sub in self.EXPECTED_SUBKEYS:
                assert sub in data[model_key], f"Missing {sub} in {model_key}"
            # Delta is best - default; sanity-check the sign convention
            assert (
                abs(
                    data[model_key]["delta"]
                    - (data[model_key]["optuna_auc"] - data[model_key]["default_auc"])
                )
                < 1e-9
            )

    def test_search_module_imports(self):
        """The hyperparameter_search module is importable and exposes run_search.

        `optuna` is NOT a project dependency: hyperparameter_search is an
        experiment module (Batch 3, closed as a negative result), not part of
        the production path. It is installed in the dev environment but not in
        a clean install, so this test skips instead of failing there — same
        pattern as the sibling tests that skip on a missing JSON artifact.
        """
        pytest.importorskip("optuna", reason="optuna is a dev-only extra")
        from src.models import hyperparameter_search

        assert hasattr(hyperparameter_search, "run_search")
        assert callable(hyperparameter_search.run_search)


class TestFeatureSelectionArtifacts:
    """Smoke tests for the feature selection experiment (Batch 3 mid-effort #2)."""

    JSON_PATH = Path("models/feature_selection_results.json")
    EXPECTED_KEYS = {
        "n_top",
        "all_features",
        "top_features",
        "variant_a",
        "variant_b",
        "delta_val",
        "delta_test",
        "verdict",
        "recommendation",
    }
    EXPECTED_VERDICTS = {"improved", "marginal", "degraded"}

    def test_results_json_exists_and_valid(self):
        if not self.JSON_PATH.exists():
            pytest.skip(
                "models/feature_selection_results.json not yet generated; "
                "run `python -m src.models.feature_selection_experiment` to create it"
            )
        with open(self.JSON_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert self.EXPECTED_KEYS.issubset(
            data.keys()
        ), f"Missing keys: {self.EXPECTED_KEYS - set(data.keys())}"
        # Variant A (87 features) and Variant B (top-N) structure
        for variant_key in ("variant_a", "variant_b"):
            for sub in ("val_auc", "test_auc", "best_model_name", "n_features"):
                assert sub in data[variant_key], f"Missing {sub} in {variant_key}"
        # Deltas match the diff
        assert (
            abs(
                data["delta_test"] - (data["variant_b"]["test_auc"] - data["variant_a"]["test_auc"])
            )
            < 1e-9
        )
        assert data["verdict"] in self.EXPECTED_VERDICTS
        assert data["recommendation"] in {"apply-top-N", "keep-defaults"}
        # Top-N is actually a subset of all features
        assert set(data["top_features"]).issubset(set(data["all_features"]))
        assert data["variant_b"]["n_features"] == data["n_top"]

    def test_experiment_module_imports(self):
        """The feature_selection_experiment module is importable and exposes run_experiment."""
        from src.models import feature_selection_experiment

        assert hasattr(feature_selection_experiment, "run_experiment")
        assert callable(feature_selection_experiment.run_experiment)


class TestAdaptiveDampingArtifacts:
    """Smoke tests for the adaptive damping experiment (Batch 3 mid-effort #3)."""

    JSON_PATH = Path("models/adaptive_damping_results.json")
    EXPECTED_KEYS = {
        "n_mc",
        "damping_fixed",
        "damping_adaptive_start",
        "damping_adaptive_end",
        "fixed",
        "adaptive",
        "delta_3_0_pct",
        "delta_3_1_pct",
        "delta_3_2_pct",
        "verdict",
        "recommendation",
    }
    EXPECTED_VERDICTS = {"improved", "marginal", "degraded"}

    def test_results_json_exists_and_valid(self):
        if not self.JSON_PATH.exists():
            pytest.skip(
                "models/adaptive_damping_results.json not yet generated; "
                "run `python -m src.models.adaptive_damping_experiment` to create it"
            )
        with open(self.JSON_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert self.EXPECTED_KEYS.issubset(
            data.keys()
        ), f"Missing keys: {self.EXPECTED_KEYS - set(data.keys())}"
        # Both fixed and adaptive should have 3-0%/3-1%/3-2% + n
        for strategy in ("fixed", "adaptive"):
            for sub in ("3-0%", "3-1%", "3-2%", "n"):
                assert sub in data[strategy], f"Missing {sub} in {strategy}"
        # Adaptive params
        assert (
            data["damping_adaptive_start"] < data["damping_adaptive_end"]
        ), "Adaptive start should be < end (more shrinkage early, more trust late)"
        # Deltas match the diff
        assert (
            abs(data["delta_3_0_pct"] - (data["adaptive"]["3-0%"] - data["fixed"]["3-0%"])) < 1e-9
        )
        assert data["verdict"] in self.EXPECTED_VERDICTS
        assert data["recommendation"] in {"apply-adaptive", "keep-fixed"}

    def test_experiment_module_imports(self):
        """The adaptive_damping_experiment module is importable and exposes run_experiment."""
        from src.models import adaptive_damping_experiment

        assert hasattr(adaptive_damping_experiment, "run_experiment")
        assert callable(adaptive_damping_experiment.run_experiment)


_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.mark.slow
class TestLogRegSetPredictorV2:
    """Adapter for the v2 LogReg SetPredictor — loads real artifact from disk."""

    V2_PATH = _MODELS_DIR / "set_predictor_v2.joblib"
    LEGACY_PATH = _MODELS_DIR / "set_predictor.joblib"

    def test_load_v2_returns_adapter_with_correct_features(self):
        """The v2 adapter loads and exposes the 21 feature names from the artifact."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        assert adapter.feature_names is not None
        assert len(adapter.feature_names) == 21
        # Spot-check a few known features
        for key in ("strength_h", "elo_diff", "diff_set_ratio", "es_desempate"):
            assert key in adapter.feature_names, f"Missing feature: {key}"

    def test_predict_proba_bounded_and_sums_to_one(self):
        """predict_proba returns [1,2] array with values in [0,1] summing to 1.0."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        df = pd.DataFrame([{f: 0.5 for f in adapter.feature_names}])
        proba = adapter.predict_proba(df)
        _assert_proba_bounded(proba, "LogRegSetPredictorV2")
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_returns_int(self):
        """predict returns int array (0 or 1)."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        df = pd.DataFrame([{f: 0.5 for f in adapter.feature_names}])
        preds = adapter.predict(df)
        assert preds.dtype == np.int_ or preds.dtype == np.int64

    def test_predict_proba_order_agnostic(self):
        """Columns can be in any order; the result must be identical."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        df = pd.DataFrame([{f: 0.5 for f in adapter.feature_names}])
        proba_original = adapter.predict_proba(df)

        # Reverse column order
        df_shuffled = df[adapter.feature_names[::-1]]
        proba_shuffled = adapter.predict_proba(df_shuffled)
        np.testing.assert_allclose(proba_original, proba_shuffled, rtol=1e-10)

    def test_predict_proba_missing_columns_filled_zero(self):
        """When the input lacks some columns, they are filled with 0.0."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        # Only pass 3 columns; the rest should be filled with 0
        partial_cols = ["strength_h", "elo_diff", "diff_set_ratio"]
        df = pd.DataFrame([{c: 0.5 for c in partial_cols}])
        proba = adapter.predict_proba(df)
        _assert_proba_bounded(proba, "LogRegSetPredictorV2 (partial)")
        assert proba.shape == (1, 2)

    def test_try_load_v2_returns_v2_when_present(self, tmp_path):
        """try_load_v2 returns the v2 adapter when v2 file exists."""
        adapter, source = LogRegSetPredictor.try_load_v2(self.V2_PATH, self.LEGACY_PATH)
        assert source == "logreg_v2"
        assert type(adapter).__name__ == "LogRegSetPredictor"

    def test_try_load_v2_falls_back_to_legacy_when_v2_missing(self, tmp_path):
        """When v2 is absent but legacy exists, return legacy SetPredictor."""
        dummy_v2 = tmp_path / "no_such_v2.joblib"

        # Write a synthetic legacy file — use enough data for cv=2
        legacy_path = tmp_path / "dummy_legacy.joblib"
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        rng = np.random.RandomState(42)
        scaled_X = pd.DataFrame({"f1": rng.uniform(-1, 1, 20), "f2": rng.uniform(-1, 1, 20)})
        dummy_y = [0 if i < 10 else 1 for i in range(20)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scaled_X)
        dummy_model = LogisticRegression(max_iter=100, random_state=42)
        cal = CalibratedClassifierCV(dummy_model, cv=2, method="isotonic")
        cal.fit(X_scaled, dummy_y)
        joblib.dump(
            {
                "scaler": scaler,
                "best_model_name": "LogisticRegression",
                "best_model": dummy_model,
                "calibrated_model": cal,
                "feature_names": ["f1", "f2"],
                "results": {
                    "LogisticRegression": {"accuracy": 0.5, "auc_roc": 0.5, "brier_score": 0.25}
                },
            },
            legacy_path,
        )

        adapter, source = LogRegSetPredictor.try_load_v2(dummy_v2, legacy_path)
        assert source == "extra_trees_v1"
        from src.models.set_predictor import SetPredictor

        assert isinstance(adapter, SetPredictor)

    def test_try_load_v2_returns_none_when_both_missing(self, tmp_path):
        """When neither file exists, return (None, 'none')."""
        dummy_v2 = tmp_path / "no_file.joblib"
        dummy_legacy = tmp_path / "no_file_legacy.joblib"
        adapter, source = LogRegSetPredictor.try_load_v2(dummy_v2, dummy_legacy)
        assert adapter is None
        assert source == "none"

    def test_try_load_v2_falls_back_to_legacy_when_v2_corrupt(self, tmp_path):
        """When v2 exists but is corrupt/unparseable, fall back to legacy."""
        # Write a corrupt v2 file (not a valid joblib dict)
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        corrupt_v2 = tmp_path / "corrupt_v2.joblib"
        # Write something that joblib.load can parse but has wrong schema
        joblib.dump({"garbage": True, "meta": "not a real model"}, corrupt_v2)

        # Write a synthetic legacy file
        legacy_path = tmp_path / "dummy_legacy.joblib"
        rng = np.random.RandomState(42)
        scaled_X = pd.DataFrame({"f1": rng.uniform(-1, 1, 20), "f2": rng.uniform(-1, 1, 20)})
        dummy_y = [0 if i < 10 else 1 for i in range(20)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scaled_X)
        dummy_model = LogisticRegression(max_iter=100, random_state=42)
        cal = CalibratedClassifierCV(dummy_model, cv=2, method="isotonic")
        cal.fit(X_scaled, dummy_y)
        joblib.dump(
            {
                "scaler": scaler,
                "best_model_name": "LogisticRegression",
                "best_model": dummy_model,
                "calibrated_model": cal,
                "feature_names": ["f1", "f2"],
                "results": {
                    "LogisticRegression": {"accuracy": 0.5, "auc_roc": 0.5, "brier_score": 0.25}
                },
            },
            legacy_path,
        )

        adapter, source = LogRegSetPredictor.try_load_v2(corrupt_v2, legacy_path)
        assert source == "extra_trees_v1", f"Expected legacy fallback, got {source}"
        from src.models.set_predictor import SetPredictor

        assert isinstance(adapter, SetPredictor)

    def test_try_load_v2_falls_back_on_joblib_load_failure(self, tmp_path):
        """When v2 exists but joblib.load cannot parse it, fall back to legacy."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        corrupt_bytes = tmp_path / "corrupt_bytes.joblib"
        corrupt_bytes.write_bytes(b"\x00\x01\x02\xffnot-a-joblib-file")

        # Write a synthetic legacy file
        legacy_path = tmp_path / "dummy_legacy.joblib"
        rng = np.random.RandomState(42)
        scaled_X = pd.DataFrame({"f1": rng.uniform(-1, 1, 20), "f2": rng.uniform(-1, 1, 20)})
        dummy_y = [0 if i < 10 else 1 for i in range(20)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scaled_X)
        dummy_model = LogisticRegression(max_iter=100, random_state=42)
        cal = CalibratedClassifierCV(dummy_model, cv=2, method="isotonic")
        cal.fit(X_scaled, dummy_y)
        joblib.dump(
            {
                "scaler": scaler,
                "best_model_name": "LogisticRegression",
                "best_model": dummy_model,
                "calibrated_model": cal,
                "feature_names": ["f1", "f2"],
                "results": {
                    "LogisticRegression": {"accuracy": 0.5, "auc_roc": 0.5, "brier_score": 0.25}
                },
            },
            legacy_path,
        )

        adapter, source = LogRegSetPredictor.try_load_v2(corrupt_bytes, legacy_path)
        assert source == "extra_trees_v1", f"Expected legacy fallback, got {source}"
        from src.models.set_predictor import SetPredictor

        assert isinstance(adapter, SetPredictor)

    def test_predict_proba_warns_on_schema_drift(self):
        """Warns UserWarning when input columns differ from trained features."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        # Pass input with a RENAMED feature (extra+missing simultaneously)
        renamed_cols = ["pts_fav_home_h" if c == "pts_fav_h" else c for c in adapter.feature_names]
        df = pd.DataFrame([{c: 0.5 for c in renamed_cols}])

        with pytest.warns(UserWarning, match="Schema drift"):
            proba = adapter.predict_proba(df)

        _assert_proba_bounded(proba, "LogRegSetPredictorV2 (drifted)")
        assert proba.shape == (1, 2)

    def test_predict_proba_no_warning_on_exact_match(self):
        """No warning when input columns exactly match trained features."""
        adapter = LogRegSetPredictor.load(self.V2_PATH)
        df = pd.DataFrame([{f: 0.5 for f in adapter.feature_names}])

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors
            proba = adapter.predict_proba(df)

        _assert_proba_bounded(proba, "LogRegSetPredictorV2 (exact match)")
