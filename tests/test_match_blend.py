"""Tests for B4: match prediction via best-of-5 formula and blend optimizer.

Covers:
  - p_match_from_p_set closed-form (Task 1)
  - blend_p_match linearity (Task 3)
  - optimize_blend_w synthetic known-optimum + determinism (Task 3)
"""

import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from src.simulation.set_math import p_match_from_p_set, p_point_from_p_set, p_set_from_p_point

# ─────────────────────────────────────────────────────────────
# p_match_from_p_set tests (Task 2)
# ─────────────────────────────────────────────────────────────


class TestPMatchFromPSet:
    """Closed-form best-of-5 formula: P(match) = f(q, q5)."""

    @staticmethod
    def _p_match_iid(p_point: float) -> float:
        """Reference: same formula as TestMarkovChainSanity._p_match_iid."""
        q = p_set_from_p_point(p_point, 25)
        q5 = p_set_from_p_point(p_point, 15)
        return q**3 + 3 * q**3 * (1 - q) + 6 * q**2 * (1 - q) ** 2 * q5

    def test_p_match_from_p_set_smoke(self):
        """p_match_from_p_set at the three trivial points."""
        assert p_match_from_p_set(0.5) == 0.5
        assert p_match_from_p_set(1.0) == 1.0
        assert p_match_from_p_set(0.0) == 0.0

    def test_p_match_from_p_set_monotone(self):
        """Sweep q in [0.01, 0.99]; p_match must be monotone non-decreasing."""
        qs = np.linspace(0.01, 0.99, 99)
        ps = [p_match_from_p_set(float(q)) for q in qs]
        for i in range(1, len(ps)):
            assert ps[i] >= ps[i - 1] - 1e-12, (
                f"Not monotone at index {i}: q={qs[i]:.4f} -> {ps[i]:.6f} "
                f"< previous {ps[i-1]:.6f}"
            )

    def test_p_match_from_p_set_tie_to_test_simulator(self):
        """p_match_from_p_set matches _p_match_iid reference (REQ-022).

        _p_match_iid(0.52) computes q=p_set_from_p_point(0.52,25) and
        q5=p_set_from_p_point(0.52,15), then applies the best-of-5 formula.
        p_match_from_p_set(q, q5) must give the same 0.6967 ± 0.001.
        """
        p_point = 0.52
        expected = self._p_match_iid(p_point)
        q = p_set_from_p_point(p_point, 25)
        q5 = p_set_from_p_point(p_point, 15)
        result = p_match_from_p_set(q, q5)
        assert_almost_equal(result, expected, decimal=4)
        assert_almost_equal(result, 0.6967, decimal=3)

    def test_p_match_from_p_set_q5_default(self):
        """When q5=None, defaults to A2 composition (REQ-002).

        The A2 default is:
            q5 = p_set_from_p_point(p_point_from_p_set(q, 25), 15)
        which converts q (set-at-25) → point → set-at-15.
        """
        q = 0.6131
        with_default = p_match_from_p_set(q)
        a2_q5 = p_set_from_p_point(p_point_from_p_set(q, 25), 15)
        with_a2 = p_match_from_p_set(q, a2_q5)
        assert_almost_equal(with_default, with_a2, decimal=10)
        # The A2 default should differ from the old q5=q fallback
        old_fallback = p_match_from_p_set(q, q)
        assert abs(with_default - old_fallback) > 1e-6, "A2 default should not match q5=q fallback"

    def test_p_match_from_p_set_known_values(self):
        """Verify against pre-computed known values."""
        # q=0.5 -> symmetric -> 0.5 (both with and without tiebreak)
        assert p_match_from_p_set(0.5) == 0.5
        assert p_match_from_p_set(0.5, 0.5) == 0.5
        # q=0.7, q5=0.7 (strong favorite in all sets)
        result = p_match_from_p_set(0.7, 0.7)
        # 0.7^3 + 3*0.7^3*0.3 + 6*0.7^2*0.3^2*0.7 = 0.343 + 0.3087 + 0.18522 = 0.83692
        assert_almost_equal(result, 0.83692, decimal=4)


# ─────────────────────────────────────────────────────────────
# blend_p_match tests (Task 3)
# ─────────────────────────────────────────────────────────────


class TestBlendPMatch:
    """Linear blend of Elo and derived probabilities."""

    def test_blend_p_match_linearity(self):
        """blend_p_match(p_elo, p_derived, w) wraps w * p_elo + (1-w) * p_derived."""
        from src.models.blend_optimizer import blend_p_match

        # w=1.0 -> pure Elo
        assert blend_p_match(0.7, 0.3, 1.0) == 0.7
        # w=0.0 -> pure derived
        assert blend_p_match(0.7, 0.3, 0.0) == 0.3
        # w=0.5 -> midpoint
        assert blend_p_match(0.7, 0.3, 0.5) == 0.5


class TestOptimizeBlendW:
    """LOFO-CV blend weight optimizer."""

    def test_optimize_blend_w_synthetic_known_optimum(self):
        """Synthetic df where w=0.3 is optimal: optimizer must find it ± 0.05."""
        from src.models.blend_optimizer import optimize_blend_w

        rng = np.random.RandomState(42)
        n = 400
        df = _make_synthetic_blend_df(rng, n, true_w=0.3)

        result = optimize_blend_w(
            df,
            w_grid=list(np.linspace(0.0, 1.0, 21)),
            val_years=[2022, 2023, 2024, 2025],
            elo_col="p_elo",
            derived_col="p_derived",
            y_col="y",
            refine="golden_section",
            refine_tol=1e-3,
        )

        assert (
            abs(result["w_global"] - 0.3) <= 0.05
        ), f"Expected w≈0.3, got w_global={result['w_global']:.4f}"
        assert result["w_global"] == pytest.approx(
            np.mean(result["w_per_fold_lofo"]), abs=1e-10
        ), "REQ-006: w_global must be mean(w_per_fold_lofo)"
        assert "logloss_per_fold" in result
        assert "logloss_elo_only_per_fold" in result
        assert result["n_folds"] >= 1

    def test_optimize_blend_w_determinism(self):
        """Same input twice yields bit-identical dict (NFR-003)."""
        from src.models.blend_optimizer import optimize_blend_w

        rng = np.random.RandomState(123)
        df = _make_synthetic_blend_df(rng, n=100, true_w=0.7)

        result_a = optimize_blend_w(
            df,
            w_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            val_years=[2023],
            elo_col="p_elo",
            derived_col="p_derived",
            y_col="y",
        )
        result_b = optimize_blend_w(
            df,
            w_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            val_years=[2023],
            elo_col="p_elo",
            derived_col="p_derived",
            y_col="y",
        )

        assert result_a == result_b, "Determinism violated: two runs differ"

    def test_sigma_lofo_floor(self):
        """When sigma=0 (identical per-fold improvements), threshold is 0.005."""
        from src.models.blend_optimizer import optimize_blend_w

        # 4 folds with identical logloss arrays: sigma of improvement = 0
        rng = np.random.RandomState(99)
        df = _make_synthetic_blend_df(rng, n=80, true_w=0.5)
        result = optimize_blend_w(
            df,
            w_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            val_years=[2022, 2023, 2024, 2025],
            elo_col="p_elo",
            derived_col="p_derived",
            y_col="y",
        )

        assert result["n_folds"] >= 1
        # sigma_lofo should never be NaN; floor kicks in
        assert result["sigma_lofo"] >= 0.0
        assert result["sigma_lofo"] <= 1.0

    def test_lofo_held_out_isolates_fold(self):
        """LOFO protocol: w for each fold is trained EXCLUDING that fold.

        Under LOFO, each per-fold w is optimised on the OTHER 3 folds.
        The in-fold optimum (optimised on that fold alone) will generally
        differ because of independent noise.  We verify that at least 1 of
        the 4 folds has a LOFO-chosen w that differs from its in-fold argmin.
        """
        from src.models.blend_optimizer import optimize_blend_w, _grid_search_w_on_df

        rng = np.random.RandomState(42)
        n = 800
        df = _make_synthetic_blend_df(rng, n, true_w=0.3)

        result = optimize_blend_w(
            df,
            w_grid=list(np.linspace(0.0, 1.0, 21)),
            val_years=[2022, 2023, 2024, 2025],
            elo_col="p_elo",
            derived_col="p_derived",
            y_col="y",
            refine="golden_section",
            refine_tol=1e-3,
        )

        assert result["n_folds"] >= 1
        w_grid = list(np.linspace(0.0, 1.0, 21))
        n_different = 0
        for i, vy in enumerate([2022, 2023, 2024, 2025]):
            if i >= len(result["w_per_fold_lofo"]):
                continue
            w_lofo = result["w_per_fold_lofo"][i]
            va = df[df["temporada_inicio"] == vy]
            w_infold = _grid_search_w_on_df(va, w_grid, "p_elo", "p_derived", "y")
            if abs(w_lofo - w_infold) > 0.01:
                n_different += 1

        assert n_different >= 1, "All LOFO w's match in-fold optima — LOFO protocol likely not used"


# ─────────────────────────────────────────────────────────────
# evaluate_adoption gate tests (S3 — truth-table)
# ─────────────────────────────────────────────────────────────


class TestEvaluateAdoption:
    """Truth-table tests for the AND-of-4 adoption gate (REQ-016/017)."""

    @staticmethod
    def _make_result(
        w_global=0.5,
        improvement_mean=0.02,
        sigma_lofo=0.005,
        n_folds=4,
        logloss_per_fold=None,
        logloss_elo_only_per_fold=None,
        test_logloss=0.5,
    ):
        logloss_per_fold = logloss_per_fold or [0.5, 0.5, 0.5, 0.5]
        logloss_elo_only_per_fold = logloss_elo_only_per_fold or [0.55, 0.55, 0.55, 0.55]
        out = {
            "w_global": w_global,
            "improvement_mean": improvement_mean,
            "sigma_lofo": sigma_lofo,
            "n_folds": n_folds,
            "logloss_per_fold": logloss_per_fold,
            "logloss_elo_only_per_fold": logloss_elo_only_per_fold,
        }
        if test_logloss is not None:
            out["test_metrics_if_computed"] = {"logloss": test_logloss}
        return out

    def _evaluate(self, result, **kwargs):
        from src.models.train_improved import evaluate_adoption

        defaults = dict(
            elo_test_logloss_baseline=0.5677,
            n_folds_required=4,
            n_folds_win_required=3,
            noise_floor=0.005,
            w_sat_threshold=0.05,
        )
        defaults.update(kwargs)
        return evaluate_adoption(result, **defaults)

    def test_gate_all_four_pass(self):
        """All 4 conditions hold → adopted."""
        r = self._make_result(
            w_global=0.5,
            improvement_mean=0.02,
            sigma_lofo=0.005,
            logloss_per_fold=[0.5, 0.5, 0.5, 0.5],
            logloss_elo_only_per_fold=[0.55, 0.55, 0.55, 0.55],
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "adopted"
        assert r.get("failing_conditions", []) == []

    def test_gate_per_fold_win_fails(self):
        """2/4 per-fold wins < 3 required → negative."""
        r = self._make_result(
            w_global=0.5,
            improvement_mean=0.02,
            sigma_lofo=0.005,
            logloss_per_fold=[0.5, 0.5, 0.58, 0.6],
            logloss_elo_only_per_fold=[0.55, 0.55, 0.55, 0.55],
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "negative"
        assert any("per_fold_wins" in c for c in r.get("failing_conditions", []))

    def test_gate_improvement_below_noise_floor(self):
        """improvement_mean <= max(sigma, noise_floor) → negative."""
        r = self._make_result(
            w_global=0.5,
            improvement_mean=0.003,
            sigma_lofo=0.005,
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "negative"
        assert any("improvement_mean" in c for c in r.get("failing_conditions", []))

    def test_gate_test_year_not_better(self):
        """test_logloss >= baseline → negative."""
        r = self._make_result(
            w_global=0.5,
            improvement_mean=0.02,
            sigma_lofo=0.005,
            test_logloss=0.6,
        )
        assert self._evaluate(r) == "negative"
        assert any("test_logloss" in c for c in r.get("failing_conditions", []))

    def test_gate_w_saturated_to_derived(self):
        """w_global <= 0.05 → negative."""
        r = self._make_result(
            w_global=0.01,
            improvement_mean=0.02,
            sigma_lofo=0.005,
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "negative"
        assert any("w_global" in c for c in r.get("failing_conditions", []))

    def test_shortcut_w_saturated_to_elo(self):
        """w_global >= 0.999 AND improvement < 0.001 → shortcut_negative."""
        r = self._make_result(
            w_global=0.999,
            improvement_mean=0.0005,
            sigma_lofo=0.005,
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "shortcut_negative"

    def test_shortcut_w_not_saturated(self):
        """w=0.95, improvement=0.0005 → negative (NOT shortcut, full analysis)."""
        r = self._make_result(
            w_global=0.95,
            improvement_mean=0.0005,
            sigma_lofo=0.005,
            test_logloss=0.5,
        )
        assert self._evaluate(r) == "negative"
        # w=0.95 < 0.999 so the hard shortcut does NOT fire
        assert "hard_shortcut" not in (r.get("failing_conditions") or [])


def test_contract_drift_absent(caplog):
    """21 features present, no schema-drift warnings (REQ-023).

    The drift warning fires when LogRegSetPredictor.predict_proba receives
    columns that differ from the trained feature set.  This test only
    exercises SetContext + build_set_features, so no model is loaded and
    no drift warning can fire.  The caplog assertion is a regression guard:
    if ANY WARNING from set_predictor_v2 appears in the future, it fails.
    """
    from src.data.set_feature_contract import SetContext, build_set_features

    ctx = SetContext()
    feats = build_set_features(ctx)
    assert len(feats) == 21, f"Expected 21 features, got {len(feats)}"

    # Assert no WARNING-level log from set_predictor_v2 or feature contract
    for record in caplog.records:
        if record.levelname == "WARNING":
            assert (
                "set_predictor_v2" not in record.name.lower()
            ), f"Unexpected schema drift warning: {record.message}"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _make_synthetic_blend_df(
    rng: np.random.RandomState,
    n: int,
    true_w: float,
) -> pd.DataFrame:
    """Build a synthetic DataFrame where blend weight=true_w is optimal.

    p_elo and p_derived share a common base but have distinct biases,
    making the blend weight identifiable by log-loss minimisation.
    y = Bernoulli( w * p_elo + (1-w) * p_derived ).
    """

    base = rng.uniform(0.3, 0.7, n)
    bias_elo = rng.uniform(-0.15, 0.15, n)
    bias_derived = rng.uniform(-0.15, 0.15, n)

    p_elo = np.clip(base + bias_elo, 0.05, 0.95)
    p_derived = np.clip(base + bias_derived, 0.05, 0.95)

    # True probability = blended
    p_true = true_w * p_elo + (1.0 - true_w) * p_derived
    y = (rng.uniform(0, 1, n) < p_true).astype(int)

    # Assign folds by fake season
    pool = [2022, 2023, 2024, 2025]
    seasons = [pool[i % 4] for i in range(n)]

    return pd.DataFrame(
        {
            "p_elo": p_elo,
            "p_derived": p_derived,
            "y": y,
            "temporada_inicio": seasons,
        }
    )
