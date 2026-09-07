"""Truth-table tests for the B5-adapted AND-of-4 gate in `evaluate_adoption`.

REQ-024: a 6-case truth-table for the B5 gate plus 1 B4 regression case.
B4 #200 lesson: the B4 path must keep returning the same verdicts as before
the B5 extension (R-DRIFT-1).
"""

from __future__ import annotations

from src.models.train_improved import evaluate_adoption

# ── Helpers ──────────────────────────────────────────────────────────────────


def _b5_result(
    *,
    churn_coef_per_fold: list[float] | None = None,
    churn_coef_global: float = 0.5,
    churn_coef_std_err: float = 0.2,
    z_stat: float = 2.0,
    improvement_mean: float = 0.02,
    sigma_lofo: float = 0.003,
    logloss_mean: float = 0.5550,
    logloss_elo_only_mean: float = 0.5650,
    logloss_constant: float = 0.6931,
    test_logloss: float = 0.5550,
    elo_baseline: float = 0.5677,
) -> dict:
    """Build a synthetic B5 result dict for the truth-table cases.

    Defaults are chosen to make ALL 4 conditions pass and trigger ``adopted``,
    unless a specific field is overridden in the test.
    """
    return {
        "churn_coef_per_fold": (
            churn_coef_per_fold if churn_coef_per_fold is not None else [0.3, 0.4, 0.5, 0.6]
        ),
        "churn_coef_global": churn_coef_global,
        "churn_coef_std_err": churn_coef_std_err,
        "z_stat": z_stat,
        "improvement_mean": improvement_mean,
        "sigma_lofo": sigma_lofo,
        "logloss_mean": logloss_mean,
        "logloss_elo_only_mean": logloss_elo_only_mean,
        "logloss_constant": logloss_constant,
        "test_metrics_if_computed": {"logloss": test_logloss},
    }


# ── 6 truth-table cases for the B5 gate (REQ-024) ────────────────────────────


def test_b5_all_pass_adopted() -> None:
    """All 4 conditions pass → verdict ``adopted``, ``failing_conditions=[]``."""
    result = _b5_result()  # defaults: all 4 conditions pass
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "adopted"
    assert result["failing_conditions"] == []


def test_b5_cond1_fail() -> None:
    """Cond1 fails (2/4 positive folds) → ``negative``, lists cond1."""
    result = _b5_result(churn_coef_per_fold=[0.3, -0.1, 0.4, -0.2])  # 2 positive
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "negative"
    assert any("cond1" in c for c in result["failing_conditions"])


def test_b5_cond2_fail() -> None:
    """Cond2 fails (improvement below noise floor) → ``negative``, lists cond2."""
    result = _b5_result(
        improvement_mean=0.001,  # below default noise_floor=0.005
        sigma_lofo=0.001,
    )
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "negative"
    assert any("cond2" in c for c in result["failing_conditions"])


def test_b5_cond3_fail() -> None:
    """Cond3 fails (test-2025 logloss ≥ baseline) → ``negative``, lists cond3."""
    result = _b5_result(test_logloss=0.5800)  # > baseline 0.5677
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "negative"
    assert any("cond3" in c for c in result["failing_conditions"])


def test_b5_cond4_fail() -> None:
    """Cond4 fails (coef ≤ 0) → ``negative``, lists cond4."""
    result = _b5_result(
        churn_coef_global=-0.1,  # negative coef
        z_stat=-1.5,
    )
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "negative"
    assert any("cond4" in c for c in result["failing_conditions"])


def test_b5_cond4_fail_low_z() -> None:
    """Cond4 fails (positive coef but |z| ≤ 1) → ``negative``, lists cond4."""
    result = _b5_result(
        churn_coef_global=0.05,
        z_stat=0.8,  # |z| < 1.0
    )
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "negative"
    assert any("cond4" in c for c in result["failing_conditions"])


def test_b5_shortcut_negative() -> None:
    """Hard shortcut (|coef|≈0 AND logloss==constant) → ``shortcut_negative``."""
    result = _b5_result(
        churn_coef_global=0.0,  # |coef| < 1e-6
        logloss_mean=0.6931,  # equal to logloss_constant
    )
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=0.5677,
        gate="b5",
    )
    assert verdict == "shortcut_negative"
    assert result["failing_conditions"] == ["hard_shortcut"]


# ── B4 regression (R-DRIFT-1: B4 path unchanged after the additive extension) ─


def test_b4_path_unchanged() -> None:
    """B4 path with the new ``gate`` default still returns the B4 verdicts.

    The B4 gate uses ``w_global`` (not a LogReg coefficient) and the AND-of-4
    win/improvement/sigma/w_sat pattern. A NEGATIVE B4 result (w_global=0.5,
    1/4 wins) must still return ``negative`` after the additive extension.
    """
    b4_result = {
        "w_global": 0.5,
        "w_per_fold_lofo": [0.5, 0.5, 0.5, 0.5],
        "logloss_per_fold": [0.6, 0.6, 0.6, 0.6],
        "logloss_elo_only_per_fold": [0.6, 0.6, 0.6, 0.6],
        "logloss_mean": 0.6,
        "logloss_elo_only_mean": 0.6,
        "improvement_mean": 0.0,
        "sigma_lofo": 0.0,
        "n_folds": 4,
        "n_wins": 1,  # 1/4 — fails the win condition
    }
    verdict = evaluate_adoption(
        b4_result,
        elo_test_logloss_baseline=0.5677,
        gate="b4",  # explicit B4
    )
    assert verdict == "negative"
    assert any("wins" in c for c in b4_result["failing_conditions"])
