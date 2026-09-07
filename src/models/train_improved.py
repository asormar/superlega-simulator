"""
train_improved.py — Entrenamiento mejorado (resultado del PLAN_MEJORA_PRECISION).

Consolida los hallazgos de las fases del plan en artefactos reproducibles,
SIN romper la pipeline de producción existente. Guarda modelos v2 y un
snapshot de precisión para comparar antes/después.

El snapshot sigue la misma forma que precision_baseline.json:
  - champion: nombre del modelo campeón
  - cv: métricas de validación rolling-origin (media ± std sobre folds)
  - test: métricas en el conjunto de test held-out
  - n_features: número de features usadas

Hallazgos clave (ver comparación al final del plan):
  1. Las features enriquecidas por temporada completa tenían LEAKAGE; medidas
     honestamente el AUC de match era 0.53, no 0.71.
  2. Con dataset pequeño y ruidoso, un modelo Elo con margen de victoria
     (sin entrenamiento) supera a los árboles: AUC 0.75, logloss 0.585.
  3. Las temporadas viejas (2016-2020, 34-55 partidos, home-win ~0.35)
     enseñan el signo equivocado. Recencia (half-life ~1.5) o entrenar solo
     2022+ lo corrige.
  4. Modelos lineales regularizados (LogReg) baten a los árboles profundos
     en este régimen de datos, tanto en set como en match.

Config final:
  - MATCH: probabilidad de Elo con margen (rolling, sin leakage).
  - SET: LogReg C=0.5 con pesos de recencia (half-life=2), features de set.
"""

import sys
import json
import logging
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    brier_score_loss,
)

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import SET_FEATURE_COLS
from src.data.rolling_features import build_rolling_match_features
from src.data.set_feature_contract import SetContext, build_set_features
from src.simulation.set_math import p_match_from_p_set

b4_logger = logging.getLogger("b4_blend")

MODELS_DIR = BASE_DIR / "models"

# Config final (derivada de la experimentación de las fases)
MATCH_FEATURES = [
    "elo_diff",
    "diff_form_ewma",
    "h2h_win_rate_h",
    "diff_set_ratio",
    "diff_point_ratio",
]
RECENT_TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
SET_RECENCY_HALFLIFE = 2.0


def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "n": int(len(y)),
    }


def _set_rolling_cv(data: dict) -> dict:
    """Rolling-origin CV for the v2 LogReg (C=0.5, recency half-life=2).

    Folds (sliding window, train fijo 2022-2024):
      - train [2022, 2023] → val 2024
      - train [2023, 2024] → val 2025

    Returns dict with logloss_mean, logloss_std, auc_mean, auc_std,
    brier_mean, acc_mean, n_folds.
    """
    ds = data["set_features"].copy()
    cols = [c for c in SET_FEATURE_COLS if c in ds.columns]

    fold_configs = [
        ([2022, 2023], 2024),
        ([2023, 2024], 2025),
    ]

    per_fold = {"logloss": [], "auc": [], "brier": [], "acc": []}

    for train_years, val_year in fold_configs:
        tr = ds[ds.temporada_inicio.isin(train_years)]
        va = ds[ds.temporada_inicio == val_year]
        if len(va) == 0 or len(tr) == 0:
            continue
        X_tr, y_tr = tr[cols].fillna(0), tr["ganador_set_local"]
        X_va, y_va = va[cols].fillna(0), va["ganador_set_local"]
        if y_va.nunique() < 2:
            continue

        # Recency weights: closer seasons matter more
        sw = 0.5 ** ((val_year - tr.temporada_inicio.values) / SET_RECENCY_HALFLIFE)

        model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        model.fit(X_tr, y_tr, sample_weight=sw)
        p = model.predict_proba(X_va)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)

        per_fold["logloss"].append(log_loss(y_va, p))
        per_fold["auc"].append(roc_auc_score(y_va, p))
        per_fold["brier"].append(brier_score_loss(y_va, p))
        per_fold["acc"].append(accuracy_score(y_va, (p >= 0.5).astype(int)))

    out = {"n_folds": len(per_fold["logloss"])}
    for k, v in per_fold.items():
        out[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
        out[f"{k}_std"] = float(np.std(v)) if v else float("nan")
    return out


def train_match(dfm: pd.DataFrame) -> dict:
    """Modelo de match: probabilidad de Elo con margen (sin entrenamiento).

    Returns dict con la misma forma que precision_baseline.json (match).
    """
    te = dfm[dfm.temporada_inicio == TEST_SEASON]
    p_elo = te["elo_win_prob_h"].values
    m = _metrics(te["gana_local"], p_elo)

    # El "modelo" es el sistema Elo (parámetros en rolling_features).
    joblib.dump(
        {
            "type": "margin_elo",
            "features": ["elo_win_prob_h"],
            "note": "Probabilidad de Elo con margen, rolling sin leakage.",
        },
        MODELS_DIR / "match_elo_v2.joblib",
    )

    return {
        "champion": "margin_elo",
        "cv": {
            "n_folds": 0,
            "note": "Elo es determinista (sin entrenamiento), no requiere CV.",
        },
        "test": {
            "test_season": TEST_SEASON,
            "n_test": m["n"],
            "logloss": m["logloss"],
            "auc": m["auc"],
            "brier": m["brier"],
            "acc": m["acc"],
        },
        "n_features": 1,  # solo elo_win_prob_h
    }


def train_set(data: dict) -> dict:
    """Modelo de set: LogReg regularizado con pesos de recencia.

    Returns dict con la misma forma que precision_baseline.json (set).
    Lee el CSV de features v2 (A3 contract) en lugar del legacy de la pipeline.
    """
    v2_path = BASE_DIR / "DB" / "features" / "set_features_v2.csv"
    ds = pd.read_csv(v2_path)
    data["set_features"] = ds  # so _set_rolling_cv uses the same v2 data
    cols = [c for c in SET_FEATURE_COLS if c in ds.columns]

    tr = ds[ds.temporada_inicio.isin(RECENT_TRAIN_SEASONS)]
    te = ds[ds.temporada_inicio == TEST_SEASON]
    sw = 0.5 ** ((TEST_SEASON - tr.temporada_inicio.values) / SET_RECENCY_HALFLIFE)

    model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    model.fit(tr[cols].fillna(0), tr["ganador_set_local"], sample_weight=sw)
    p = model.predict_proba(te[cols].fillna(0))[:, 1]
    test_m = _metrics(te["ganador_set_local"], p)

    # Guardar artefacto v2 (sin cambios)
    joblib.dump(
        {
            "type": "logreg_recency",
            "model": model,
            "features": cols,
            "recency_halflife": SET_RECENCY_HALFLIFE,
            "train_seasons": RECENT_TRAIN_SEASONS,
        },
        MODELS_DIR / "set_predictor_v2.joblib",
    )

    # Rolling-origin CV (independiente del test)
    cv = _set_rolling_cv(data)

    return {
        "champion": "LogisticRegression",
        "cv": cv,
        "test": {
            "test_season": TEST_SEASON,
            "n_test": test_m["n"],
            "logloss": test_m["logloss"],
            "auc": test_m["auc"],
            "brier": test_m["brier"],
            "acc": test_m["acc"],
        },
        "n_features": len(cols),
    }


def evaluate_adoption(
    result: dict,
    *,
    elo_test_logloss_baseline: float,
    n_folds_required: int = 4,
    n_folds_win_required: int = 3,
    noise_floor: float = 0.005,
    w_sat_threshold: float = 0.05,
    gate: str = "b4",
) -> str:
    """Evaluate the B4 or B5 adoption gate (REQ-016 / REQ-017 / REQ-019).

    Args:
        result: Dict with per-fold and global metrics.
        elo_test_logloss_baseline: Elo-only test logloss from precision_improved.json.
        n_folds_required: Number of LOFO folds required (default 4).
        n_folds_win_required: Number of folds where improvement is positive (default 3).
        noise_floor: Minimum improvement threshold (default 0.005).
        w_sat_threshold: Saturation threshold for B4 w_global (default 0.05).
        gate: ``"b4"`` (default, B4 AND-of-4) or ``"b5"`` (B5-adapted AND-of-4).

    Returns one of ``"adopted"``, ``"negative"``, or ``"shortcut_negative"``.
    Populates ``result["failing_conditions"]`` with human-readable names of
    conditions that FAILED (empty list on adoption).
    """
    failing_conditions = []

    if gate == "b5":
        # ── B5-adapted AND-of-4 gate (REQ-017..REQ-021) ────────────────
        # Hard shortcut (REQ-021)
        churn_coef = result.get("churn_coef_global", 0.0)
        logloss_logreg = result.get("logloss_mean", float("inf"))
        logloss_constant = result.get("logloss_constant", float("inf"))
        if abs(churn_coef) < 1e-6 and abs(logloss_logreg - logloss_constant) < 1e-9:
            failing_conditions.append("hard_shortcut")
            result["failing_conditions"] = failing_conditions
            return "shortcut_negative"

        # Condition 1: count(positive churn_coef_per_fold) >= 3
        churn_coefs = result.get("churn_coef_per_fold", [])
        n_positive = sum(1 for c in churn_coefs if c > 0)
        cond1_pass = n_positive >= n_folds_win_required
        if not cond1_pass:
            failing_conditions.append(
                f"cond1: positive_churn_coefs={n_positive}/{len(churn_coefs)} < required={n_folds_win_required}"
            )

        # Condition 2: improvement_mean > max(sigma_lofo, noise_floor)
        improvement = result["improvement_mean"]
        sigma = result.get("sigma_lofo", 0.0)
        imp_threshold = max(sigma, noise_floor)
        cond2_pass = improvement > imp_threshold
        if not cond2_pass:
            failing_conditions.append(
                f"cond2: improvement_mean={improvement:.6f} <= max(sigma={sigma:.4f}, "
                f"noise_floor={noise_floor})"
            )

        # Condition 3: test-2025 logloss < Elo baseline
        test_metrics = result.get("test_metrics_if_computed")
        cond3_pass = False
        if test_metrics is not None and "logloss" in test_metrics:
            cond3_pass = test_metrics["logloss"] < elo_test_logloss_baseline
        if not cond3_pass:
            test_ll = test_metrics.get("logloss", float("nan")) if test_metrics else float("nan")
            failing_conditions.append(
                f"cond3: test_logloss={test_ll:.6f} >= baseline={elo_test_logloss_baseline:.6f}"
            )

        # Condition 4: churn_coef_global > 0 AND |z| > 1.0
        churn_coef = result.get("churn_coef_global", 0.0)
        z_stat = result.get("z_stat", 0.0)
        cond4_pass = churn_coef > 0 and abs(z_stat) > 1.0
        if not cond4_pass:
            failing_conditions.append(
                f"cond4: churn_coef={churn_coef:.6f}, |z|={abs(z_stat):.4f} — need coef>0 AND |z|>1.0"
            )

        result["failing_conditions"] = failing_conditions

        if cond1_pass and cond2_pass and cond3_pass and cond4_pass:
            return "adopted"
        return "negative"

    # ── B4 path (default, UNTOUCHED — R-DRIFT-1) ──────────────────────────
    # ── REQ-017: Hard shortcut (both conditions required) ────────────────
    if result["w_global"] >= 0.999 and result["improvement_mean"] < 0.001:
        failing_conditions.append("hard_shortcut")
        result["failing_conditions"] = failing_conditions
        return "shortcut_negative"

    # ── REQ-016: AND-of-4 adoption gate ──────────────────────────────────
    # Condition 1: per-fold win count >= n_folds_win_required
    n_wins = sum(
        1
        for i in range(result["n_folds"])
        if result["logloss_per_fold"][i] < result["logloss_elo_only_per_fold"][i]
    )
    cond1_pass = n_wins >= n_folds_win_required
    if not cond1_pass:
        failing_conditions.append(
            f"per_fold_wins={n_wins}/{result['n_folds']} < required={n_folds_win_required}"
        )

    # Condition 2: improvement_mean > max(sigma_lofo, noise_floor)
    improvement = result["improvement_mean"]
    sigma = result.get("sigma_lofo", 0.0)
    imp_threshold = max(sigma, noise_floor)
    cond2_pass = improvement > imp_threshold
    if not cond2_pass:
        failing_conditions.append(
            f"improvement_mean={improvement:.6f} <= max(sigma={sigma:.4f}, "
            f"noise_floor={noise_floor})"
        )

    # Condition 3: test-2025 logloss < Elo baseline
    test_metrics = result.get("test_metrics_if_computed")
    cond3_pass = False
    if test_metrics is not None and "logloss" in test_metrics:
        cond3_pass = test_metrics["logloss"] < elo_test_logloss_baseline
    if not cond3_pass:
        test_ll = test_metrics.get("logloss", float("nan")) if test_metrics else float("nan")
        failing_conditions.append(
            f"test_logloss={test_ll:.6f} >= baseline={elo_test_logloss_baseline:.6f}"
        )

    # Condition 4: w_global > w_sat_threshold (not saturated to derived)
    cond4_pass = result["w_global"] > w_sat_threshold
    if not cond4_pass:
        failing_conditions.append(
            f"w_global={result['w_global']:.4f} <= saturation_threshold={w_sat_threshold}"
        )

    result["failing_conditions"] = failing_conditions

    if cond1_pass and cond2_pass and cond3_pass and cond4_pass:
        return "adopted"
    return "negative"


def run_b4_route(
    *,
    test_year: int = 2025,
    val_years: tuple = (2021, 2022, 2023, 2024),
) -> str:
    """B4 experiment route: LOFO-CV blend of Elo + SetPredictor-derived match prob.

    Steps:
      1. Load sets_partidos.csv and build rolling match features (includes p_elo).
      2. Load set_predictor_v2.joblib (fallback to error).
      3. For each match, construct a SetContext (pre-match, 0-0 set state),
         compute build_set_features → predict_proba → p_set → p_match_from_p_set.
      4. Call optimize_blend_w with the enriched DataFrame.
      5. Apply REQ-017 hard-shortcut and REQ-016 AND-of-4 gate.
      6. Write models/b4_blend_results.json.
      7. Return one of 'adopted', 'negative', 'shortcut_negative'.

    Args:
        test_year: Held-out season (default 2025).
        val_years: Validation folds for LOFO-CV.

    Returns:
        Verdict string.
    """
    import logging

    logger = logging.getLogger("b4_blend")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("  B4 — Match predictor from SetPredictor (best-of-5)")
    logger.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    sp_path = BASE_DIR / "DB" / "sets_partidos.csv"
    if not sp_path.exists():
        raise FileNotFoundError(f"Missing source data: {sp_path}")
    sp = pd.read_csv(sp_path, encoding="utf-8")
    logger.info(f"Loaded {len(sp)} rows from sets_partidos.csv")

    dfm = build_rolling_match_features(sp)
    logger.info(f"Built {len(dfm)} rolling match features")

    # ── Step 2: Load SetPredictor v2 ───────────────────────────────────
    v2_path = MODELS_DIR / "set_predictor_v2.joblib"
    if not v2_path.exists():
        logger.warning("set_predictor_v2.joblib NOT FOUND — falling back to auto-NEGATIVE")
        return _write_negative_fallback(
            "SetPredictor v2 artifact missing from models/",
            test_year,
        )

    v2_artifact = joblib.load(v2_path)
    v2_model = v2_artifact.get("model")
    v2_features = v2_artifact.get("features", SET_FEATURE_COLS)
    if v2_model is None:
        logger.warning("SetPredictor v2 artifact has no 'model' key — auto-NEGATIVE")
        return _write_negative_fallback(
            "SetPredictor v2 model key missing",
            test_year,
        )

    logger.info(f"Loaded SetPredictor v2 ({len(v2_features)} features)")

    # ── Step 3: Compute p_derived per match ────────────────────────────
    rows = []
    for _, row in dfm.iterrows():
        ctx = SetContext(
            temporada_inicio=int(row.get("temporada_inicio", 0)),
            jornada_num=int(row.get("jornada_num", 0)),
            match_id=str(row.get("partido_id", "")),
            set_index=1,
            equipo_local=str(row.get("home", "")),
            equipo_visitante=str(row.get("away", "")),
            elo_local=float(row.get("elo_h", 1500.0)),
            elo_visitante=float(row.get("elo_a", 1500.0)),
            strength_local=float(row.get("strength_h", 0.5)),
            strength_visitante=float(row.get("strength_a", 0.5)),
            h_win_rate_global=float(row.get("h_win_rate_global", 0.5)),
            a_win_rate_global=float(row.get("a_win_rate_global", 0.5)),
            h_set_win_rate=float(row.get("h_set_win_rate", 0.5)),
            a_set_win_rate=float(row.get("a_set_win_rate", 0.5)),
            h_form_ewma=float(row.get("h_form_ewma", 0.5)),
            a_form_ewma=float(row.get("a_form_ewma", 0.5)),
            h_set_diff_exp=float(row.get("h_set_diff_exp", 0.0)),
            a_set_diff_exp=float(row.get("a_set_diff_exp", 0.0)),
            h_point_ratio=float(row.get("h_point_ratio", 0.5)),
            a_point_ratio=float(row.get("a_point_ratio", 0.5)),
            h2h_win_rate=float(row.get("h2h_win_rate_h", 0.5)),
            sets_h_antes=0,
            sets_a_antes=0,
            prev_home_won=-1,
            target_score=25,
        )
        feats = build_set_features(ctx)
        feat_df = pd.DataFrame([feats])[v2_features]
        p_set = float(v2_model.predict_proba(feat_df.fillna(0))[0, 1])
        p_derived = p_match_from_p_set(p_set)

        rows.append(
            {
                "p_elo": float(row.get("elo_win_prob_h", 0.5)),
                "p_derived": p_derived,
                "y": int(row.get("gana_local", 0)),
                "temporada_inicio": int(row.get("temporada_inicio", 0)),
            }
        )

    blend_df = pd.DataFrame(rows)
    logger.info(
        f"Built blend DataFrame: {len(blend_df)} rows, "
        f"{blend_df['temporada_inicio'].nunique()} seasons"
    )

    # ── Step 4: LOFO-CV optimisation ───────────────────────────────────
    from src.models.blend_optimizer import optimize_blend_w

    result = optimize_blend_w(
        blend_df,
        w_grid=list(np.linspace(0.0, 1.0, 21)),
        val_years=list(val_years),
        elo_col="p_elo",
        derived_col="p_derived",
        y_col="y",
        refine="golden_section",
        refine_tol=1e-3,
    )

    # ── Step 5: Compute test-2025 metrics (needed by gate condition 3) ──
    te = blend_df[blend_df["temporada_inicio"] == test_year]
    if len(te) > 10:
        p_blend = (
            result["w_global"] * te["p_elo"].values
            + (1.0 - result["w_global"]) * te["p_derived"].values
        )
        p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
        test_metrics = {
            "n": int(len(te)),
            "logloss": float(log_loss(te["y"].values, p_blend)),
            "auc": float(roc_auc_score(te["y"].values, p_blend)),
            "brier": float(brier_score_loss(te["y"].values, p_blend)),
            "acc": float(accuracy_score(te["y"].values, (p_blend >= 0.5).astype(int))),
        }
    else:
        test_metrics = None

    # ── Step 6: Apply gates (REQ-016 / REQ-017) ─────────────────────────
    # Read Elo test-logloss baseline from precision_improved.json
    baseline_path = MODELS_DIR / "precision_improved.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            precision_data = json.load(f)
        elo_baseline = precision_data.get("match", {}).get("test", {}).get("logloss", float("inf"))
    else:
        elo_baseline = float("inf")

    # Merge test_metrics into result so evaluate_adoption can read them
    result["test_metrics_if_computed"] = test_metrics
    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=elo_baseline,
    )
    failing_conditions = result.get("failing_conditions", [])

    # ── Write results ──────────────────────────────────────────────────
    output = {
        "shortcut": (result["w_global"] >= 0.999 and result["improvement_mean"] < 0.001),
        "verdict": verdict,
        "w_global": result["w_global"],
        "w_per_fold_lofo": result["w_per_fold_lofo"],
        "logloss_per_fold": result["logloss_per_fold"],
        "logloss_elo_only_per_fold": result["logloss_elo_only_per_fold"],
        "logloss_mean": result["logloss_mean"],
        "logloss_elo_only_mean": result["logloss_elo_only_mean"],
        "improvement_mean": result["improvement_mean"],
        "sigma_lofo": result["sigma_lofo"],
        "n_folds": result["n_folds"],
        "test_year": test_year,
        "val_years": list(val_years),
        "test_metrics_if_computed": test_metrics,
        "failing_conditions": failing_conditions,
        "note": (
            "B4 blend experiment: P_final = w * P_elo + (1-w) * P_derived "
            "where P_derived = p_match_from_p_set(SetPredictor.predict_proba)"
        ),
    }

    out_path = MODELS_DIR / "b4_blend_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {out_path}")

    # Summary banner (NFR-005)
    banner = f"  >>> B4 VERDICT: {verdict.upper()}  (w={result['w_global']:.4f}, Δ={result['improvement_mean']:+.4f})"
    logger.info("─" * 60)
    logger.info(banner)
    logger.info("─" * 60)

    return verdict


def run_b5_route(
    *,
    test_year: int = 2025,
    val_years: tuple = (2021, 2022, 2023, 2024),
) -> str:
    """B5 experiment route: LOFO-CV LogReg on [logit(elo_win_prob_h), diff_roster_continuity].

    Steps:
      1. Read Elo baseline from precision_improved.json (FAIL LOUD if missing).
      2. Load match_features.csv (1322 × 69).
      3. Skip rows with imputed roster_continuity (~18% from R-SAMPLE).
      4. Build X = [logit(elo_win_prob_h), diff_roster_continuity], y = gana_local,
         w = recency weight (half-life 2).
      5. LOFO-CV: for each val fold f, fit LogReg(C=0.5) on the OTHER 3 folds,
         evaluate on f.
      6. Per-fold: churn_coef_per_fold[f], logloss_per_fold[f], logloss_elo_only_per_fold[f].
      7. Global fit on all data: churn_coef_global, churn_coef_std_err (manual
         inverse-Hessian, no statsmodels dep — OQ-2).
      8. Apply AND-of-4 gate (REQ-017..REQ-021).
      9. Write models/b5_churn_results.json.
      10. Return verdict.

    Returns:
        Verdict string: ``"adopted"``, ``"negative"``, or ``"shortcut_negative"``.
    """
    logger = logging.getLogger("b5_churn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("  B5 — Roster churn LogReg (AND-of-4 gate)")
    logger.info("=" * 60)

    # ── Step 1: Read Elo baseline (FAIL LOUD if missing — R-ELO-BASELINE) ──
    baseline_path = MODELS_DIR / "precision_improved.json"
    if not baseline_path.exists():
        raise RuntimeError(
            "R-ELO-BASELINE: precision_improved.json not found. "
            "Run `python -m src.models.train_improved` first."
        )
    with open(baseline_path) as f:
        precision_data = json.load(f)
    elo_baseline = precision_data.get("match", {}).get("test", {}).get("logloss")
    if elo_baseline is None:
        raise RuntimeError(
            "R-ELO-BASELINE: match.test.logloss not found in precision_improved.json. "
            "Run `python -m src.models.train_improved` first."
        )
    logger.info(f"Elo test-logloss baseline: {elo_baseline:.6f}")

    # ── Step 2: Load match_features.csv ──────────────────────────────────
    mf_path = BASE_DIR / "DB" / "features" / "match_features.csv"
    if not mf_path.exists():
        raise FileNotFoundError(f"Missing match_features.csv: {mf_path}")
    dfm = pd.read_csv(mf_path)
    # temporada is "YYYY/YYYY" string — convert to int (start year) for recency math.
    dfm["temporada"] = dfm["temporada"].str.split("/").str[0].astype(int)
    logger.info(f"Loaded {len(dfm)} rows from match_features.csv ({dfm.shape[1]} cols)")

    # ── Step 3: Skip imputed rows (REQ-016) ──────────────────────────────
    # Roster continuity is imputed for first-season teams (~18% of rows).
    # We detect imputed rows by checking if the continuity value equals the
    # league median for that season.  We compute the median per season.
    n_total = len(dfm)
    for side in ("h_roster_continuity", "a_roster_continuity"):
        season_medians = dfm.groupby("temporada")[side].transform("median")
        dfm[f"{side}_is_imputed"] = np.abs(dfm[side] - season_medians) < 1e-6
    dfm["any_imputed"] = (
        dfm["h_roster_continuity_is_imputed"] | dfm["a_roster_continuity_is_imputed"]
    )
    n_imputed = dfm["any_imputed"].sum()
    df_clean = dfm[~dfm["any_imputed"]].copy()
    n_skipped = n_imputed
    logger.info(
        f"Imputed rows: {n_imputed}/{n_total} ({100*n_imputed/n_total:.1f}%) — skipped for primary analysis"
    )

    # ── Step 4: Build X, y, w ────────────────────────────────────────────
    from scipy.special import logit

    df_clean["logit_elo"] = logit(np.clip(df_clean["elo_win_prob_h"].values, 1e-6, 1 - 1e-6))
    feature_cols = ["logit_elo", "diff_roster_continuity"]
    X = df_clean[feature_cols].fillna(0).values
    y = df_clean["gana_local"].values

    # Recency weights: half-life 2 seasons
    current_max_season = df_clean["temporada"].max()
    df_clean["recency_w"] = 0.5 ** ((current_max_season - df_clean["temporada"].values) / 2.0)
    w = df_clean["recency_w"].values

    logger.info(f"Primary analysis: {len(df_clean)} rows (skipped {n_skipped} imputed)")

    # ── Step 5: LOFO-CV ──────────────────────────────────────────────────
    val_year_list = list(val_years)
    churn_coef_per_fold = []
    logloss_per_fold = []
    logloss_elo_only_per_fold = []

    for vy in val_year_list:
        tr_mask = df_clean["temporada"] != vy
        va_mask = df_clean["temporada"] == vy
        X_tr, y_tr, w_tr = X[tr_mask.values], y[tr_mask.values], w[tr_mask.values]
        X_va, y_va = X[va_mask.values], y[va_mask.values]

        if len(X_va) < 5 or len(np.unique(y_va)) < 2:
            logger.warning(f"  fold {vy}: too few rows or single class, skipping")
            continue

        model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        p_va = np.clip(p_va, 1e-6, 1 - 1e-6)

        # Elo-only baseline on this fold
        logit_elo_va = X_va[:, 0]
        p_elo_va = 1.0 / (1.0 + np.exp(-logit_elo_va))
        p_elo_va = np.clip(p_elo_va, 1e-6, 1 - 1e-6)

        churn_coef = float(model.coef_[0, 1])
        churn_coef_per_fold.append(churn_coef)
        logloss_per_fold.append(float(log_loss(y_va, p_va)))
        logloss_elo_only_per_fold.append(float(log_loss(y_va, p_elo_va)))

        logger.info(
            f"  fold {vy}: churn_coef={churn_coef:+.6f}  "
            f"logloss={logloss_per_fold[-1]:.4f}  "
            f"elo={logloss_elo_only_per_fold[-1]:.4f}  "
            f"improvement={logloss_elo_only_per_fold[-1] - logloss_per_fold[-1]:+.4f}"
        )

    n_folds = len(churn_coef_per_fold)
    if n_folds == 0:
        logger.error("No valid folds — writing NEGATIVE fallback")
        return _write_b5_fallback("No valid LOFO folds", test_year)

    improvements = [e - b for e, b in zip(logloss_elo_only_per_fold, logloss_per_fold)]
    improvement_mean = float(np.mean(improvements))
    sigma_lofo = float(np.std(improvements)) if len(improvements) > 1 else 0.0
    sigma_lofo = max(sigma_lofo, 0.005)

    # ── Step 7: Global fit on all data ────────────────────────────────────
    model_all = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    model_all.fit(X, y, sample_weight=w)
    churn_coef_global = float(model_all.coef_[0, 1])

    # Manual inverse-Hessian for std_err (no statsmodels dep — OQ-2)
    p_all = model_all.predict_proba(X)[:, 1]
    p_all = np.clip(p_all, 1e-6, 1 - 1e-6)
    # Hessian = X^T diag(p*(1-p)) X  (for log-loss, each obs contributes p*(1-p) * x_i x_i^T)
    diag_w = (p_all * (1 - p_all)) * w  # recency-weighted
    Xw = X * diag_w[:, np.newaxis]
    hessian = Xw.T @ X
    try:
        cov = np.linalg.inv(hessian)
        churn_coef_std_err = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        logger.warning("Hessian singular — falling back to pseudo-inverse for std_err")
        cov = np.linalg.pinv(hessian)
        churn_coef_std_err = float(np.sqrt(cov[1, 1]))
    z_stat = churn_coef_global / churn_coef_std_err if churn_coef_std_err > 0 else 0.0

    # ── Step 7b: Train-only model for HELD-OUT test-2025 evaluation (W1 fix) ─
    # The global `model_all` trained on ALL non-imputed rows (including test_year)
    # produced optimistic test_metrics. We re-fit on train-only rows to get a
    # held-out evaluation (ml-calibration skill: never fit+eval on same data).
    tr_only_mask = df_clean["temporada"] != test_year
    X_tr_only = X[tr_only_mask.values]
    y_tr_only = y[tr_only_mask.values]
    w_tr_only = w[tr_only_mask.values]
    model_test = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    model_test.fit(X_tr_only, y_tr_only, sample_weight=w_tr_only)

    # ── Step 8: Compute test-2025 metrics (HELD-OUT via model_test) ───────
    te_mask = df_clean["temporada"] == test_year
    if te_mask.sum() > 10:
        X_te = X[te_mask.values]
        y_te = y[te_mask.values]
        p_te = model_test.predict_proba(X_te)[:, 1]
        p_te = np.clip(p_te, 1e-6, 1 - 1e-6)
        p_elo_te = 1.0 / (1.0 + np.exp(-X_te[:, 0]))
        p_elo_te = np.clip(p_elo_te, 1e-6, 1 - 1e-6)
        # In-sample for comparison (model_all on test_year — optimistic, not used
        # in the gate). Kept for transparency only.
        p_te_in_sample = model_all.predict_proba(X_te)[:, 1]
        p_te_in_sample = np.clip(p_te_in_sample, 1e-6, 1 - 1e-6)
        test_metrics = {
            "n": int(len(y_te)),
            "logloss": float(log_loss(y_te, p_te)),
            "logloss_elo": float(log_loss(y_te, p_elo_te)),
            "auc": float(roc_auc_score(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
            "acc": float(accuracy_score(y_te, (p_te >= 0.5).astype(int))),
            "logloss_in_sample": float(log_loss(y_te, p_te_in_sample)),
            "evaluation_mode": "held_out",
        }
    else:
        test_metrics = None

    # ── Step 9: Apply AND-of-4 gate ──────────────────────────────────────
    result = {
        "n_folds": n_folds,
        "churn_coef_per_fold": churn_coef_per_fold,
        "logloss_per_fold": logloss_per_fold,
        "logloss_elo_only_per_fold": logloss_elo_only_per_fold,
        "logloss_mean": float(np.mean(logloss_per_fold)) if logloss_per_fold else float("nan"),
        "logloss_elo_only_mean": (
            float(np.mean(logloss_elo_only_per_fold)) if logloss_elo_only_per_fold else float("nan")
        ),
        "improvement_mean": improvement_mean,
        "sigma_lofo": sigma_lofo,
        "churn_coef_global": churn_coef_global,
        "churn_coef_std_err": churn_coef_std_err,
        "z_stat": z_stat,
        "test_metrics_if_computed": test_metrics,
    }

    verdict = evaluate_adoption(
        result,
        elo_test_logloss_baseline=elo_baseline,
        gate="b5",
    )
    failing_conditions = result.get("failing_conditions", [])

    # ── Sensitivity check: re-run with imputed rows INCLUDED ────────────
    df_sens = dfm.copy()
    df_sens["logit_elo"] = np.log(
        np.clip(df_sens["elo_win_prob_h"].values, 1e-6, 1 - 1e-6)
        / (1 - np.clip(df_sens["elo_win_prob_h"].values, 1e-6, 1 - 1e-6))
    )
    X_sens = df_sens[feature_cols].fillna(0).values
    y_sens = df_sens["gana_local"].values
    w_sens = 0.5 ** ((current_max_season - df_sens["temporada"].values) / 2.0)

    churn_coef_per_fold_sens = []
    logloss_per_fold_sens = []
    logloss_elo_only_per_fold_sens = []
    for vy in val_year_list:
        tr_mask = df_sens["temporada"] != vy
        va_mask = df_sens["temporada"] == vy
        X_tr_s, y_tr_s, w_tr_s = (
            X_sens[tr_mask.values],
            y_sens[tr_mask.values],
            w_sens[tr_mask.values],
        )
        X_va_s, y_va_s = X_sens[va_mask.values], y_sens[va_mask.values]
        if len(X_va_s) < 5 or len(np.unique(y_va_s)) < 2:
            continue
        m_s = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        m_s.fit(X_tr_s, y_tr_s, sample_weight=w_tr_s)
        p_s = m_s.predict_proba(X_va_s)[:, 1]
        p_s = np.clip(p_s, 1e-6, 1 - 1e-6)
        p_elo_s = 1.0 / (1.0 + np.exp(-X_va_s[:, 0]))
        p_elo_s = np.clip(p_elo_s, 1e-6, 1 - 1e-6)
        churn_coef_per_fold_sens.append(float(m_s.coef_[0, 1]))
        logloss_per_fold_sens.append(float(log_loss(y_va_s, p_s)))
        logloss_elo_only_per_fold_sens.append(float(log_loss(y_va_s, p_elo_s)))

    n_folds_sens = len(churn_coef_per_fold_sens)
    if n_folds_sens > 0:
        # Global fit on sensitivity data
        model_sens = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        model_sens.fit(X_sens, y_sens, sample_weight=w_sens)
        churn_coef_sens = float(model_sens.coef_[0, 1])
        p_sens = model_sens.predict_proba(X_sens)[:, 1]
        p_sens = np.clip(p_sens, 1e-6, 1 - 1e-6)
        diag_w_sens = (p_sens * (1 - p_sens)) * w_sens
        hessian_sens = (X_sens * diag_w_sens[:, np.newaxis]).T @ X_sens
        try:
            cov_sens = np.linalg.inv(hessian_sens)
            se_sens = float(np.sqrt(cov_sens[1, 1]))
        except np.linalg.LinAlgError:
            se_sens = 0.0
        z_sens = churn_coef_sens / se_sens if se_sens > 0 else 0.0

        # Sensitivity verdict
        sens_improvements = [
            e - b for e, b in zip(logloss_elo_only_per_fold_sens, logloss_per_fold_sens)
        ]
        sens_improvement_mean = float(np.mean(sens_improvements))
        sens_sigma = float(np.std(sens_improvements)) if len(sens_improvements) > 1 else 0.0
        sens_sigma = max(sens_sigma, 0.005)
        sens_result = {
            "n_folds": n_folds_sens,
            "churn_coef_per_fold": churn_coef_per_fold_sens,
            "churn_coef_global": churn_coef_sens,
            "churn_coef_std_err": se_sens,
            "z_stat": z_sens,
            "logloss_per_fold": logloss_per_fold_sens,
            "logloss_elo_only_per_fold": logloss_elo_only_per_fold_sens,
            "logloss_mean": (
                float(np.mean(logloss_per_fold_sens)) if logloss_per_fold_sens else float("nan")
            ),
            "improvement_mean": sens_improvement_mean,
            "sigma_lofo": sens_sigma,
            "test_metrics_if_computed": test_metrics,
        }
        sensitivity_verdict = evaluate_adoption(
            sens_result,
            elo_test_logloss_baseline=elo_baseline,
            gate="b5",
        )
    else:
        sensitivity_verdict = "no_folds"

    # ── Write results ────────────────────────────────────────────────────
    output = {
        "shortcut": (abs(churn_coef_global) < 1e-6 and abs(improvement_mean) < 1e-9),
        "verdict": verdict,
        "churn_coef_global": churn_coef_global,
        "churn_coef_std_err": churn_coef_std_err,
        "z_stat": z_stat,
        "churn_coef_per_fold": churn_coef_per_fold,
        "logloss_per_fold": logloss_per_fold,
        "logloss_elo_only_per_fold": logloss_elo_only_per_fold,
        "logloss_mean": float(np.mean(logloss_per_fold)) if logloss_per_fold else float("nan"),
        "logloss_elo_only_mean": (
            float(np.mean(logloss_elo_only_per_fold)) if logloss_elo_only_per_fold else float("nan")
        ),
        "improvement_mean": improvement_mean,
        "sigma_lofo": sigma_lofo,
        "n_folds": n_folds,
        "n_imputed_skipped": int(n_skipped),
        "sample_population": int(len(df_clean)),
        "elo_baseline_logloss": float(elo_baseline),
        "test_year": test_year,
        "val_years": list(val_years),
        "test_metrics": test_metrics,
        "test_metrics_if_computed": test_metrics,
        "failing_conditions": failing_conditions,
        "sensitivity_verdict": sensitivity_verdict,
        "note": (
            "B5 churn experiment: LogReg(C=0.5) on [logit(elo_win_prob_h), diff_roster_continuity] "
            "with recency weight (half-life=2). Imputed rows skipped for primary; "
            "sensitivity includes them."
        ),
    }

    out_path = MODELS_DIR / "b5_churn_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {out_path}")

    # Summary banner (NFR-005)
    banner = (
        f"  >>> B5 VERDICT: {verdict.upper()}  "
        f"(churn_coef={churn_coef_global:+.6f}, Δ={improvement_mean:+.4f})"
    )
    logger.info("─" * 60)
    logger.info(banner)
    if verdict == "negative":
        logger.info(f"  Failing conditions: {failing_conditions}")
    logger.info("─" * 60)

    return verdict


def _write_b5_fallback(reason: str, test_year: int) -> str:
    """Write a degenerate B5 result when data is unavailable."""
    out_path = MODELS_DIR / "b5_churn_results.json"
    output = {
        "shortcut": True,
        "verdict": "shortcut_negative",
        "churn_coef_global": 0.0,
        "churn_coef_std_err": 0.0,
        "z_stat": 0.0,
        "churn_coef_per_fold": [],
        "logloss_per_fold": [],
        "logloss_elo_only_per_fold": [],
        "logloss_mean": None,
        "logloss_elo_only_mean": None,
        "improvement_mean": None,
        "sigma_lofo": 0.0,
        "n_folds": 0,
        "n_imputed_skipped": 0,
        "test_year": test_year,
        "val_years": [],
        "test_metrics_if_computed": None,
        "failing_conditions": [f"Data unavailable: {reason}"],
        "sensitivity_verdict": "no_folds",
        "note": ("B5 fallback — data unavailable; verdict is shortcut_negative."),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    b5_logger = logging.getLogger("b5_churn")
    b5_logger.info(f"Fallback results written to {out_path}")
    b5_logger.info("  >>> B5 VERDICT: SHORTCUT_NEGATIVE  (data unavailable)")
    return "shortcut_negative"


def _write_negative_fallback(reason: str, test_year: int) -> str:
    """Write a degenerate B4 result when SetPredictor v2 is unavailable.

    R-DRIFT-1: the B4 call sites in ``run_b4_route`` still reference this
    function name; the implementation writes ``b4_blend_results.json`` with
    ``w_global=1.0`` (pure Elo) when SetPredictor v2 is missing.
    """
    out_path = MODELS_DIR / "b4_blend_results.json"
    output = {
        "shortcut": True,
        "verdict": "shortcut_negative",
        "w_global": 1.0,
        "w_per_fold_lofo": [],
        "logloss_per_fold": [],
        "logloss_elo_only_per_fold": [],
        "logloss_mean": None,
        "logloss_elo_only_mean": None,
        "improvement_mean": None,
        "sigma_lofo": 0.0,
        "n_folds": 0,
        "test_year": test_year,
        "val_years": [],
        "test_metrics_if_computed": None,
        "failing_conditions": [f"SetPredictor unavailable: {reason}"],
        "note": (
            "B4 fallback — SetPredictor v2 not loaded; blend degenerates " "to w=1.0 (pure Elo)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    b4_logger.info(f"Fallback results written to {out_path}")
    b4_logger.info("  >>> B4 VERDICT: SHORTCUT_NEGATIVE  (w=1.0, SetPredictor missing)")
    return "shortcut_negative"


def main():
    print("=" * 70)
    print("  ENTRENAMIENTO MEJORADO — resultado del plan de precisión")
    print("=" * 70)
    data = run_pipeline()
    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    dfm = build_rolling_match_features(sp)

    print("\n[MATCH] Elo con margen (rolling, sin leakage)...")
    m_match = train_match(dfm)
    print(
        f"  TEST {TEST_SEASON}: AUC={m_match['test']['auc']:.4f} "
        f"logloss={m_match['test']['logloss']:.4f} brier={m_match['test']['brier']:.4f} "
        f"acc={m_match['test']['acc']:.4f} (n={m_match['test']['n_test']})"
    )
    print(f"  CV: {m_match['cv']['n_folds']} folds (sin entrenamiento)")

    print("\n[SET] LogReg con recencia...")
    m_set = train_set(data)
    print(
        f"  TEST {TEST_SEASON}: AUC={m_set['test']['auc']:.4f} "
        f"logloss={m_set['test']['logloss']:.4f} brier={m_set['test']['brier']:.4f} "
        f"acc={m_set['test']['acc']:.4f} (n={m_set['test']['n_test']})"
    )
    cv = m_set["cv"]
    print(
        f"  CV ({cv['n_folds']} folds): logloss={cv.get('logloss_mean', 'N/A'):.4f}±{cv.get('logloss_std', 0):.4f} "
        f"auc={cv.get('auc_mean', 'N/A'):.4f}±{cv.get('auc_std', 0):.4f}"
    )

    snapshot = {
        "match": m_match,
        "set": m_set,
        "config": {
            "match_features": MATCH_FEATURES,
            "train_seasons": RECENT_TRAIN_SEASONS,
            "test_season": TEST_SEASON,
        },
    }
    out = MODELS_DIR / "precision_improved.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\n  Snapshot guardado en {out}")
    print("  Artefactos: match_elo_v2.joblib, set_predictor_v2.joblib")


if __name__ == "__main__":
    main()
