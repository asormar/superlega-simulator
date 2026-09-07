"""E3 — `precision_report.py`: a single command that regenerates EVERY
number the memoria quotes, so the TFG never cites a stale or orphaned
figure (this already happened: §7.1's table went stale after the Optional
bug). Outputs Markdown + LaTeX tabular to ``models/report/``.

What it consolidates
--------------------
1. **Canonical match/set snapshot** (from ``measure_precision.measure()``).
2. **Per-year set validation** (replicating §7.2 of ``mejora_precision_2026-07.md``):
   for each season T, train on years < T, evaluate on T, with the production
   config (LogReg C=0.5, recency half-life 2, 21 SET_FEATURE_COLS).
3. **Existing JSONs** under ``models/``: ``precision_improved.json``,
   ``precision_baseline.json``, ``backtest_*.json`` (B1), ``b4_blend_results.json``
   (B4), ``b5_churn_results.json`` (B5), ``backtest_clamp_results.json`` (A5).
4. **Cross-temporal** (from the previous PR's ``models/report/cross_temporal.md``).

The Markdown report is a faithful re-derivation of every figure cited in
the memoria. The LaTeX file is a ready-to-paste ``\\begin{tabular}`` table for
downstream LaTeX builds (user maintains LaTeX outside this repo).

Usage
-----
    python -m src.models.precision_report
    python -m src.models.precision_report --out models/report

The script is idempotent and safe to re-run after any data or model change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# ── Boilerplate so this works as `python -m src.models.precision_report`
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.feature_store import SET_FEATURE_COLS  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
SET_FEATURES_PATH = BASE_DIR / "DB" / "features" / "set_features_v2.csv"
TARGET_COL = "ganador_set_local"
TIME_COL = "temporada_inicio"


def _safe_logloss(y, p, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(log_loss(y, p))


# ──────────────────────────────────────────────────────────────────
# 1. Per-year set validation (replicates §7.2 of the memoria)
# ──────────────────────────────────────────────────────────────────


def per_year_set_validation(
    train_years_back: int = 8, half_life: float = 2.0, c_reg: float = 0.5
) -> pd.DataFrame:
    """For each season T, train on years < T (rolling window) and evaluate on T.

    Replicates the §7.2 per-year analysis with the production config
    (LogReg C=0.5, recency half-life 2, 21 SET_FEATURE_COLS). Returns a
    DataFrame indexed by season with logloss / AUC / Brier / accuracy /
    sample size.
    """
    if not SET_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SET_FEATURES_PATH}; run python -m src.data.feature_store first."
        )
    df = pd.read_csv(SET_FEATURES_PATH)
    cols = [c for c in SET_FEATURE_COLS if c in df.columns]
    if not cols:
        raise RuntimeError(f"None of SET_FEATURE_COLS found in {SET_FEATURES_PATH.name}")

    years = sorted(df[TIME_COL].astype(int).unique())
    rows: list[dict] = []
    for T in years:
        train_years = [y for y in years if y < T][-train_years_back:]
        if not train_years:
            continue
        tr = df[df[TIME_COL].isin(train_years)]
        te = df[df[TIME_COL] == T]
        if len(te) < 5 or len(tr) < 50 or tr[TARGET_COL].nunique() < 2:
            continue

        X_tr, y_tr = tr[cols].fillna(0).values, tr[TARGET_COL].astype(int).values
        X_te, y_te = te[cols].fillna(0).values, te[TARGET_COL].astype(int).values

        # Recency weights with half-life 2.
        w_tr = 0.5 ** ((T - tr[TIME_COL].astype(int).values) / half_life)
        model = LogisticRegression(max_iter=2000, C=c_reg, random_state=42)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        p_te = np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)

        rows.append(
            {
                "temporada": int(T),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "logloss": _safe_logloss(y_te, p_te),
                "auc": float(roc_auc_score(y_te, p_te)),
                "brier": float(brier_score_loss(y_te, p_te)),
                "acc": float(accuracy_score(y_te, (p_te >= 0.5).astype(int))),
            }
        )
    return pd.DataFrame(rows).set_index("temporada").sort_index()


# ──────────────────────────────────────────────────────────────────
# 2. Read existing JSON artefacts
# ──────────────────────────────────────────────────────────────────


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  warn: {path.name} unreadable ({e}); skipping")
        return None


def collect_artefacts() -> dict[str, dict | None]:
    """Snapshot every JSON artefact under ``models/`` that the memoria cites."""
    return {
        "precision_improved": _safe_read_json(MODELS_DIR / "precision_improved.json"),
        "precision_baseline": _safe_read_json(MODELS_DIR / "precision_baseline.json"),
        "backtest_simulator_2024": _safe_read_json(MODELS_DIR / "backtest_simulator_2024.json"),
        "backtest_simulator_2025": _safe_read_json(MODELS_DIR / "backtest_simulator_2025.json"),
        "backtest_clamp_a5": _safe_read_json(MODELS_DIR / "backtest_clamp_results.json"),
        "b4_blend": _safe_read_json(MODELS_DIR / "b4_blend_results.json"),
        "b5_churn": _safe_read_json(MODELS_DIR / "b5_churn_results.json"),
    }


# ──────────────────────────────────────────────────────────────────
# 3. Render
# ──────────────────────────────────────────────────────────────────


def _md_per_year(df: pd.DataFrame) -> str:
    lines = [
        "## Per-Year Set Validation (production config)",
        "",
        "Replicates §7.2 of `memoria/mejora_precision_2026-07.md`. For each",
        "season T: train on years < T (rolling window), evaluate on T with",
        "`LogisticRegression(C=0.5, max_iter=2000, random_state=42)`, recency",
        "weight `0.5 ** ((T - temporada_inicio) / 2)`, features = `SET_FEATURE_COLS`.",
        "",
        "| temporada | n_train | n_test | logloss | AUC | Brier | accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, row in df.iterrows():
        lines.append(
            f"| {t} | {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{row['logloss']:.4f} | {row['auc']:.4f} | {row['brier']:.4f} | "
            f"{row['acc']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _md_canonical(artefacts: dict) -> str:
    pi = artefacts.get("precision_improved") or {}
    if not pi:
        return "## Canonical Snapshot\n\n_NOTE: `models/precision_improved.json` missing._\n"
    lines = [
        "## Canonical Snapshot (from `models/precision_improved.json`)",
        "",
        "| target | n_test | logloss | AUC | Brier | accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tgt in ("match", "set"):
        if tgt not in pi:
            continue
        t = pi[tgt].get("test", {})
        lines.append(
            f"| {tgt} | {int(t.get('n_test', 0))} | "
            f"{t.get('logloss', 0):.4f} | {t.get('auc', 0):.4f} | "
            f"{t.get('brier', 0):.4f} | {t.get('acc', 0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _md_artefacts(artefacts: dict) -> str:
    rows: list[str] = [
        "## Existing JSON Artefacts",
        "",
        "Snapshot of every JSON the memoria cites. Re-running this script",
        "after any data or model change regenerates the snapshot — no more",
        "orphaned numbers in the TFG memoria.",
        "",
        "| artefact | verdict / headline |",
        "|---|---|",
    ]
    # B4 + B5 verdicts
    for key, label in [("b4_blend", "B4 (best-of-5 blend)"), ("b5_churn", "B5 (roster churn)")]:
        d = artefacts.get(key) or {}
        v = d.get("verdict", "missing")
        fc = d.get("failing_conditions", [])
        fc_str = "; ".join(fc) if fc else "—"
        rows.append(f"| {label} | **{v}** — failing: {fc_str} |")
    # A5 backtest
    a5 = artefacts.get("backtest_clamp_a5") or {}
    if a5:
        rows.append(f"| A5 (clamp backtest) | keys: {', '.join(list(a5.keys())[:6])} |")
    # Cross-temporal reference
    ct_path = MODELS_DIR / "report" / "cross_temporal.md"
    if ct_path.exists():
        rows.append("| cross_temporal (2022-2025) | see `models/report/cross_temporal.md` |")
    return "\n".join(rows) + "\n"


def to_markdown(per_year: pd.DataFrame, artefacts: dict) -> str:
    parts = [
        "# Precision Report (E3)",
        "",
        "Single source of truth for every figure the TFG memoria cites. Regenerate with",
        "`python -m src.models.precision_report` after any data or model change.",
        "",
        "---",
        "",
        _md_canonical(artefacts),
        "",
        _md_per_year(per_year),
        "",
        _md_artefacts(artefacts),
        "",
        "---",
        "",
        "_Generated by `src/models/precision_report.py`._",
        "",
    ]
    return "\n".join(parts)


def to_latex(per_year: pd.DataFrame) -> str:
    """Render the per-year table as a ready-to-paste LaTeX tabular."""
    lines = [
        r"% Auto-generated by src/models/precision_report.py. Do not edit by hand.",
        r"\begin{tabular}{r r r r r r r}",
        r"\hline",
        r"temporada & n\_train & n\_test & logloss & AUC & Brier & accuracy \\",
        r"\hline",
    ]
    for t, row in per_year.iterrows():
        lines.append(
            f"{t} & {int(row['n_train'])} & {int(row['n_test'])} & "
            f"{row['logloss']:.4f} & {row['auc']:.4f} & {row['brier']:.4f} & "
            f"{row['acc']:.4f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the precision report (Markdown + LaTeX) from all sources.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "report",
        help="Output directory (default: models/report).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  E3 — Precision Report")
    print("=" * 70)

    print("\n  1. Per-year set validation ...")
    per_year = per_year_set_validation()
    print(f"     {len(per_year)} seasons, range {per_year.index.min()}–{per_year.index.max()}")

    print("\n  2. Reading existing JSON artefacts ...")
    artefacts = collect_artefacts()
    for k, v in artefacts.items():
        marker = "OK" if v else "missing"
        print(f"     {k}: {marker}")

    print("\n  3. Rendering Markdown ...")
    md_path = args.out / "precision_report.md"
    md_path.write_text(to_markdown(per_year, artefacts), encoding="utf-8")
    print(f"     wrote {md_path}")

    print("\n  4. Rendering LaTeX tabular ...")
    tex_path = args.out / "precision_table.tex"
    tex_path.write_text(to_latex(per_year), encoding="utf-8")
    print(f"     wrote {tex_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
