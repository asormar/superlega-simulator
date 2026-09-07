"""Cross-temporal robustness check for the production margin-Elo model.

Goal: verify the test-time precision metrics published in
``models/precision_improved.json`` (logloss=0.5677, AUC=0.762 on 2025) are
stable when the SAME production model is evaluated on a different held-out
test year (2024, 2023, 2022). If the numbers are similar (±0.02 logloss,
±0.03 AUC), the precision claim is robust; if they diverge wildly, the
2025 metric may have been over-fit / over-sampled.

This is purely an evaluation script. It does NOT retrain anything. It reads
the pre-computed ``DB/features/match_features.csv`` (which contains
``elo_win_prob_h`` per match, the production signal) and computes standard
binary classification metrics on each non-train test year.

Usage
-----
    python -m src.models.cross_temporal_check
    python -m src.models.cross_temporal_check --years 2024 2023
    python -m src.models.cross_temporal_check --save

Outputs a Markdown table to stdout (and ``models/report/cross_temporal.md``
with --save) so it can be quoted in the memoria or appendix without
re-deriving the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ── Boilerplate so this works as `python -m src.models.cross_temporal_check`
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
MF_PATH = BASE_DIR / "DB" / "features" / "match_features.csv"
PRECISION_PATH = MODELS_DIR / "precision_improved.json"
REPORT_DIR = MODELS_DIR / "report"


def _safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Log loss with eps-clipped probabilities to avoid log(0)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, p))


def _eval_year(df_year: pd.DataFrame) -> dict:
    """Compute classification metrics for a year subset using ``elo_win_prob_h``."""
    if len(df_year) < 5:
        return {
            "n": int(len(df_year)),
            "logloss": float("nan"),
            "auc": float("nan"),
            "brier": float("nan"),
            "acc": float("nan"),
        }
    y = df_year["gana_local"].astype(int).values
    p = df_year["elo_win_prob_h"].astype(float).values
    return {
        "n": int(len(y)),
        "logloss": _safe_log_loss(y, p),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def cross_temporal_report(years: list[int]) -> pd.DataFrame:
    """Compute precision metrics per test year and return a summary table.

    The 2025 numbers from ``precision_improved.json`` are added as a reference
    column (they were computed on the same data via the production pipeline).
    """
    if not MF_PATH.exists():
        raise FileNotFoundError(f"Missing {MF_PATH}; run python -m src.data.feature_store first.")
    dfm = pd.read_csv(MF_PATH)
    if "elo_win_prob_h" not in dfm.columns:
        raise RuntimeError(
            "elo_win_prob_h not in match_features.csv — the production signal is missing."
        )

    # The published baseline (precision_improved.json) is the canonical 2025 number.
    precision = (
        json.loads(PRECISION_PATH.read_text(encoding="utf-8")) if PRECISION_PATH.exists() else {}
    )
    published_2025 = precision.get("match", {}).get("test", {})

    rows: list[dict] = []
    # temporada is 'YYYY/YYYY' string — convert to start-year int for matching.
    temporada_int = dfm["temporada"].astype(str).str.split("/").str[0].astype(int)
    dfm = dfm.assign(_temporada_int=temporada_int)
    for year in sorted(years):
        m = _eval_year(dfm[dfm["_temporada_int"] == year])
        m["year"] = int(year)
        m["source"] = "cross_temporal_check" if year != 2025 else "precision_improved.json"
        # Replace the 2025 row with the published baseline for direct comparison.
        if year == 2025 and published_2025:
            m.update(
                {
                    "n": int(published_2025.get("n_test", m["n"])),
                    "logloss": float(published_2025.get("logloss", m["logloss"])),
                    "auc": float(published_2025.get("auc", m["auc"])),
                    "brier": float(published_2025.get("brier", m["brier"])),
                    "acc": float(published_2025.get("acc", m["acc"])),
                }
            )
        rows.append(m)

    return pd.DataFrame(rows).set_index("year").sort_index()


def _to_markdown(df: pd.DataFrame) -> str:
    """Render the report as a Markdown table (suitable for the memoria)."""
    lines = [
        "# Cross-Temporal Robustness Check",
        "",
        "Production margin-Elo evaluated on each held-out year. The 2025 row is the",
        "canonical number from `models/precision_improved.json`; the other rows are",
        "computed by `python -m src.models.cross_temporal_check` on the same production",
        "Elo signal from `DB/features/match_features.csv`.",
        "",
        "| Year | n | logloss | AUC | Brier | accuracy | source |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for year, row in df.iterrows():
        lines.append(
            f"| {year} | {int(row['n'])} | {row['logloss']:.4f} | "
            f"{row['auc']:.4f} | {row['brier']:.4f} | {row['acc']:.4f} | {row['source']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-temporal robustness check for the production margin-Elo.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Test years to evaluate. Default: 2022 2023 2024 2025.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also write the report to models/report/cross_temporal.md.",
    )
    args = parser.parse_args()

    df = cross_temporal_report(args.years)
    md = _to_markdown(df)
    print(md)
    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / "cross_temporal.md"
        out.write_text(md, encoding="utf-8")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
