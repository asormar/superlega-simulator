"""
adaptive_damping_experiment.py — A/B test: fixed vs adaptive damping (Batch 3 mid-effort #3).

The MatchPredictor's shrinkage toward the neutral prior (0.5) is controlled
by a single `damping` parameter. Currently it's fixed at 0.5 (MATCH_PREDICTOR_DAMPING).
memoria/match_predictor.md §8.3 notes that this could be adaptive: higher
damping (more shrinkage) early in the season when features are cold, lower
damping (more trust in the model) late in the season when features are warm.

This script runs the same season simulation under two damping strategies
and compares the resulting distribution of 3-0 / 3-1 / 3-2 sets.

Method
------
- Same SuperLega teams (12 from the current season), same round-robin schedule.
- Same seed for both runs (so the schedule and team strengths match).
- 100 MC runs per strategy, deterministic schedule.
- Metric: % of sets that end 3-0, 3-1, 3-2. Lower 3-0% with higher 3-2% means
  the simulation is more "chaotic" / less deterministic. The docs claim
  real SuperLega has ~40% 3-0 (target).

Output
------
- Models saved to models/adaptive_damping_results.json
- Includes fixed and adaptive distributions + delta.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.team_mapper import get_superliga_teams
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.constants import (
    MATCH_PREDICTOR_DAMPING,
    adaptive_damping,
    ADAPTIVE_DAMPING_START,
    ADAPTIVE_DAMPING_END,
    SUPERLEGA_TOTAL_JORNADAS,
)

MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "adaptive_damping_results.json"

CURRENT_SUPERLEGA = get_superliga_teams("2024/2025")


def _run_one_strategy(damping, seed: int) -> Counter:
    """Run one deterministic season with the given damping and return set-score counts."""
    sim = SeasonSimulator()
    sim_result = sim.simulate_season(
        teams=CURRENT_SUPERLEGA,
        double_round_robin=True,
        seed=seed,
        damping=damping,
    )
    counts = Counter()
    for m in sim_result["matches"]:
        s = sorted([m.sets_home, m.sets_away], reverse=True)
        key = f"{s[0]}-{s[1]}"
        counts[key] += 1
    return counts


def _distribution(counts: Counter) -> dict:
    """Convert counts to % distribution of 3-0, 3-1, 3-2."""
    total = sum(counts.values())
    if total == 0:
        return {"3-0%": 0.0, "3-1%": 0.0, "3-2%": 0.0, "n": 0}
    return {
        "3-0%": round(100 * counts.get("3-0", 0) / total, 2),
        "3-1%": round(100 * counts.get("3-1", 0) / total, 2),
        "3-2%": round(100 * counts.get("3-2", 0) / total, 2),
        "n": total,
    }


def run_experiment(n_mc: int = 100) -> dict:
    """
    Run the A/B experiment: fixed damping 0.5 vs adaptive (0.7 -> 0.3 linear).

    Args:
        n_mc: number of Monte Carlo runs per strategy.
    """
    print("=" * 70)
    print(
        f"  ADAPTIVE DAMPING EXPERIMENT — fixed 0.5 vs adaptive "
        f"({ADAPTIVE_DAMPING_START}->{ADAPTIVE_DAMPING_END})"
    )
    print("=" * 70)
    print(f"  Teams: {len(CURRENT_SUPERLEGA)} ({', '.join(CURRENT_SUPERLEGA[:4])}...)")
    print(f"  Total jornadas: {SUPERLEGA_TOTAL_JORNADAS}")
    print(f"  MC runs per strategy: {n_mc}")
    print()

    # Aggregate counts across MC runs
    def aggregate(damping):
        agg = Counter()
        t0 = time.time()
        for seed in range(n_mc):
            agg.update(_run_one_strategy(damping, seed))
        return agg, time.time() - t0

    print("[1/3] Running fixed damping (0.5)...")
    fixed_counts, fixed_elapsed = aggregate(damping=None)
    fixed_dist = _distribution(fixed_counts)
    print(f"  Fixed:  {fixed_dist}  ({fixed_elapsed:.1f}s)")

    print(
        f"[2/3] Running adaptive damping ({ADAPTIVE_DAMPING_START} -> {ADAPTIVE_DAMPING_END} linear)..."
    )
    adaptive_counts, adaptive_elapsed = aggregate(damping=adaptive_damping)
    adaptive_dist = _distribution(adaptive_counts)
    print(f"  Adapt:  {adaptive_dist}  ({adaptive_elapsed:.1f}s)")

    # Compare
    print("\n[3/3] Comparing distributions...")
    delta_3_0 = adaptive_dist["3-0%"] - fixed_dist["3-0%"]
    delta_3_1 = adaptive_dist["3-1%"] - fixed_dist["3-1%"]
    delta_3_2 = adaptive_dist["3-2%"] - fixed_dist["3-2%"]
    print(f"  Delta 3-0%: {delta_3_0:+.2f}")
    print(f"  Delta 3-1%: {delta_3_1:+.2f}")
    print(f"  Delta 3-2%: {delta_3_2:+.2f}")

    # Real SuperLega reference: ~40% 3-0, ~30% 3-1, ~30% 3-2 (rough estimate
    # from memoria, real distribution varies by season and team strength)
    reference = {"3-0%": 40.0, "3-1%": 30.0, "3-2%": 30.0}
    fixed_l1 = sum(abs(fixed_dist[k] - reference[k]) for k in reference)
    adaptive_l1 = sum(abs(adaptive_dist[k] - reference[k]) for k in reference)
    print("\n  Distance to ~40/30/30 reference (L1 sum):")
    print(f"    Fixed:   {fixed_l1:.2f}")
    print(f"    Adaptive: {adaptive_l1:.2f}")
    closer = (
        "adaptive" if adaptive_l1 < fixed_l1 else ("fixed" if fixed_l1 < adaptive_l1 else "tie")
    )
    print(f"    Closer to reference: {closer}")

    if closer == "adaptive" and abs(adaptive_l1 - fixed_l1) > 1.0:
        verdict = "improved"
        recommendation = "apply-adaptive"
    elif closer == "fixed" and abs(fixed_l1 - adaptive_l1) > 1.0:
        verdict = "degraded"
        recommendation = "keep-fixed"
    else:
        verdict = "marginal"
        recommendation = "keep-fixed"

    # Save
    results = {
        "n_mc": n_mc,
        "damping_fixed": MATCH_PREDICTOR_DAMPING,
        "damping_adaptive_start": ADAPTIVE_DAMPING_START,
        "damping_adaptive_end": ADAPTIVE_DAMPING_END,
        "damping_adaptive_total_jornadas": SUPERLEGA_TOTAL_JORNADAS,
        "reference_3_0_30": reference,
        "fixed": fixed_dist,
        "adaptive": adaptive_dist,
        "delta_3_0_pct": delta_3_0,
        "delta_3_1_pct": delta_3_1,
        "delta_3_2_pct": delta_3_2,
        "fixed_l1_to_reference": fixed_l1,
        "adaptive_l1_to_reference": adaptive_l1,
        "closer_to_reference": closer,
        "verdict": verdict,
        "recommendation": recommendation,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-mc", type=int, default=100, help="MC runs per strategy (default 100)")
    args = p.parse_args()
    run_experiment(n_mc=args.n_mc)
