"""
Pinned tests for set_feature_contract.py — contract-only, no runtime wiring.

REQ-021..025, SCN-001, SCN-002, SCN-007, SCN-008.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.data.feature_store import SET_FEATURE_COLS
from src.data.set_feature_contract import SetContext, build_set_features

# ── Shared context for deterministic tests ──────────────────────────────────

_BASE_CTX = SetContext(
    temporada_inicio=2024,
    jornada_num=11,
    match_id="2024/2025_m00001",
    set_index=3,
    equipo_local="Trento",
    equipo_visitante="Perugia",
    elo_local=1550.0,
    elo_visitante=1480.0,
    strength_local=0.57,
    strength_visitante=0.48,
    h_win_rate_global=0.55,
    a_win_rate_global=0.50,
    h_set_win_rate=0.58,
    a_set_win_rate=0.48,
    h_form_ewma=0.53,
    a_form_ewma=0.49,
    h_set_diff_exp=0.15,
    a_set_diff_exp=-0.10,
    h_point_ratio=0.55,
    a_point_ratio=0.45,
    h2h_win_rate=0.50,
    sets_h_antes=1,
    sets_a_antes=1,
    prev_home_won=1,
    target_score=25,
)


# ── Test helpers ────────────────────────────────────────────────────────────

TEAMS_12 = [
    "Trento",
    "Perugia",
    "Piacenza",
    "Verona",
    "Lube",
    "Milano",
    "Modena",
    "Monza",
    "Cisterna",
    "Padova",
    "Taranto",
    "Grottazzolina",
]


# ── Tests ───────────────────────────────────────────────────────────────────


class TestContractDeterminism:
    """REQ-023, SCN-001: build_set_features is deterministic."""

    def test_contract_deterministic(self):
        """Call build_set_features 100x on the same ctx — all equal."""
        r0 = build_set_features(_BASE_CTX)
        for _ in range(99):
            r = build_set_features(_BASE_CTX)
            assert r == r0, f"Feature dict changed on repeat call: {r} != {r0}"

    def test_contract_pure_no_mutation(self):
        """REQ-024, SCN-002: build_set_features does not mutate ctx."""
        ctx_before = dataclasses.asdict(_BASE_CTX)
        _ = build_set_features(_BASE_CTX)
        ctx_after = dataclasses.asdict(_BASE_CTX)
        assert ctx_before == ctx_after, "SetContext mutated by build_set_features"


class TestContractSchema:
    """REQ-002: build_set_features returns exactly the 21 SET_FEATURE_COLS."""

    def test_all_21_columns_present(self):
        """Assert set equality + length 21."""
        feats = build_set_features(_BASE_CTX)
        assert len(feats) == 21, f"Expected 21 features, got {len(feats)}"
        assert set(feats.keys()) == set(SET_FEATURE_COLS), (
            f"Feature keys mismatch. "
            f"Missing: {set(SET_FEATURE_COLS) - set(feats.keys())}. "
            f"Extra: {set(feats.keys()) - set(SET_FEATURE_COLS)}."
        )


class TestPtsFavNoLiveScore:
    """REQ-025, SCN-007: pts_fav_h/a come from historical ratio, not live score."""

    def test_pts_fav_no_live_score(self):
        """Build a context with h_point_ratio=0.55; assert pts_fav_h ~ 0.55."""
        ctx = dataclasses.replace(
            _BASE_CTX,
            h_point_ratio=0.55,
            a_point_ratio=0.45,
        )
        feats = build_set_features(ctx)
        assert feats["pts_fav_h"] == pytest.approx(
            0.55, abs=1e-9
        ), f"pts_fav_h={feats['pts_fav_h']} should be h_point_ratio=0.55"
        assert feats["pts_fav_a"] == pytest.approx(
            0.45, abs=1e-9
        ), f"pts_fav_a={feats['pts_fav_a']} should be a_point_ratio=0.45"
        # Verify momentum is discrete (decision #2)
        assert feats["momentum_h"] in (
            0.0,
            0.5,
            1.0,
        ), f"momentum_h={feats['momentum_h']} should be discrete {{0, 0.5, 1.0}}"
        # This ctx has prev_home_won=1 -> momentum_h should be 1.0
        assert feats["momentum_h"] == 1.0


@pytest.mark.slow
class TestPSetDiscriminates:
    """REQ-022, SCN-008: p_set varies across the 132 A5 pairs."""

    @pytest.fixture(scope="class")
    def v2_model(self):
        """Load the v2 SetPredictor from disk."""
        import joblib
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parent.parent
        model_path = BASE_DIR / "models" / "set_predictor_v2.joblib"
        if not model_path.exists():
            pytest.fail(
                "set_predictor_v2.joblib required for discrimination test — run `python -m src.models.train_improved` first"
            )
        loaded = joblib.load(model_path)
        return loaded

    @pytest.fixture(scope="class")
    def team_data(self):
        """Real Elo ratings and strengths for TEAMS_12."""
        from src.data.rolling_features import (
            get_historical_team_elo,
            elo_to_strength,
            ELO_BASE,
        )

        elo_dict = get_historical_team_elo()
        return {
            t: {
                "elo": elo_dict.get(t, ELO_BASE),
                "strength": elo_to_strength(elo_dict.get(t, ELO_BASE)),
            }
            for t in TEAMS_12
        }

    @pytest.fixture(scope="class")
    def p_set_values(self, v2_model, team_data):
        """Compute p_set for all 132 ordered pairs of TEAMS_12 with real strengths."""
        import pandas as pd
        from src.data.set_feature_contract import SetContext, build_set_features

        model = v2_model["model"]
        features = v2_model["features"]
        results = []

        for home in TEAMS_12:
            hd = team_data[home]
            for away in TEAMS_12:
                if home == away:
                    continue
                ad = team_data[away]
                # Build a SetContext using REAL Elo/strength for each pair
                ctx = SetContext(
                    temporada_inicio=2024,
                    jornada_num=11,
                    match_id=f"{home}_vs_{away}",
                    set_index=1,
                    equipo_local=home,
                    equipo_visitante=away,
                    elo_local=hd["elo"],
                    elo_visitante=ad["elo"],
                    strength_local=hd["strength"],
                    strength_visitante=ad["strength"],
                    h_set_win_rate=0.5,
                    a_set_win_rate=0.5,
                    h_form_ewma=0.5,
                    a_form_ewma=0.5,
                    h_point_ratio=0.5,
                    a_point_ratio=0.5,
                    h2h_win_rate=0.5,
                    sets_h_antes=0,
                    sets_a_antes=0,
                    prev_home_won=-1,
                    target_score=25,
                )
                feats = build_set_features(ctx)
                row = {f: feats.get(f, 0.0) for f in features}
                df = pd.DataFrame([row])
                proba = model.predict_proba(df.fillna(0))
                p_set = float(proba[0, 1])
                results.append({"home": home, "away": away, "p_set": p_set})

        return [r["p_set"] for r in results]

    def test_p_set_discriminates(self, p_set_values):
        """std(p_set) > 0.05 (today it was ~0.007)."""
        std = float(np.std(p_set_values))
        assert std > 0.05, (
            f"p_set std={std:.4f} across 132 pairs. "
            f"Expected > 0.05. The contract is not discriminating."
        )


@pytest.mark.slow
class TestRuntimeConsumerPin:
    """R3#1 (CRITICAL): The runtime consumer _eval_set_predictor does NOT
    override contract-passed values with live score.

    Regression guard: if a future commit re-adds live-score override inside
    _simulate_set, simulate_match, or _eval_set_predictor (e.g., overwriting
    pts_fav_h with score_home / target_score), this test will fail because the
    prediction from _eval_set_predictor will differ from the expected value
    computed from the contract-only path.
    """

    V2_PATH = Path(__file__).resolve().parent.parent / "models" / "set_predictor_v2.joblib"

    @pytest.fixture(scope="class")
    def v2_adapter(self):
        """Load the real v2 LogRegSetPredictor."""
        from src.models.set_predictor_v2 import LogRegSetPredictor

        return LogRegSetPredictor.load(self.V2_PATH)

    def test_eval_set_predictor_ignores_live_score(self, v2_adapter):
        """_eval_set_predictor returns prediction consistent with contract
        pts_fav_h=0.55, NOT with score_home/(score_home+score_away)."""
        import pandas as pd
        from src.simulation.simulator import MatchSimulator

        # Build contract features with h_point_ratio=0.55
        ctx = dataclasses.replace(
            _BASE_CTX,
            h_point_ratio=0.55,
            a_point_ratio=0.45,
        )
        feats = build_set_features(ctx)

        # Compute expected prediction directly from the model using
        # contract-only features (pts_fav_h=0.55 via h_point_ratio)
        df = pd.DataFrame([{f: feats.get(f, 0.0) for f in v2_adapter.feature_names}])
        expected = float(v2_adapter.predict_proba(df)[0, 1])

        # Call _eval_set_predictor with live-score values that WOULD differ
        # if the old override were re-added:
        #   score_home=25, score_away=0  =>  pts_fav would be 1.0 if overridden
        #   vs contract: pts_fav_h=0.55
        sim = MatchSimulator()
        result = sim._eval_set_predictor(
            set_predictor=v2_adapter,
            set_context_base=feats,
            score_home=25,
            score_away=0,
            target_score=25,
            sets_home_antes=0,
            sets_away_antes=0,
        )

        assert result is not None, "_eval_set_predictor returned None"
        assert result == pytest.approx(expected, abs=1e-9), (
            f"_eval_set_predictor returned {result} but contract-only path "
            f"gives {expected}. If this fails, a live-score override may "
            f"have been re-introduced."
        )
