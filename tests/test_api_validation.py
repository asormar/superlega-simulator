"""Tests for API Pydantic validation 422 paths + happy-path 200 responses.

Covers every ``_val_*`` validator in ``SimularPartidoRequest`` and
``SimularTemporadaRequest``, plus the ``_val_diff_teams`` cross-string-same-key
quirk documented in the spec.
"""

from fastapi.testclient import TestClient
from src.api.main import app

# ─────────────────────────────────────────────────────────────
# 422 validation paths
# ─────────────────────────────────────────────────────────────


class Test422TeamValidation:
    """_val_team: empty, non-string, unrecognised team."""

    def test_val_team_empty_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "",
                "visitante": "Perugia",
            },
        )
        assert resp.status_code == 422

    def test_val_team_non_string_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": 42,
                "visitante": "Perugia",
            },
        )
        assert resp.status_code == 422

    def test_val_team_unknown_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "NonExistentTeam",
                "visitante": "Perugia",
            },
        )
        assert resp.status_code == 422


class Test422DiffTeams:
    """_val_diff_teams: same team rejected (normalised)."""

    def test_val_diff_teams_same_string_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Trento",
                "visitante": "Trento",
            },
        )
        assert resp.status_code == 422

    def test_val_diff_teams_cross_string_same_key_rejected(self, app_with_synthetic):
        """Different raw strings that normalise to the same canonical key.

        ``normalize_team_name("Diatec Trentino")`` and
        ``normalize_team_name("Trento")`` both → ``"Trento"``, so the source's
        ``_val_diff_teams`` rejects the pair.  The Batch 2b spec R8 says this
        SHOULD be accepted (no 422); this test pins the CURRENT source behaviour
        and flags the spec discrepancy.
        """
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Diatec Trentino",
                "visitante": "Trento",
            },
        )
        assert resp.status_code == 422


class Test422Seed:
    """_val_seed — negative seed rejected."""

    def test_val_seed_negative_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Trento",
                "visitante": "Perugia",
                "semilla": -1,
            },
        )
        assert resp.status_code == 422


class Test422Strength:
    """_val_strength — out-of-[0,1] range."""

    def test_val_strength_too_low_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Trento",
                "visitante": "Perugia",
                "fuerza_local": -0.1,
            },
        )
        assert resp.status_code == 422

    def test_val_strength_too_high_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Trento",
                "visitante": "Perugia",
                "fuerza_local": 1.1,
            },
        )
        assert resp.status_code == 422


class Test422Half:
    """_val_half — invalid half value."""

    def test_val_half_invalid_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/temporada",
            json={
                "equipos": ["Trento", "Perugia"],
                "half": "invalid",
            },
        )
        assert resp.status_code == 422


class Test422HalfState:
    """_val_half_state — half='second' without first_half_state."""

    def test_val_half_state_missing_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/temporada",
            json={
                "equipos": ["Trento", "Perugia"],
                "half": "second",
            },
        )
        assert resp.status_code == 422


class Test422EquiposCap:
    """_val_equipos — too many / too few teams."""

    def test_equipos_cap_13_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/temporada",
            json={
                "equipos": [f"Team{i}" for i in range(13)],
            },
        )
        assert resp.status_code == 422

    def test_equipos_less_than_2_rejected(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/temporada",
            json={
                "equipos": ["Trento"],
            },
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────
# Happy-path 200 responses
# ─────────────────────────────────────────────────────────────


class TestHappyPath:
    """Endpoints return 200 for valid requests."""

    def test_get_equipos_returns_200(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.get("/api/equipos")
        assert resp.status_code == 200

    def test_get_equipo_detail_returns_200(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.get("/api/equipos/Trento")
        assert resp.status_code == 200

    def test_post_simular_partido_returns_200(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/partido",
            json={
                "local": "Trento",
                "visitante": "Perugia",
                "semilla": 42,
                "generar_puntos": False,
                "generar_stats_jugadores": False,
            },
        )
        assert resp.status_code == 200

    def test_post_simular_temporada_returns_200(self, app_with_synthetic):
        client = TestClient(app)
        resp = client.post(
            "/api/simular/temporada",
            json={
                "equipos": ["Trento", "Perugia"],
                "doble_vuelta": True,
                "semilla": 42,
                "use_match_predictor": False,
                "use_set_calibration": False,
            },
        )
        assert resp.status_code == 200
