"""Tests de src/simulation/set_math.py (A2).

Pinean la conversion PUNTO <-> SET que centra el clamp adaptativo.
"""

import pytest

from src.simulation.set_math import p_set_from_p_point, p_point_from_p_set


class TestPSetFromPPoint:
    """Forma cerrada: P(set | p_punto)."""

    @pytest.mark.parametrize("target", [15, 25])
    def test_fair_point_gives_fair_set(self, target):
        """p=0.5 -> p_set=0.5 exactamente (autoconsistencia del modelo)."""
        assert p_set_from_p_point(0.5, target) == pytest.approx(0.5, abs=1e-9)

    def test_known_value_25(self):
        """Valor de referencia pineado.

        NOTA (Guardrail 4): el plan consolidado (A2, paso 1) indicaba
        `p_set_from_p_point(0.52, 25) ~ 0.66 (+-0.02)`. Ese valor del plan es
        INCORRECTO; el valor real de la formula es 0.6131. La formula se valida
        de forma independiente con `test_fair_point_gives_fair_set` (da 0.5
        exacto en p=0.5) y con la monotonia, asi que lo que estaba mal era la
        constante esperada del documento, no la implementacion.
        """
        assert p_set_from_p_point(0.52, 25) == pytest.approx(0.6131, abs=0.001)

    def test_known_value_15(self):
        assert p_set_from_p_point(0.52, 15) == pytest.approx(0.5889, abs=0.001)

    @pytest.mark.parametrize("target", [15, 25])
    def test_monotonic_increasing(self, target):
        vals = [p_set_from_p_point(p / 100, target) for p in range(35, 66)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    @pytest.mark.parametrize("target", [15, 25])
    def test_symmetry(self, target):
        """P(set|p) + P(set|1-p) == 1: el set siempre lo gana alguien."""
        for p in (0.45, 0.48, 0.52, 0.55, 0.60):
            total = p_set_from_p_point(p, target) + p_set_from_p_point(1 - p, target)
            assert total == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("target", [15, 25])
    def test_bounded(self, target):
        for p in (0.05, 0.3, 0.5, 0.7, 0.95):
            assert 0.0 <= p_set_from_p_point(p, target) <= 1.0

    def test_longer_set_amplifies_edge(self):
        """Un set a 25 amplifica mas la ventaja que uno a 15 (mas puntos)."""
        assert p_set_from_p_point(0.55, 25) > p_set_from_p_point(0.55, 15)


class TestPPointFromPSet:
    """Inversa numerica por biseccion."""

    @pytest.mark.parametrize("target", [15, 25])
    def test_fair_set_gives_fair_point(self, target):
        assert p_point_from_p_set(0.5, target) == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.parametrize("target", [15, 25])
    @pytest.mark.parametrize("p_point", [0.45, 0.48, 0.50, 0.52, 0.55, 0.58])
    def test_roundtrip(self, target, p_point):
        """p_point -> p_set -> p_point recupera el original."""
        p_set = p_set_from_p_point(p_point, target)
        assert p_point_from_p_set(p_set, target) == pytest.approx(p_point, abs=1e-3)

    @pytest.mark.parametrize("target", [15, 25])
    def test_monotonic_increasing(self, target):
        vals = [p_point_from_p_set(p / 100, target) for p in range(10, 91)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    @pytest.mark.parametrize("target", [15, 25])
    def test_saturates_without_error(self, target):
        """Entradas degeneradas se saturan en vez de reventar."""
        for p_set in (0.0, 1.0, -0.5, 1.5):
            val = p_point_from_p_set(p_set, target)
            assert 0.01 <= val <= 0.99

    def test_scale_compression_is_the_point_of_a2(self):
        """El motivo de A2: p_set extremo -> p_punto casi neutro.

        Un favorito con P(set)=0.75 solo necesita P(punto)~0.55, NO 0.75.
        Centrar el clamp de punto en p_set era un error de escala.
        """
        assert p_point_from_p_set(0.75, 25) == pytest.approx(0.546, abs=0.01)
        assert p_point_from_p_set(0.25, 25) == pytest.approx(0.454, abs=0.01)

    def test_cache_is_active(self):
        p_point_from_p_set.cache_clear()
        p_point_from_p_set(0.63, 25)
        p_point_from_p_set(0.63, 25)
        assert p_point_from_p_set.cache_info().hits >= 1
