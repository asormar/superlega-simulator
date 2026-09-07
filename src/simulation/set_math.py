"""set_math.py — Conversion entre probabilidad de PUNTO, SET y PARTIDO.

Motivacion (A2 del plan consolidado): el clamp adaptativo centraba el rango de
P(ganar un PUNTO) en `p_set`, que vive en la escala de SET. Es un error de
escala: un favorito con P(set)=0.75 solo necesita P(punto)~0.53, no 0.75.

Estas funciones son la conversion que faltaba:

    p_set_from_p_point : escala PUNTO -> escala SET  (forma cerrada)
    p_point_from_p_set : escala SET   -> escala PUNTO (inversa numerica)
    p_match_from_p_set : escala SET  -> escala PARTIDO (best-of-5 forma cerrada)

Modelo: puntos i.i.d. con probabilidad `p` para el local. Un set se gana a
`target` puntos con ventaja de 2 (deuce indefinido). Es una simplificacion
--los puntos reales no son independientes (sideout, momentum)-- pero es la
misma aproximacion que ya usa el resto del simulador y basta para fijar la
ESCALA, que es lo que A2 corrige.

Rangos utiles de referencia (target=25):
    p_punto 0.50 -> p_set 0.5000
    p_punto 0.52 -> p_set 0.6131
    p_punto 0.55 -> p_set 0.7641
    p_punto 0.60 -> p_set 0.9264

Notese lo estrecha que es la banda util: toda la escala de set se recorre con
variaciones de centesimas en la escala de punto. Por eso el centro del clamp
debe construirse en escala de PUNTO.

B4 anade la tercera conversion:

    p_match_from_p_set : escala SET -> escala PARTIDO (best-of-5 forma cerrada)

Formula: P(match) = q^3 + 3*q^3*(1-q) + 6*q^2*(1-q)^2*q5

Donde q = P(set a 25 puntos) y q5 = P(set a 15 puntos, tiebreak).
Si q5 no se especifica, se usa la composicion A2:
    q5 = p_set_from_p_point(p_point_from_p_set(q, 25), 15)
que convierte q a punto (target=25) y luego a set a 15 (target=15).
"""

from functools import lru_cache
from math import comb


def p_set_from_p_point(p: float, target: int = 25) -> float:
    """P(ganar un set a `target` puntos | p = P(ganar cada punto), i.i.d.).

    Args:
        p: probabilidad de ganar un punto individual, en (0, 1).
        target: puntos para cerrar el set (25 normal, 15 en el quinto).

    Returns:
        Probabilidad de ganar el set, en (0, 1). Monotona creciente en `p`.
    """
    # Victoria sin deuce: gana target-a-j con j <= target-2. Los primeros
    # (target-1+j) puntos contienen target-1 ganados y j perdidos; el ultimo
    # punto lo gana el ganador del set.
    win = sum(comb(target - 1 + j, j) * p**target * (1 - p) ** j for j in range(target - 1))
    # Deuce: llegar a (target-1)-(target-1) y despues ganar por 2 (geometrico).
    deuce_reach = comb(2 * (target - 1), target - 1) * (p * (1 - p)) ** (target - 1)
    p_deuce_win = p * p / (p * p + (1 - p) * (1 - p))
    return win + deuce_reach * p_deuce_win


@lru_cache(maxsize=4096)
def p_point_from_p_set(p_set: float, target: int = 25) -> float:
    """Inversa numerica de `p_set_from_p_point` por biseccion.

    `p_set_from_p_point` es estrictamente monotona creciente, asi que la
    biseccion converge siempre. Se redondea `p_set` a 3 decimales para que la
    cache LRU sea efectiva (el clamp se evalua una vez por set, pero
    monte_carlo_simulate lo repite n_simulations veces con el mismo valor).

    Args:
        p_set: probabilidad de ganar el set, se satura a [0.001, 0.999].
        target: puntos para cerrar el set (25 normal, 15 en el quinto).

    Returns:
        Probabilidad de punto equivalente, en [0.01, 0.99].
    """
    p_set = min(max(round(p_set, 3), 0.001), 0.999)
    lo, hi = 0.01, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2
        if p_set_from_p_point(mid, target) < p_set:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def p_match_from_p_set(q: float, q5: float | None = None) -> float:
    """P(ganar el partido al mejor de 5 | q = P(ganar un set a 25)).

    Formula cerrada para best-of-5 con sets a 25 puntos (q) y posible
    tiebreak a 15 puntos (q5). Si q5 es None, se usa la composicion A2:
    p_set_from_p_point(p_point_from_p_set(q, 25), 15).

    La formula cuenta tres caminos:
      - 3-0: gana los 3 primeros sets: q^3
      - 3-1: gana 3 de los primeros 4 perdiendo exactamente 1: 3*q^3*(1-q)
      - 3-2: gana 3 de 5 perdiendo 2 (el 5.o set se juega a 15): 6*q^2*(1-q)^2*q5

    Args:
        q: probabilidad de ganar un set a 25 puntos.
        q5: probabilidad de ganar el tiebreak a 15 puntos (set 5).
            Si es None, se usa la composicion A2:
            p_set_from_p_point(p_point_from_p_set(q, 25), 15).

    Returns:
        Probabilidad de ganar el partido, en [0, 1].
    """
    if q5 is None:
        q5 = p_set_from_p_point(p_point_from_p_set(q, 25), 15)
    # 3-0
    win_3_0 = q**3
    # 3-1 (gana 3 de 4, pierde exactamente 1)
    win_3_1 = 3.0 * q**3 * (1.0 - q)
    # 3-2 (gana 3 de 5, los primeros 4 sets son 2-2, el 5o se gana con prob q5)
    win_3_2 = 6.0 * q**2 * (1.0 - q) ** 2 * q5
    return win_3_0 + win_3_1 + win_3_2
