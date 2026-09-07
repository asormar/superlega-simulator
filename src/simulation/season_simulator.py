"""
season_simulator.py — Simulador de temporadas completas.

Genera un calendario round-robin y simula todos los partidos
de una temporada de la SuperLega.
"""

import itertools
import random
from typing import Optional
from dataclasses import dataclass
from src.simulation.simulator import MatchSimulator, MatchResult
from src.simulation.constants import (
    HOME_ADVANTAGE_STRENGTH_BONUS,
    STRENGTH_CLAMP_RANGE,
    MATCH_PREDICTOR_DAMPING,
)

# ─────────────────────────────────────────────────────────────
# Sistema de puntos SuperLega
# ─────────────────────────────────────────────────────────────


def match_points(sets_winner: int, sets_loser: int) -> tuple[int, int]:
    """
    Calcula los puntos de clasificacion segun el resultado.
    SuperLega usa:
      3-0 o 3-1: ganador 3 pts, perdedor 0 pts
      3-2: ganador 2 pts, perdedor 1 pt
    """
    if sets_winner == 3 and sets_loser <= 1:
        return (3, 0)
    elif sets_winner == 3 and sets_loser == 2:
        return (2, 1)
    else:
        return (0, 0)  # No deberia pasar


# ─────────────────────────────────────────────────────────────
# Generador de calendario
# ─────────────────────────────────────────────────────────────


def generate_round_robin(teams: list[str], double: bool = True) -> list[tuple[str, str]]:
    """
    Genera un calendario round-robin.
    Si double=True, cada par juega dos veces (ida y vuelta).
    """
    matches = []
    for home, away in itertools.permutations(teams, 2):
        if not double:
            # Solo ida: evitar duplicados
            if (away, home) in matches:
                continue
        matches.append((home, away))

    # Mezclar para orden aleatorio
    random.shuffle(matches)
    return matches


def generate_jornadas(
    teams: list[str],
    double: bool = False,
    seed: int | None = None,
) -> list[list[tuple[str, str]]]:
    """
    Genera un calendario round-robin CLASICO agrupado por jornadas.

    Cada jornada contiene N/2 partidos (con N equipos) y todos los equipos
    juegan exactamente una vez por jornada. Es el formato real de la SuperLega.

    Algoritmo del circulo (circle method):
      - Fijar el primer equipo y rotar el resto N-1 veces.
      - En cada rotacion se emparejan equipos opuestos en el circulo.
      - Con double=True se concatena la vuelta con la localia invertida.

    Tras construir la estructura se aplica una baraja controlada por ``seed``
    (un ``random.Random`` local, para no contaminar el RNG global que usa
    ``simulate_match``):
      - Se reordena la lista de jornadas de la ida.
      - Se reordena el orden de los partidos dentro de cada jornada.
      - Si ``double=True`` la vuelta se baraja de forma independiente con una
        semilla derivada (``seed + 1`` si se proporciono, si no aleatoria).
    La asignacion local/visitante y los emparejamientos del circulo NO se
    tocan: solo cambia el orden del calendario.

    Args:
        teams: lista de N equipos (N >= 2, se recomienda N par).
        double: si True, genera tambien la vuelta (localia invertida).
        seed: semilla para la baraja. Si es None se usa una semilla aleatoria
            (cada llamada produce un calendario distinto).

    Returns:
        Lista de jornadas; cada jornada es una lista de tuplas (home, away).
        Para N par hay N-1 jornadas en cada vuelta, N*(N-1)/2 partidos en total.
    """
    if len(teams) < 2:
        return []

    # Si N es impar, anadimos un "bye" (None) que ocupa la plaza del que descansa.
    teams_list = list(teams)
    has_bye = len(teams_list) % 2 == 1
    if has_bye:
        teams_list = teams_list + [None]

    n = len(teams_list)
    half = n // 2
    jornadas: list[list[tuple[str, str]]] = []

    # Rotacion del circulo: fija teams_list[0], rota el resto.
    rotating = teams_list[1:]
    for round_idx in range(n - 1):
        round_matches: list[tuple[str, str]] = []
        # Emparejamos el fijo con el primero del bloque rotante
        first = rotating[0]
        if teams_list[0] is not None and first is not None:
            if round_idx % 2 == 0:
                round_matches.append((teams_list[0], first))
            else:
                round_matches.append((first, teams_list[0]))
        # El resto se empareja por oposicion en el circulo
        for i in range(1, half):
            a = rotating[i]
            b = rotating[n - 1 - i]
            if a is None or b is None:
                continue
            if i % 2 == 0:
                round_matches.append((a, b))
            else:
                round_matches.append((b, a))
        jornadas.append(round_matches)
        # Rotar: mover el ultimo del bloque al principio
        rotating = [rotating[-1]] + rotating[:-1]

    # Si anadimos bye, lo eliminamos de las jornadas
    if has_bye:
        cleaned: list[list[tuple[str, str]]] = []
        for j in jornadas:
            cleaned.append([(h, a) for h, a in j if h is not None and a is not None])
        jornadas = cleaned

    # Baraja controlada por semilla: cambia el orden del calendario sin
    # tocar emparejamientos ni local/visitante. Usamos un Random local para
    # no contaminar el RNG global que consume simulate_match.
    rng_ida = random.Random(seed)
    rng_ida.shuffle(jornadas)
    for j in jornadas:
        rng_ida.shuffle(j)

    if double:
        # Vuelta: localia invertida respecto a la ida (espejo), y barajada
        # de forma independiente con una semilla derivada.
        jornadas_vuelta = _invert_localia(jornadas)
        seed_vuelta = (seed + 1) if seed is not None else None
        rng_vuelta = random.Random(seed_vuelta)
        rng_vuelta.shuffle(jornadas_vuelta)
        for j in jornadas_vuelta:
            rng_vuelta.shuffle(j)
        return jornadas + jornadas_vuelta
    return jornadas


def _invert_localia(jornadas: list[list[tuple[str, str]]]) -> list[list[tuple[str, str]]]:
    """Invierte la localia de cada partido y devuelve una nueva lista de jornadas."""
    inverted: list[list[tuple[str, str]]] = []
    for j in jornadas:
        inverted.append([(away, home) for home, away in j])
    # Invertimos tambien el orden de las jornadas para que la vuelta no
    # se sienta como una repeticion especular inmediata.
    inverted.reverse()
    return inverted


# ─────────────────────────────────────────────────────────────
# Clase de clasificacion
# ─────────────────────────────────────────────────────────────


@dataclass
class TeamStanding:
    """Posicion de un equipo en la clasificacion."""

    team: str
    points: int = 0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    sets_won: int = 0
    sets_lost: int = 0
    points_scored: int = 0  # Puntos de volleyball (no de clasificacion)
    points_conceded: int = 0
    wins_3_0: int = 0
    wins_3_1: int = 0
    wins_3_2: int = 0
    losses_2_3: int = 0
    losses_1_3: int = 0
    losses_0_3: int = 0

    @property
    def set_ratio(self) -> float:
        return self.sets_won / max(self.sets_lost, 1)

    @property
    def point_ratio(self) -> float:
        return self.points_scored / max(self.points_conceded, 1)


# ─────────────────────────────────────────────────────────────
# Simulador de temporada
# ─────────────────────────────────────────────────────────────


class SeasonSimulator:
    """
    Simula una temporada completa de la SuperLega.

    Integra opcionalmente MatchPredictor (para calibrar las fuerzas
    de equipo antes de cada partido) y SetPredictor (para calibrar
    el clamp de probabilidad punto a punto).
    """

    def __init__(
        self,
        simulator: Optional[MatchSimulator] = None,
        team_strengths: Optional[dict] = None,
        set_predictor: Optional[object] = None,
        match_predictor: Optional[object] = None,
        feature_builder: Optional[object] = None,
    ):
        self.simulator = simulator or MatchSimulator()
        self.team_strengths = team_strengths or {}
        self.set_predictor = set_predictor
        self.match_predictor = match_predictor
        self.feature_builder = feature_builder

    # ─────────────────────────────────────────────────────────
    # Serializacion de estado (para endpoints jornada-a-jornada)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def serialize_standings(standings_list) -> list[dict]:
        """Convierte una lista de TeamStanding a dicts JSON-serializables."""
        out = []
        for s in standings_list:
            out.append(
                {
                    "equipo": s.team,
                    "puntos": s.points,
                    "pj": s.matches_played,
                    "pg": s.wins,
                    "pp": s.losses,
                    "sg": s.sets_won,
                    "sp": s.sets_lost,
                    "sr": round(s.set_ratio, 2),
                    "pts_favor": s.points_scored,
                    "pts_contra": s.points_conceded,
                    "v3_0": s.wins_3_0,
                    "v3_1": s.wins_3_1,
                    "v3_2": s.wins_3_2,
                    "d2_3": s.losses_2_3,
                    "d1_3": s.losses_1_3,
                    "d0_3": s.losses_0_3,
                }
            )
        return out

    @staticmethod
    def parse_standings(serialized: list[dict]) -> dict:
        """Inverso de serialize_standings; devuelve dict {team: TeamStanding}."""
        standings = {}
        for s_data in serialized or []:
            ts = TeamStanding(team=s_data["equipo"])
            ts.points = s_data.get("puntos", 0)
            ts.matches_played = s_data.get("pj", 0)
            ts.wins = s_data.get("pg", 0)
            ts.losses = s_data.get("pp", 0)
            ts.sets_won = s_data.get("sg", 0)
            ts.sets_lost = s_data.get("sp", 0)
            ts.points_scored = s_data.get("pts_favor", 0)
            ts.points_conceded = s_data.get("pts_contra", 0)
            ts.wins_3_0 = s_data.get("v3_0", 0)
            ts.wins_3_1 = s_data.get("v3_1", 0)
            ts.wins_3_2 = s_data.get("v3_2", 0)
            ts.losses_2_3 = s_data.get("d2_3", 0)
            ts.losses_1_3 = s_data.get("d1_3", 0)
            ts.losses_0_3 = s_data.get("d0_3", 0)
            standings[s_data["equipo"]] = ts
        return standings

    @staticmethod
    def serialize_player_stats(player_stats: dict) -> list[dict]:
        """Convierte el dict de player stats a una lista."""
        return list(player_stats.values())

    @staticmethod
    def parse_player_stats(serialized) -> dict:
        """Inverso de serialize_player_stats; devuelve dict {team|name: {...}}."""
        if not serialized:
            return {}
        if isinstance(serialized, dict):
            return serialized
        out = {}
        for p in serialized:
            key = f"{p['equipo']}|{p['jugador']}"
            out[key] = p
        return out

    def simulate_jornada(
        self,
        schedule: list[list[tuple[str, str]]],
        jornada_index: int,
        current_standings: list[dict],
        current_player_stats,
        equipos: list[str],
        seed: Optional[int] = None,
        use_match_predictor: bool = True,
        use_set_calibration: bool = True,
        damping: Optional[object] = None,
    ) -> dict:
        """
        Simula SOLO una jornada del calendario (modo dinamico para el frontend).

        El backend es stateless: el frontend envia el estado acumulado
        (standings, player_stats) y la simulacion avanza jornada a jornada
        bajo demanda. Cada jornada usa una semilla derivada
        ``seed * 1000 + jornada_index`` para que los resultados sean
        reproducibles jornada por jornada (no requieren ejecutar las previas).

        Args:
            schedule: lista agrupada de jornadas ``[[(home, away), ...], ...]``.
            jornada_index: indice 0-based de la jornada a simular.
            current_standings: lista de standings serializados.
            current_player_stats: dict o lista de player stats acumulados.
            equipos: lista de equipos de la temporada.
            seed: semilla para reproducibilidad.
            use_match_predictor / use_set_calibration: igual que simulate_season.

        Returns:
            dict con la jornada simulada, standings actualizados, player stats
            actualizados y flag is_complete.
        """
        import numpy as _np

        if jornada_index < 0 or jornada_index >= len(schedule):
            return {
                "jornada_index": jornada_index,
                "jornada_matches": [],
                "updated_standings": current_standings or [],
                "updated_player_stats": (
                    current_player_stats
                    if isinstance(current_player_stats, list)
                    else list((current_player_stats or {}).values())
                ),
                "is_complete": True,
                "error": f"jornada_index fuera de rango: {jornada_index}",
            }

        # Semilla derivada por jornada: garantiza determinismo sin replay.
        derived_seed = (seed if seed is not None else 0) * 1000 + jornada_index
        random.seed(derived_seed)
        _np.random.seed(derived_seed)

        standings = self.parse_standings(current_standings)
        player_season_stats = self.parse_player_stats(current_player_stats)

        # Asegurar que existen entradas para todos los equipos
        for t in equipos:
            if t not in standings:
                standings[t] = TeamStanding(team=t)

        # Simular los partidos de la jornada actual
        jornada_matches = []
        jornada_num = jornada_index + 1
        for home, away in schedule[jornada_index]:
            h_str = self.team_strengths.get(home, 0.5)
            a_str = self.team_strengths.get(away, 0.5)
            h_str_adj = min(h_str + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)

            match_features_df = None
            set_pred = None
            team_feats = None

            if (
                use_match_predictor
                and self.match_predictor
                and self.feature_builder
                and hasattr(self.feature_builder, "build_features")
            ):
                try:
                    match_features_df = self.feature_builder.build_features(home, away, jornada_num)
                    # Señal de partido: probabilidad de Elo con margen (limpia,
                    # sin leakage); MatchPredictor solo como fallback.
                    _elo_p = match_features_df.iloc[0].get("elo_win_prob_h", None)
                    if self.match_predictor.feature_names:
                        match_features_df = match_features_df.reindex(
                            columns=self.match_predictor.feature_names,
                            fill_value=0.0,
                        )
                    if _elo_p is not None:
                        p_match_home = float(_elo_p)
                    else:
                        p_match_home = self.match_predictor.predict_proba(match_features_df)[0, 1]
                    # Resolve damping: None -> fixed, callable -> per-jornada
                    if damping is None:
                        current_damping = MATCH_PREDICTOR_DAMPING
                    elif callable(damping):
                        current_damping = damping(jornada_num)
                    else:
                        current_damping = float(damping)
                    h_str_adj, a_str = self._calibrate_strengths(
                        h_str_adj,
                        a_str,
                        float(p_match_home),
                        damping=current_damping,
                    )
                    team_feats = self._extract_set_team_features(match_features_df)
                except Exception as e:
                    import sys

                    print(
                        f"  [WARN] MatchPredictor fallo para {home} vs {away}: {e}", file=sys.stderr
                    )
                    h_str_adj = min(h_str + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)
                    match_features_df = None

            if use_set_calibration and self.set_predictor:
                set_pred = self.set_predictor

            # Extraer features para PointProbabilityModel
            point_match_features = None
            if match_features_df is not None and not match_features_df.empty:
                _row = match_features_df.iloc[0]
                point_match_features = {
                    f: float(_row[f]) if f in match_features_df.columns else 0.0
                    for f in [
                        "elo_diff",
                        "diff_win_rate_global",
                        "diff_set_win_rate",
                        "diff_dominancia",
                        "diff_set_ratio",
                        "diff_forma_efectiva",
                    ]
                }

            match = self.simulator.simulate_match(
                home_team=home,
                away_team=away,
                home_strength=h_str_adj,
                away_strength=a_str,
                match_features=point_match_features,
                generate_points=False,
                generate_player_stats=True,
                set_predictor=set_pred,
                team_features=team_feats,
            )

            jornada_matches.append(match)
            self._update_standings(standings, match)
            self._accumulate_player_stats(player_season_stats, match, home, "home")
            self._accumulate_player_stats(player_season_stats, match, away, "away")

            if self.feature_builder and hasattr(self.feature_builder, "update"):
                pts_home = sum(s.score_home for s in match.sets)
                pts_away = sum(s.score_away for s in match.sets)
                self.feature_builder.update(
                    home,
                    away,
                    match.sets_home,
                    match.sets_away,
                    match.winner,
                    points_local=pts_home,
                    points_visitante=pts_away,
                )

        sorted_standings = sorted(
            standings.values(),
            key=lambda s: (s.points, s.set_ratio, s.point_ratio),
            reverse=True,
        )

        return {
            "jornada_index": jornada_index,
            "jornada_matches": jornada_matches,
            "updated_standings": self.serialize_standings(sorted_standings),
            "updated_player_stats": self.serialize_player_stats(player_season_stats),
            "is_complete": jornada_index >= len(schedule) - 1,
        }

    def simulate_season(
        self,
        teams: list[str],
        double_round_robin: bool = True,
        seed: Optional[int] = None,
        half: Optional[str] = None,
        first_half_state: Optional[dict] = None,
        use_match_predictor: bool = True,
        use_set_calibration: bool = True,
        damping: Optional[object] = None,
    ) -> dict:
        """
        Simula una temporada completa o una mitad.

        Args:
            teams: lista de equipos
            double_round_robin: True = ida y vuelta
            seed: semilla para reproducibilidad
            half: None = completa, "first" = solo primera vuelta, "second" = solo segunda
            first_half_state: estado de la primera vuelta (para continuar segunda)
            use_match_predictor: si True, calibra fuerzas con MatchPredictor
            use_set_calibration: si True, calibra clamp con SetPredictor

        Returns:
            dict con standings, matches, player_season_stats, y estadisticas
        """
        if seed is not None:
            random.seed(seed)

        if half == "second" and first_half_state:
            # Continuar desde la primera vuelta
            standings = {}
            for s_data in first_half_state["standings"]:
                ts = TeamStanding(team=s_data["equipo"])
                ts.points = s_data["puntos"]
                ts.matches_played = s_data["pj"]
                ts.wins = s_data["pg"]
                ts.losses = s_data["pp"]
                ts.sets_won = s_data["sg"]
                ts.sets_lost = s_data["sp"]
                ts.points_scored = s_data.get("pts_favor", 0)
                ts.points_conceded = s_data.get("pts_contra", 0)
                ts.wins_3_0 = s_data.get("v3_0", 0)
                ts.wins_3_1 = s_data.get("v3_1", 0)
                ts.wins_3_2 = s_data.get("v3_2", 0)
                ts.losses_2_3 = s_data.get("d2_3", 0)
                ts.losses_1_3 = s_data.get("d1_3", 0)
                ts.losses_0_3 = s_data.get("d0_3", 0)
                standings[s_data["equipo"]] = ts

            player_season_stats = first_half_state.get("player_season_stats", {})

            # Generar solo la vuelta
            schedule = self._generate_return_leg(teams)
        else:
            standings = {team: TeamStanding(team=team) for team in teams}
            player_season_stats = {}

            if half == "first":
                # Solo ida
                schedule = generate_round_robin(teams, double=False)
            else:
                # Completa
                schedule = generate_round_robin(teams, double=double_round_robin)

        # Simular partidos
        all_matches = []
        jornada_num = 0
        for i, (home, away) in enumerate(schedule):
            jornada_num = i + 1

            h_str = self.team_strengths.get(home, 0.5)
            a_str = self.team_strengths.get(away, 0.5)
            h_str_adj = min(h_str + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)  # ventaja de campo

            # Calibrar fuerzas con MatchPredictor
            match_features_df = None
            set_pred = None
            team_feats = None

            if (
                use_match_predictor
                and self.match_predictor
                and self.feature_builder
                and hasattr(self.feature_builder, "build_features")
            ):
                try:
                    match_features_df = self.feature_builder.build_features(
                        home,
                        away,
                        jornada_num,
                    )
                    # Señal de partido: probabilidad de Elo con margen (limpia,
                    # sin leakage). Reemplaza al MatchPredictor de 87 features,
                    # que medido honestamente estaba a nivel de azar (AUC 0.53).
                    # El MatchPredictor queda solo como fallback.
                    _elo_p = match_features_df.iloc[0].get("elo_win_prob_h", None)
                    # Alinear columnas con lo que espera el modelo (fallback)
                    if self.match_predictor.feature_names:
                        match_features_df = match_features_df.reindex(
                            columns=self.match_predictor.feature_names,
                            fill_value=0.0,
                        )
                    if _elo_p is not None:
                        p_match_home = float(_elo_p)
                    else:
                        p_match_home = self.match_predictor.predict_proba(match_features_df)[0, 1]
                    # Resolve damping: None -> fixed, callable -> per-jornada
                    if damping is None:
                        current_damping = MATCH_PREDICTOR_DAMPING
                    elif callable(damping):
                        current_damping = damping(jornada_num)
                    else:
                        current_damping = float(damping)
                    h_str_adj, a_str = self._calibrate_strengths(
                        h_str_adj,
                        a_str,
                        float(p_match_home),
                        damping=current_damping,
                    )
                    # Preparar team features para SetPredictor
                    team_feats = self._extract_set_team_features(
                        match_features_df,
                    )
                except Exception as e:
                    import sys

                    print(
                        f"  [WARN] MatchPredictor fallo para {home} vs {away}: {e}", file=sys.stderr
                    )
                    h_str_adj = min(h_str + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)
                    match_features_df = None

            if use_set_calibration and self.set_predictor:
                set_pred = self.set_predictor

            # Extraer features para PointProbabilityModel
            point_match_features = None
            if match_features_df is not None and not match_features_df.empty:
                _row = match_features_df.iloc[0]
                point_match_features = {
                    f: float(_row[f]) if f in match_features_df.columns else 0.0
                    for f in [
                        "elo_diff",
                        "diff_win_rate_global",
                        "diff_set_win_rate",
                        "diff_dominancia",
                        "diff_set_ratio",
                        "diff_forma_efectiva",
                    ]
                }

            match = self.simulator.simulate_match(
                home_team=home,
                away_team=away,
                home_strength=h_str_adj,
                away_strength=a_str,
                match_features=point_match_features,
                generate_points=False,
                generate_player_stats=True,
                set_predictor=set_pred,
                team_features=team_feats,
            )

            all_matches.append(match)
            self._update_standings(standings, match)

            # Acumular stats de jugadores
            self._accumulate_player_stats(player_season_stats, match, home, "home")
            self._accumulate_player_stats(player_season_stats, match, away, "away")

            # Actualizar feature builder
            if self.feature_builder and hasattr(self.feature_builder, "update"):
                pts_home = sum(s.score_home for s in match.sets)
                pts_away = sum(s.score_away for s in match.sets)
                self.feature_builder.update(
                    home,
                    away,
                    match.sets_home,
                    match.sets_away,
                    match.winner,
                    points_local=pts_home,
                    points_visitante=pts_away,
                )

        # Ordenar clasificacion
        sorted_standings = sorted(
            standings.values(),
            key=lambda s: (s.points, s.set_ratio, s.point_ratio),
            reverse=True,
        )

        return {
            "standings": sorted_standings,
            "matches": all_matches,
            "schedule": schedule,
            "total_matches": len(all_matches),
            "player_season_stats": player_season_stats,
            "half": half,
        }

    def _generate_return_leg(self, teams: list[str]) -> list[tuple[str, str]]:
        """Genera solo los partidos de vuelta (invertidos respecto a ida).

        Para cada par (A, B) que aparece en la ida, genera (B, A) como vuelta.
        """
        seen = set()
        vuelta = []
        for home, away in itertools.permutations(teams, 2):
            if (away, home) in seen:
                vuelta.append((home, away))
            else:
                seen.add((home, away))
        random.shuffle(vuelta)
        return vuelta

    def _accumulate_player_stats(
        self,
        season_stats: dict,
        match: MatchResult,
        team: str,
        side: str,
    ):
        """Acumula stats de jugadores de un partido en el acumulado de temporada."""
        for s in match.sets:
            stats_list = s.home_player_stats if side == "home" else s.away_player_stats
            for p in stats_list or []:
                name = p.get("jugador", "Desconocido")
                key = f"{team}|{name}"
                if key not in season_stats:
                    season_stats[key] = {
                        "equipo": team,
                        "jugador": name,
                        "partidos": 0,
                        "sets": 0,
                        "puntos": 0,
                        "aces": 0,
                        "ataques_ganados": 0,
                        "bloqueos": 0,
                        "recepciones_exc": 0,
                        "errores_saque": 0,
                    }
                ps = season_stats[key]
                ps["sets"] += 1
                ps["puntos"] += p.get("puntos", 0)
                ps["aces"] += p.get("aces", 0)
                ps["ataques_ganados"] += p.get("ataques_ganados", 0)
                ps["bloqueos"] += p.get("bloqueos", 0)
                ps["recepciones_exc"] += p.get("recepciones_exc", 0)
                ps["errores_saque"] += p.get("errores_saque", 0)

        # Incrementar partidos (una vez por match, no por set)
        for s in match.sets[:1]:
            stats_list = s.home_player_stats if side == "home" else s.away_player_stats
            for p in stats_list or []:
                name = p.get("jugador", "Desconocido")
                key = f"{team}|{name}"
                if key in season_stats:
                    season_stats[key]["partidos"] += 1

    def _update_standings(self, standings: dict, match: MatchResult):
        """Actualiza la clasificacion con el resultado de un partido."""
        home = standings[match.home_team]
        away = standings[match.away_team]

        home.matches_played += 1
        away.matches_played += 1

        home.sets_won += match.sets_home
        home.sets_lost += match.sets_away
        away.sets_won += match.sets_away
        away.sets_lost += match.sets_home

        # Contar puntos de volleyball
        for s in match.sets:
            home.points_scored += s.score_home
            home.points_conceded += s.score_away
            away.points_scored += s.score_away
            away.points_conceded += s.score_home

        # Puntos de clasificacion
        w_pts, l_pts = match_points(
            max(match.sets_home, match.sets_away),
            min(match.sets_home, match.sets_away),
        )

        if match.winner == "home":
            home.wins += 1
            home.points += w_pts
            away.losses += 1
            away.points += l_pts

            resultado = f"{match.sets_home}-{match.sets_away}"
            if resultado == "3-0":
                home.wins_3_0 += 1
                away.losses_0_3 += 1
            elif resultado == "3-1":
                home.wins_3_1 += 1
                away.losses_1_3 += 1
            elif resultado == "3-2":
                home.wins_3_2 += 1
                away.losses_2_3 += 1
        else:
            away.wins += 1
            away.points += w_pts
            home.losses += 1
            home.points += l_pts

            resultado = f"{match.sets_away}-{match.sets_home}"
            if resultado == "3-0":
                away.wins_3_0 += 1
                home.losses_0_3 += 1
            elif resultado == "3-1":
                away.wins_3_1 += 1
                home.losses_1_3 += 1
            elif resultado == "3-2":
                away.wins_3_2 += 1
                home.losses_2_3 += 1

    @staticmethod
    def _calibrate_strengths(
        h_str: float,
        a_str: float,
        p_target: float,
        damping: float = MATCH_PREDICTOR_DAMPING,
    ) -> tuple[float, float]:
        """
        Ajusta h_str para que la probabilidad base del Markov
        se acerque a p_target (prediccion del MatchPredictor).

        El damping previene sobrecorreccion: el MatchPredictor
        tiene AUC=0.71, no es perfecto.
        """
        if p_target <= 0 or p_target >= 1:
            return h_str, a_str

        total = h_str + a_str
        if total <= 0:
            return h_str, a_str

        p_base = h_str / total
        if p_base <= 0 or p_base >= 1:
            return h_str, a_str

        odds_target = p_target / (1 - p_target)
        odds_base = p_base / (1 - p_base)
        k = odds_target / odds_base
        k_damped = k**damping

        h_new = h_str * k_damped
        h_new = max(STRENGTH_CLAMP_RANGE[0], min(STRENGTH_CLAMP_RANGE[1], h_new))
        return h_new, a_str

    @staticmethod
    def _extract_set_team_features(
        match_features_df,
        set_index: int = 1,
        sets_h_antes: int = 0,
        sets_a_antes: int = 0,
        prev_home_won: int = -1,
    ) -> dict:
        """
        Extrae features de equipo del DataFrame de match features
        llamando al contrato build_set_features(SetContext).

        Args:
            match_features_df: DataFrame con 1 fila de rolling features.
            set_index: numero de set (1..5).
            sets_h_antes: sets ganados por local antes de este set.
            sets_a_antes: sets ganados por visitante antes de este set.
            prev_home_won: -1 (1er set), 1 (local gano anterior), 0 (perdio).

        Returns:
            dict con las 21 SET_FEATURE_COLS (o vacio si no hay datos).
        """
        if match_features_df is None or match_features_df.empty:
            return {}

        from src.data.set_feature_contract import SetContext, build_set_features
        from src.data.rolling_features import elo_to_strength

        row = match_features_df.iloc[0]

        elo_h = float(row.get("elo_h", 1500.0))
        elo_a = float(row.get("elo_a", 1500.0))

        # Fallback chain for column names: prefer EWMA (decision #4), fall back
        # to home/away-only names, then to default 0.5.
        h_form = float(
            row.get(
                "h_form_ewma",
                row.get("h_forma_home", 0.5),
            )
        )
        a_form = float(
            row.get(
                "a_form_ewma",
                row.get("a_forma_away", 0.5),
            )
        )

        # Point ratio: prefer new "h_point_ratio" name, fall back to legacy
        h_pr = float(
            row.get(
                "h_point_ratio",
                row.get("h_pts_fav_exp", 0.5),
            )
        )
        a_pr = float(
            row.get(
                "a_point_ratio",
                row.get("a_pts_fav_exp", 0.5),
            )
        )

        ctx = SetContext(
            temporada_inicio=int(row.get("temporada_inicio", 0)),
            jornada_num=int(row.get("jornada_num", 0)),
            match_id=str(row.get("partido_id", "")),
            set_index=set_index,
            equipo_local=str(row.get("local", "")),
            equipo_visitante=str(row.get("visitante", "")),
            elo_local=elo_h,
            elo_visitante=elo_a,
            strength_local=elo_to_strength(elo_h),
            strength_visitante=elo_to_strength(elo_a),
            h_win_rate_global=float(row.get("h_win_rate", 0.5)),
            a_win_rate_global=float(row.get("a_win_rate", 0.5)),
            h_set_win_rate=float(row.get("h_set_win_rate", row.get("h_set_ratio", 0.5))),
            a_set_win_rate=float(row.get("a_set_win_rate", row.get("a_set_ratio", 0.5))),
            h_form_ewma=h_form,
            a_form_ewma=a_form,
            h_set_diff_exp=float(row.get("h_set_diff_exp", 0.0)),
            a_set_diff_exp=float(row.get("a_set_diff_exp", 0.0)),
            h_point_ratio=h_pr,
            a_point_ratio=a_pr,
            h2h_win_rate=float(row.get("h_h2h_win_rate", 0.5)),
            sets_h_antes=sets_h_antes,
            sets_a_antes=sets_a_antes,
            prev_home_won=prev_home_won,
            target_score=25 if (sets_h_antes + sets_a_antes) < 8 else 15,
        )
        return build_set_features(ctx)

    def print_standings(self, standings: list):
        """Imprime la tabla de clasificacion."""
        print(
            f"\n  {'Pos':>3} {'Equipo':<20} {'Pts':>4} {'PJ':>3} {'PG':>3} "
            f"{'PP':>3} {'SG':>3} {'SP':>3} {'SR':>5}"
        )
        print("  " + "-" * 70)

        for i, s in enumerate(standings, 1):
            print(
                f"  {i:>3} {s.team:<20} {s.points:>4} {s.matches_played:>3} "
                f"{s.wins:>3} {s.losses:>3} {s.sets_won:>3} "
                f"{s.sets_lost:>3} {s.set_ratio:>5.2f}"
            )


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.team_mapper import get_superliga_teams

    teams = get_superliga_teams("2024/2025")

    # Fuerzas estimadas (basadas en la temporada 2024/2025 real)
    strengths = {
        "Trento": 0.68,
        "Perugia": 0.65,
        "Verona": 0.60,
        "Piacenza": 0.58,
        "Lube": 0.56,
        "Milano": 0.53,
        "Modena": 0.52,
        "Monza": 0.48,
        "Cisterna": 0.45,
        "Padova": 0.47,
        "Taranto": 0.40,
        "Grottazzolina": 0.35,
    }

    print("=" * 70)
    print("  SIMULACION DE TEMPORADA 2024/2025 SuperLega")
    print("=" * 70)

    season_sim = SeasonSimulator(
        simulator=MatchSimulator(),
        team_strengths=strengths,
    )

    result = season_sim.simulate_season(
        teams=teams,
        double_round_robin=True,
        seed=42,
    )

    print(f"\n  Total partidos simulados: {result['total_matches']}")
    season_sim.print_standings(result["standings"])
