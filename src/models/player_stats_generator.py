"""
player_stats_generator.py — Generador de estadisticas de jugadores.

Ajusta distribuciones estadisticas a los datos historicos de jugadores
y genera stats realistas para partidos simulados.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class PlayerStatsGenerator:
    """
    Genera estadisticas realistas de jugadores para partidos simulados.

    Para cada equipo/temporada, ajusta distribuciones a las stats per-set
    de cada jugador. En simulacion, samplea de estas distribuciones y
    normaliza para que los totales sean consistentes con el marcador.
    """

    # Stats que generamos por jugador por set
    STAT_KEYS = [
        "puntos",
        "aces",
        "ataques_ganados",
        "bloqueos",
        "recepciones_exc",
        "errores_saque",
    ]

    def __init__(self):
        self.team_profiles = {}  # {equipo_id: {jugador: {stat: {mean, std}}}}
        self.team_rosters = {}  # {equipo_id: [lista de jugadores]}
        self._canonical_map = None  # cache: dict canonical_name -> equipo_id

    def _build_canonical_map(self):
        """Construye cache canonico -> equipo_id para busqueda rapida.

        Cuando dos equipo_id mapean al mismo nombre canonico (ej: BAM y
        CUNEOSPORT -> Cuneo), se queda con el que tenga mayor roster.
        Si hay empate, elige el equipo_id alfabeticamente primero para
        garantizar determinismo independientemente del orden de pd.unique().
        """
        from src.data.team_mapper import normalize_team_name

        self._canonical_map = {}
        for eid in self.team_profiles:
            canonical = normalize_team_name(self._extract_team_name(eid))
            if canonical not in self._canonical_map:
                self._canonical_map[canonical] = eid
            # Si hay colision (2 equipos con mismo canonico), quedarse con el
            # que tenga mas jugadores (mas reciente). En caso de empate, el
            # equipo_id con orden alfabetico menor (determinista).
            else:
                existing_id = self._canonical_map[canonical]
                existing_roster = self.team_rosters.get(existing_id, [])
                current_roster = self.team_rosters.get(eid, [])
                if len(current_roster) > len(existing_roster):
                    self._canonical_map[canonical] = eid
                elif len(current_roster) == len(existing_roster) and eid < existing_id:
                    self._canonical_map[canonical] = eid

    def _resolve_team_key(self, team_name: str) -> str:
        """
        Resuelve nombre de equipo (canonico o equipo_id) a la clave
        interna de self.team_profiles / self.team_rosters.
        """
        # 1. Busqueda directa (ya es un equipo_id valido)
        if team_name in self.team_profiles:
            return team_name
        # 2. Busqueda por canonico
        if self._canonical_map is None:
            self._build_canonical_map()
        from src.data.team_mapper import normalize_team_name

        canonical = normalize_team_name(team_name)
        if canonical in self._canonical_map:
            return self._canonical_map[canonical]
        # 3. Fallback: iterar profiles
        for eid in self.team_profiles:
            if self._extract_team_name(eid) == canonical:
                return eid
        return team_name

    def fit(self, player_stats: pd.DataFrame, team_stats: pd.DataFrame):
        """
        Ajusta las distribuciones a partir de datos historicos.

        Usa la temporada mas reciente disponible para cada equipo.
        """
        # Usar la temporada mas reciente para cada equipo
        if "temporada" in player_stats.columns:
            # Ordenar por temporada descendente
            player_stats = player_stats.sort_values("temporada", ascending=False)

        # Agrupar por equipo (usar equipo_id o inferir del archivo)
        # Primero, necesitamos mapear equipo_id a nombre canonico
        teams_processed = set()

        for equipo_id in player_stats["equipo_id"].unique():
            df_equipo = player_stats[player_stats["equipo_id"] == equipo_id]

            # Tomar la temporada mas reciente
            if "temporada" in df_equipo.columns:
                latest_season = df_equipo["temporada"].iloc[0]
                df_equipo = df_equipo[df_equipo["temporada"] == latest_season]

            # Filtrar jugadores con suficientes partidos (minimo 5 sets)
            df_equipo = df_equipo[df_equipo["sets"].fillna(0) >= 5].copy()

            if len(df_equipo) == 0:
                continue

            # Extraer nombre canonico del equipo desde equipo_id
            equipo_name = self._extract_team_name(equipo_id)

            # Calcular stats per set para cada jugador
            profile = {}
            roster = []

            for _, row in df_equipo.iterrows():
                jugador = row["jugador"]
                sets_played = max(row["sets"], 1)

                stats = {}
                for stat_key in self.STAT_KEYS:
                    if stat_key in row and pd.notna(row[stat_key]):
                        per_set = row[stat_key] / sets_played
                        # Usar distribucion Poisson-like (media y std)
                        stats[stat_key] = {
                            "mean": float(per_set),
                            "std": float(max(per_set * 0.4, 0.1)),
                            "total": float(row[stat_key]),
                            "sets": int(sets_played),
                        }
                    else:
                        stats[stat_key] = {
                            "mean": 0.0,
                            "std": 0.1,
                            "total": 0.0,
                            "sets": int(sets_played),
                        }

                # Participacion (que porcentaje de sets juega)
                stats["participation"] = float(sets_played) / float(df_equipo["sets"].max() or 1)

                profile[jugador] = stats
                roster.append(jugador)

            self.team_profiles[equipo_id] = profile
            self.team_rosters[equipo_id] = roster
            teams_processed.add(equipo_name)

        print(
            f"  [PlayerStats] {len(teams_processed)} equipos procesados, "
            f"{sum(len(r) for r in self.team_rosters.values())} jugadores"
        )

    def _extract_team_name(self, equipo_id: str) -> str:
        """Extrae el nombre canonico del equipo desde el ID del CSV.

        Los IDs reales en los CSVs de stats_por_equipo_completo son:
        APG, BASTIA, CIS-VOLLEY, LT, MC, MI-POWER, MIVER, MO, PC,
        PD, PIACENZAYOU, TN-ITAS, VRI, TA, VV, etc.

        Delega a :func:`src.data.team_id_mapper.get_canonical_team`
        para mantener un unico mapa maestro de IDs.
        """
        from src.data.team_id_mapper import get_canonical_team
        from src.data.team_mapper import normalize_team_name

        name = get_canonical_team(equipo_id)
        if name is not None:
            return name

        # Fallback: intentar normalizar
        return normalize_team_name(str(equipo_id).strip())

    def generate_set_stats(
        self,
        team_name: str,
        team_score: int,
        opponent_score: int,
    ) -> list[dict]:
        """
        Genera estadisticas de jugadores para un set simulado.

        Args:
            team_name: nombre canonico del equipo
            team_score: puntos del equipo en este set
            opponent_score: puntos del rival

        Returns:
            Lista de dicts con stats por jugador
        """
        team_key = self._resolve_team_key(team_name)
        profile = self.team_profiles.get(team_key)
        roster = self.team_rosters.get(team_key)
        if profile is None or roster is None:
            return []

        # Generar stats base para cada jugador
        player_stats = []
        for jugador in roster:
            if jugador not in profile:
                continue

            p = profile[jugador]

            # Decidir si el jugador participa (basado en participation rate)
            part = p.get("participacion", 0.8)
            if isinstance(part, dict):
                part_rate = part.get("mean", 0.8)
            else:
                part_rate = part

            if np.random.random() > part_rate:
                continue

            stats = {"jugador": jugador}
            for stat_key in self.STAT_KEYS:
                if stat_key in p and isinstance(p[stat_key], dict):
                    mean = p[stat_key]["mean"]
                    std = p[stat_key]["std"]
                    # Samplear de una distribucion normal truncada en 0
                    value = max(0, int(round(np.random.normal(mean, std))))
                    stats[stat_key] = value
                else:
                    stats[stat_key] = 0

            # Formula estandar de voleibol: PTS = ataques_ganados + aces + bloqueos
            # Los errores del rival no se atribuyen a ningun jugador.
            stats["puntos"] = (
                stats.get("ataques_ganados", 0) + stats.get("aces", 0) + stats.get("bloqueos", 0)
            )

            player_stats.append(stats)

        return player_stats

    def get_roster(self, team_name: str) -> list:
        """Devuelve el roster de un equipo, buscando por nombre canonico o equipo_id."""
        key = self._resolve_team_key(team_name)
        return self.team_rosters.get(key, [])

    def get_profile(self, team_name: str) -> dict:
        """Devuelve el perfil de jugadores de un equipo."""
        key = self._resolve_team_key(team_name)
        return self.team_profiles.get(key, {})

    def save(self, path: Optional[Path] = None):
        """Guarda los perfiles de jugadores."""
        if path is None:
            path = MODELS_DIR / "player_stats_params.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "team_profiles": self.team_profiles,
            "team_rosters": self.team_rosters,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"  Perfiles guardados en {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PlayerStatsGenerator":
        """Carga perfiles previamente guardados."""
        if path is None:
            path = MODELS_DIR / "player_stats_params.json"

        with open(path, "r", encoding="utf-8") as f:
            save_data = json.load(f)

        gen = cls()
        gen.team_profiles = save_data["team_profiles"]
        gen.team_rosters = save_data["team_rosters"]
        return gen
