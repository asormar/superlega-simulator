"""
team_mapper.py — Normalización de nombres de equipos de la SuperLega italiana.

Todos los datasets usan nombres distintos para el mismo equipo.
Este módulo unifica todos los alias a un nombre canónico.
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────
# Diccionario maestro: nombre canónico → lista de alias
# ─────────────────────────────────────────────────────────────

TEAM_ALIASES = {
    "Modena": [
        "Modena",
        "Azimut Modena",
        "Azimut Leo Shoes Modena",
        "Leo Shoes Modena",
        "ModenaModena",
        "Valsa Group Modena",
        "Modena Volley",
    ],
    "Trento": [
        "Trento",
        "Trentino",
        "Diatec Trentino",
        "Itas Trentino",
        "TrentinoTrentino",
        "Trentino Volley",
    ],
    "Perugia": [
        "Perugia",
        "Sir Safety Conad Perugia",
        "Sir Safety Perugia",
        "Sir Sicoma Monini Perugia",
        "PerugiaPerugia",
        "Sir Safety Susa Perugia",
        "Sir Susa Vim Perugia",
    ],
    "Lube": [
        "Lube",
        "Lube Civitanova",
        "Cucine Lube Civitanova",
        "Lube CivitanovaLube Civitanova",
        "Lube Banca Marche Civitanova",
    ],
    "Milano": [
        "Milano",
        "Allianz Milano",
        "Powervolley Milano",
        "MilanoMilano",
        "Revivre Milano",
    ],
    "Monza": [
        "Monza",
        "Gi Group Monza",
        "Vero Volley Monza",
        "MonzaMonza",
    ],
    "Verona": [
        "Verona",
        "VeronaVerona",
        "Calzedonia Verona",
        "Rana Verona",
        "WithU Verona",
    ],
    "Padova": [
        "Padova",
        "Kioene Padova",
        "PadovaPadova",
        "Pallavolo Padova",
    ],
    "Piacenza": [
        "Piacenza",
        "Gas Sales Piacenza",
        "PiacenzaPiacenza",
    ],
    "Piacenza Copra": [
        "Piacenza Copra",
        "Copra Elior Piacenza",
    ],
    "Cisterna": [
        "Cisterna",
        "CisternaCisterna",
        "Top Volley Cisterna",
        "Cisterna Volley",
    ],
    "Cisterna Top Volley": [
        "Cisterna Top Volley",
        "Top Volley Latina",
    ],
    "Ravenna": [
        "Ravenna",
        "RavennaRavenna",
        "Consar Ravenna",
    ],
    "Vibo Valentia": [
        "Vibo Valentia",
        "Vibo ValentiaVibo Valentia",
        "Tonno Callipo Vibo Valentia",
    ],
    "Taranto": [
        "Taranto",
        "TarantoTaranto",
        "Gioiella Prisma Taranto",
    ],
    "Grottazzolina": [
        "Grottazzolina",
        "GrottazzolinaGrottazzolina",
        "Videx Grottazzolina",
        "Yuasa Battery Grottazzolina",
    ],
    "Sora": [
        "Sora",
        "Globo Banca Popolare Sora",
    ],
    "Siena": [
        "Siena",
        "Emma Villas Siena",
    ],
    "Cuneo": [
        "Cuneo",
        "BAM Acqua San Bernardo Cuneo",
    ],
    "Castellana Grotte": [
        "Castellana Grotte",
        "Castellana Grotte New Mater",
        "BCC Castellana Grotte",
    ],
    "Molfetta": [
        "Molfetta",
        "Exprivia Molfetta",
    ],
    "Acicastello": [
        "Acicastello",
        "Papiro Catania",
    ],
}

# ─────────────────────────────────────────────────────────────
# Construir lookup invertido: alias → nombre canónico
# ─────────────────────────────────────────────────────────────

_ALIAS_LOOKUP: dict[str, str] = {}


def _build_lookup():
    """Construye el diccionario de búsqueda inversa."""
    global _ALIAS_LOOKUP
    for canonical, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            _ALIAS_LOOKUP[key] = canonical


_build_lookup()


# ─────────────────────────────────────────────────────────────
# Funciones públicas
# ─────────────────────────────────────────────────────────────


def normalize_team_name(raw_name: str) -> str:
    """
    Dado un nombre de equipo (posiblemente con alias, duplicaciones,
    o espacios extra), devuelve el nombre canónico.

    Ejemplos:
        normalize_team_name("MonzaMonza")          → "Monza"
        normalize_team_name("Sir Safety Conad Perugia") → "Perugia"
        normalize_team_name("  Azimut Modena  ")   → "Modena"
        normalize_team_name("Diatec Trentino")     → "Trento"
    """
    if not raw_name or not isinstance(raw_name, str):
        return raw_name

    cleaned = raw_name.strip()

    # 1. Búsqueda directa
    key = cleaned.lower()
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    # 2. Intentar detectar nombres duplicados (e.g., "MonzaMonza")
    # Patrón: la cadena es exactamente word+word
    deduplicated = _try_dedup(cleaned)
    if deduplicated:
        key2 = deduplicated.lower()
        if key2 in _ALIAS_LOOKUP:
            return _ALIAS_LOOKUP[key2]

    # 3. Buscar por subcadena: ¿algún alias canónico está contenido?
    for canonical, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if alias.lower() in key or key in alias.lower():
                return canonical

    # 4. No encontrado — devolver tal cual (con warning implícito)
    return cleaned


def _try_dedup(name: str) -> Optional[str]:
    """
    Detecta nombres duplicados como 'MonzaMonza' → 'Monza'
    o 'Lube CivitanovaLube Civitanova' → 'Lube Civitanova'.
    """
    n = len(name)
    if n < 2:
        return None

    # Probar si la primera mitad == segunda mitad
    if n % 2 == 0:
        half = n // 2
        if name[:half] == name[half:]:
            return name[:half]

    # Probar con longitudes variables (para nombres de longitud impar
    # o con separaciones extrañas)
    for i in range(3, n - 2):
        if name[:i] == name[i:]:
            return name[:i]

    return None


def get_canonical_names() -> list[str]:
    """Devuelve la lista de todos los nombres canónicos de equipos."""
    return sorted(TEAM_ALIASES.keys())


def get_superliga_teams(season: str = "2024/2025") -> list[str]:
    """
    Devuelve los equipos de la SuperLega para una temporada dada.
    Si no se especifica, devuelve la temporada más reciente.
    """
    # Equipos por temporada (solo SuperLega, no Serie A2)
    SUPERLIGA_SEASONS = {
        "2024/2025": [
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
        ],
        "2023/2024": [
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
        ],
        "2022/2023": [
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
            "Acicastello",
        ],
        "2021/2022": [
            "Trento",
            "Perugia",
            "Piacenza",
            "Verona",
            "Lube",
            "Milano",
            "Modena",
            "Monza",
            "Cisterna Top Volley",
            "Padova",
            "Siena",
        ],
        "2020/2021": [
            "Trento",
            "Perugia",
            "Piacenza",
            "Verona",
            "Lube",
            "Milano",
            "Modena",
            "Monza",
            "Cisterna Top Volley",
            "Padova",
            "Ravenna",
            "Vibo Valentia",
            "Taranto",
        ],
        "2019/2020": [
            "Trento",
            "Perugia",
            "Piacenza",
            "Verona",
            "Lube",
            "Milano",
            "Modena",
            "Monza",
            "Cisterna Top Volley",
            "Padova",
            "Ravenna",
            "Vibo Valentia",
            "Sora",
        ],
    }
    return SUPERLIGA_SEASONS.get(season, SUPERLIGA_SEASONS["2024/2025"])


# Equipos de la temporada actual (2024/2025)
_CURRENT_SEASON_TEAMS = {
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
}

# Todos los equipos viables: tienen ≥20 partidos en match_features
# + stats de equipo + datos de jugadores
_ALL_VIABLE_TEAMS = [
    # Actuales (temporada 2024/2025)
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
    # Históricos (con datos completos)
    "Siena",
    "Ravenna",
    "Acicastello",
    "Cuneo",
]


def get_all_viable_teams() -> list[dict]:
    """
    Devuelve todos los equipos viables con su categoría.

    Solo incluye equipos con:
    - ≥20 partidos en match_features
    - Stats de equipo (Comparacion_equipos_10_años.csv)
    - Datos de jugadores (stats_por_equipo_completo)

    Returns:
        Lista de dicts: [{"nombre": str, "categoria": "actual"|"historico"}]
    """
    teams = []
    for name in _ALL_VIABLE_TEAMS:
        teams.append(
            {
                "nombre": name,
                "categoria": "actual" if name in _CURRENT_SEASON_TEAMS else "historico",
            }
        )
    return teams


if __name__ == "__main__":
    # Tests rápidos
    tests = [
        ("MonzaMonza", "Monza"),
        ("Sir Safety Conad Perugia", "Perugia"),
        ("Diatec Trentino", "Trento"),
        ("Azimut Modena", "Modena"),
        ("Lube CivitanovaLube Civitanova", "Lube"),
        ("  Kioene Padova  ", "Padova"),
        ("Gi Group Monza", "Monza"),
        ("VeronaVerona", "Verona"),
        ("GrottazzolinaGrottazzolina", "Grottazzolina"),
        ("Vibo ValentiaVibo Valentia", "Vibo Valentia"),
        ("Emma Villas Siena", "Siena"),
        ("Videx Grottazzolina", "Grottazzolina"),
    ]

    print("=" * 50)
    print("Test de normalización de nombres de equipos")
    print("=" * 50)
    all_ok = True
    for raw, expected in tests:
        result = normalize_team_name(raw)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            all_ok = False
        print(f"  {status} '{raw}' -> '{result}' (esperado: '{expected}')")

    print(f"\n{'Todos los tests pasaron!' if all_ok else 'ALGUNOS TESTS FALLARON'}")
