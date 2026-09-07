"""
team_id_mapper.py — Mapping of ID_EQUIPO codes to canonical team names.

The raw CSVs in DB/stats_por_equipo_completo/ use short opaque ID codes
as their ID_EQUIPO column (e.g., MO, TN-ITAS, BASTIA). This module maps
those codes to the canonical team names used throughout the rest of the
codebase, with corrections for historically ambiguous codes:

  - LT → 'Cisterna Top Volley' (historical Latina team; not the modern
    Cisterna which uses CIS-VOLLEY)
  - PC → 'Piacenza Copra' (historical Piacenza team; not the modern
    Piacenza which uses PIACENZAYOU)

Usage:
    from src.data.team_id_mapper import get_canonical_team
    name = get_canonical_team("LT")       # → "Cisterna Top Volley"
    name = get_canonical_team("TN-ITAS")  # → "Trento"
    name = get_canonical_team("UNKNOWN")  # → None
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────
# Master mapping: ID_EQUIPO → canonical team name
# ─────────────────────────────────────────────────────────────
# All 22 distinct ID_EQUIPO values present across the 10-season
# DB/stats_por_equipo_completo/ dataset.
#
# LT→'Cisterna Top Volley' and PC→'Piacenza Copra' are deliberate
# disambiguations: in the raw CSVs, LT and CIS-VOLLEY both appeared
# as "Cisterna", and PC and PIACENZAYOU both as "Piacenza". The
# canonical names match the TEAM_ALIASES entries in team_mapper.py.

ID_EQUIPO_MAP: dict[str, str] = {
    # Current-season teams (2024/2025+)
    "TN-ITAS": "Trento",
    "APG": "Perugia",
    "MC": "Lube",
    "MI-POWER": "Milano",
    "VRI": "Verona",
    "MIVER": "Monza",
    "MO": "Modena",
    "PIACENZAYOU": "Piacenza",
    "CIS-VOLLEY": "Cisterna",
    "PD": "Padova",
    "TA": "Taranto",
    "BASTIA": "Grottazzolina",
    # Historical teams
    "LT": "Cisterna Top Volley",
    "PC": "Piacenza Copra",
    "RAV-ROB": "Ravenna",
    "VV": "Vibo Valentia",
    "FR-SORA": "Sora",
    "SIENA-EMMAS": "Siena",
    "BAM": "Cuneo",
    "CUNEOSPORT": "Cuneo",
    "CAST-MATER": "Castellana Grotte",
    "ACICASTELLO": "Acicastello",
}


def get_canonical_team(id_equipo: str) -> Optional[str]:
    """Map an ID_EQUIPO code to its canonical team name.

    Performs exact match first, then prefix/suffix fallback for
    edge cases where the input has extra characters (e.g. from
    concatenated CSV fields).

    Args:
        id_equipo: The raw ID_EQUIPO value from the CSV.

    Returns:
        Canonical team name, or None if no mapping exists.
    """
    eid = str(id_equipo).strip()
    if not eid:
        return None

    # Exact match first
    if eid in ID_EQUIPO_MAP:
        return ID_EQUIPO_MAP[eid]

    # Prefix/suffix fallback for edge cases
    for key, name in ID_EQUIPO_MAP.items():
        if eid.startswith(key) or eid.endswith(key):
            return name

    return None
