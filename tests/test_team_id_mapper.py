"""
Unit tests for src.data.team_id_mapper.

Covers all 22 ID_EQUIPO_MAP entries, LT/PC corrections, unknown ID
handling, and prefix/suffix fallback.
"""

from src.data.team_id_mapper import get_canonical_team, ID_EQUIPO_MAP


class TestKnownIds:
    """REQ-001: Known ID_EQUIPO codes return the correct canonical name."""

    def test_trento(self):
        assert get_canonical_team("TN-ITAS") == "Trento"

    def test_perugia(self):
        assert get_canonical_team("APG") == "Perugia"

    def test_lube(self):
        assert get_canonical_team("MC") == "Lube"

    def test_milano(self):
        assert get_canonical_team("MI-POWER") == "Milano"

    def test_verona(self):
        assert get_canonical_team("VRI") == "Verona"

    def test_monza(self):
        assert get_canonical_team("MIVER") == "Monza"

    def test_modena(self):
        assert get_canonical_team("MO") == "Modena"

    def test_piacenza(self):
        assert get_canonical_team("PIACENZAYOU") == "Piacenza"

    def test_cisterna(self):
        assert get_canonical_team("CIS-VOLLEY") == "Cisterna"

    def test_padova(self):
        assert get_canonical_team("PD") == "Padova"

    def test_taranto(self):
        assert get_canonical_team("TA") == "Taranto"

    def test_grottazzolina(self):
        assert get_canonical_team("BASTIA") == "Grottazzolina"

    def test_ravenna(self):
        assert get_canonical_team("RAV-ROB") == "Ravenna"

    def test_vibo_valentia(self):
        assert get_canonical_team("VV") == "Vibo Valentia"

    def test_sora(self):
        assert get_canonical_team("FR-SORA") == "Sora"

    def test_siena(self):
        assert get_canonical_team("SIENA-EMMAS") == "Siena"

    def test_cuneo_bam(self):
        assert get_canonical_team("BAM") == "Cuneo"

    def test_cuneo_sport(self):
        assert get_canonical_team("CUNEOSPORT") == "Cuneo"

    def test_castellana_grotte(self):
        assert get_canonical_team("CAST-MATER") == "Castellana Grotte"

    def test_acicastello(self):
        assert get_canonical_team("ACICASTELLO") == "Acicastello"


class TestCorrections:
    """REQ-002: LT/PC corrections disambiguate historically colliding codes."""

    def test_lt_maps_to_cisterna_top_volley(self):
        """LT (historical Latina) → 'Cisterna Top Volley', NOT 'Cisterna'."""
        result = get_canonical_team("LT")
        assert result == "Cisterna Top Volley", f"Got {result}"

    def test_pc_maps_to_piacenza_copra(self):
        """PC (historical Piacenza) → 'Piacenza Copra', NOT 'Piacenza'."""
        result = get_canonical_team("PC")
        assert result == "Piacenza Copra", f"Got {result}"

    def test_lt_and_cis_volley_distinct(self):
        """LT and CIS-VOLLEY are now distinct canonical entries."""
        lt = get_canonical_team("LT")
        cv = get_canonical_team("CIS-VOLLEY")
        assert lt != cv, f"LT ({lt}) and CIS-VOLLEY ({cv}) should differ"

    def test_pc_and_piacenzayou_distinct(self):
        """PC and PIACENZAYOU are now distinct canonical entries."""
        pc = get_canonical_team("PC")
        py = get_canonical_team("PIACENZAYOU")
        assert pc != py, f"PC ({pc}) and PIACENZAYOU ({py}) should differ"


class TestUnknownIds:
    """REQ-003: Unknown IDs return None, not a misleading passthrough."""

    def test_unknown_code(self):
        assert get_canonical_team("UNKNOWN") is None

    def test_empty_string(self):
        assert get_canonical_team("") is None

    def test_whitespace_only(self):
        assert get_canonical_team("   ") is None

    def test_none(self):
        """None passed as argument should return None (str(None) = 'None')."""
        assert get_canonical_team(None) is None


class TestCoverage:
    """All 22 entries in ID_EQUIPO_MAP are reachable."""

    def test_all_22_ids(self):
        """There are exactly 22 entries in the map."""
        assert len(ID_EQUIPO_MAP) == 22

    def test_every_entry_resolves(self):
        """Every key in ID_EQUIPO_MAP resolves to its own canonical name."""
        for eid, expected in ID_EQUIPO_MAP.items():
            result = get_canonical_team(eid)
            assert result == expected, f"{eid} -> {result} (expected {expected})"


class TestEdgeCases:
    """Edge cases for prefix/suffix fallback logic."""

    def test_prefix_fallback(self):
        """A value starting with a known key should resolve via prefix fallback."""
        result = get_canonical_team("MO/2025")
        # "MO" is a known key prefix
        assert result == "Modena"

    def test_suffix_fallback(self):
        """A value ending with a known key should resolve via suffix fallback."""
        result = get_canonical_team("2025_MO")
        assert result == "Modena"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before matching."""
        assert get_canonical_team("  MO  ") == "Modena"
        assert get_canonical_team("\tTN-ITAS\n") == "Trento"

    def test_case_sensitive(self):
        """Matching is case-sensitive (codes are always uppercase in CSVs)."""
        assert get_canonical_team("mo") is None  # lowercase
        assert get_canonical_team("MO") == "Modena"  # uppercase
