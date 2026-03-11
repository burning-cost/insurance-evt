"""Tests for InsurancePresets."""

import pytest
from insurance_evt import FLOOD, TPBI, SUBSIDENCE, LARGE_FIRE, STORM, get_preset, PRESETS
from insurance_evt.presets import InsurancePreset


class TestPresets:
    def test_all_presets_accessible(self):
        assert FLOOD is not None
        assert TPBI is not None
        assert SUBSIDENCE is not None
        assert LARGE_FIRE is not None
        assert STORM is not None

    def test_flood_percentile(self):
        assert FLOOD.threshold_percentile == 90.0

    def test_tpbi_percentile(self):
        assert TPBI.threshold_percentile == 95.0

    def test_flood_preferred_method(self):
        assert FLOOD.preferred_method == "gpd"

    def test_subsidence_preferred_gev(self):
        assert SUBSIDENCE.preferred_method == "gev"

    def test_xi_ranges_positive(self):
        for preset in PRESETS.values():
            lo, hi = preset.typical_xi_range
            assert lo >= 0
            assert hi > lo

    def test_tpbi_heavier_tail_than_flood(self):
        """TPBI has higher xi than flood (bodily injury vs property)."""
        assert TPBI.typical_xi_range[0] > FLOOD.typical_xi_range[0]

    def test_preset_descriptions_non_empty(self):
        for preset in PRESETS.values():
            assert len(preset.description) > 20

    def test_preset_notes_non_empty(self):
        for preset in PRESETS.values():
            assert len(preset.notes) > 20


class TestGetPreset:
    def test_flood_by_name(self):
        p = get_preset("flood")
        assert p.name == "flood"

    def test_case_insensitive(self):
        p = get_preset("FLOOD")
        assert p.name == "flood"

    def test_large_fire_with_underscore(self):
        p = get_preset("large_fire")
        assert p.name == "large_fire"

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("earthquake")

    def test_all_presets_gettable(self):
        for name in PRESETS:
            p = get_preset(name)
            assert isinstance(p, InsurancePreset)

    def test_presets_dict_has_five(self):
        assert len(PRESETS) == 5
