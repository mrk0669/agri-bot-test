"""Unit tests for the NMRSA calculation engine.

Run from the repository root:  python -m pytest nmrsa -q

The numbers checked here are hand-computed from the published expressions, so a
failure means the engine drifted from the method — not that a threshold moved.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nmrsa_engine as ng  # noqa: E402


# --------------------------------------------------------------------------
# RMR
# --------------------------------------------------------------------------

def test_basic_rmr_is_the_sum_of_five_parameters():
    rmr = ng.RMR(layer_thickness=24, structural_features=18, weatherability=14,
                 rock_strength=11, groundwater=8)
    assert rmr.basic() == 75


def test_parameter_maxima_match_the_cmri_ism_weighting():
    assert [cap for _, cap in ng.RMR_PARAMETERS.values()] == [30, 25, 20, 15, 10]
    assert sum(cap for _, cap in ng.RMR_PARAMETERS.values()) == 100


def test_adjustments_multiply_and_clamp():
    rmr = ng.RMR(30, 25, 20, 15, 10, adjustments={"depth": 0.9, "excavation": 0.8})
    assert rmr.basic() == 100
    assert rmr.adjusted() == pytest.approx(72.0)

    unadjusted = ng.RMR(30, 25, 20, 15, 10, adjustments={"depth": 2.0})
    assert unadjusted.adjusted() == 100.0  # clamped, never above 100


def test_over_rated_parameters_are_reported():
    rmr = ng.RMR(layer_thickness=35, structural_features=30)
    messages = rmr.over_rated()
    assert len(messages) == 2
    assert "maximum is 30" in messages[0]


@pytest.mark.parametrize("value,numeral", [(5, "V"), (30, "IV"), (50, "III"),
                                           (70, "II"), (90, "I"), (100, "I")])
def test_rmr_classes(value, numeral):
    assert ng.rmr_class(value)[0] == numeral


# --------------------------------------------------------------------------
# Rock load
# --------------------------------------------------------------------------

def test_rock_load_coefficient_matches_the_published_quadratic():
    # 1.7 - 0.037*40 + 0.0002*1600 = 1.7 - 1.48 + 0.32
    assert ng.rock_load_coefficient(40) == pytest.approx(0.54)
    assert ng.rock_load_coefficient(0) == pytest.approx(1.7)


def test_the_quadratic_has_roots_at_85_and_100():
    assert ng.rock_load_coefficient(85) == pytest.approx(0.0, abs=1e-9)
    assert ng.rock_load_coefficient(100) == pytest.approx(0.0, abs=1e-9)


def test_coefficient_is_floored_at_zero_between_the_roots():
    # the raw quadratic dips to about -0.011 at RMR 92.5
    assert ng.rock_load_coefficient(92.5) == 0.0


def test_rock_load_scales_with_span_and_density():
    load = ng.rock_load(span_m=4.2, density_t_m3=2.3, rmr=40)
    assert load == pytest.approx(4.2 * 2.3 * 0.54)
    assert ng.rock_load(8.4, 2.3, 40) == pytest.approx(2 * load)
    assert ng.rock_load(4.2, 4.6, 40) == pytest.approx(2 * load)


def test_rock_load_height_is_independent_of_density():
    assert ng.rock_load_height(4.2, 40) == pytest.approx(4.2 * 0.54)
    for density in (1.8, 2.3, 2.6):
        load = ng.rock_load(4.2, density, 40)
        assert load / density == pytest.approx(ng.rock_load_height(4.2, 40))


def test_good_roof_mobilises_no_rock_load():
    assert ng.rock_load(4.8, 2.4, 90) == 0.0


def test_junction_span_is_the_diagonal():
    assert ng.junction_span(4.2, 4.2) == pytest.approx(4.2 * math.sqrt(2))
    assert ng.junction_span(3.0, 4.0) == pytest.approx(5.0)


def test_a_junction_always_carries_more_load_than_its_galleries():
    for width in (3.6, 4.2, 4.8):
        gallery = ng.rock_load(width, 2.3, 45)
        junction = ng.rock_load(ng.junction_span(width, width), 2.3, 45)
        assert junction > gallery


# --------------------------------------------------------------------------
# Support design
# --------------------------------------------------------------------------

def test_support_density_and_resistance():
    support = ng.Support(bolt_capacity_t=10.0, bolts_per_row=4, row_spacing_m=1.2)
    assert support.density(4.0) == pytest.approx(4 / (4.0 * 1.2))
    assert support.resistance(4.0) == pytest.approx(10.0 * 4 / 4.8)


def test_bolts_across_respects_the_spacing_limit():
    assert ng.bolts_across(4.2) == 3       # 4.2 / 1.5 = 2.8 -> 3
    assert ng.bolts_across(5.94) == 4
    assert ng.bolts_across(1.0) == 2       # never fewer than two


def test_support_safety_factor_is_resistance_over_rock_load():
    support = ng.Support(bolt_capacity_t=10.0, bolts_per_row=4,
                         row_spacing_m=1.0, bolt_length_m=2.4)
    result = ng.design_opening("Gallery", 4.2, 2.3, 40, support, 1.5)
    expected_load = 4.2 * 2.3 * 0.54
    expected_resistance = 4 * 10.0 / (4.2 * 1.0)
    assert result.rock_load_t_m2 == pytest.approx(expected_load)
    assert result.resistance_provided_t_m2 == pytest.approx(expected_resistance)
    assert result.support_safety_factor == pytest.approx(expected_resistance / expected_load)


def test_max_row_spacing_delivers_exactly_the_required_safety_factor():
    support = ng.Support(bolt_capacity_t=10.0, bolts_per_row=4,
                         row_spacing_m=1.2, bolt_length_m=2.4)
    result = ng.design_opening("Gallery", 4.2, 2.3, 30, support, 1.5)
    tightened = ng.Support(10.0, 4, result.max_row_spacing_m, 2.4)
    rechecked = ng.design_opening("Gallery", 4.2, 2.3, 30, tightened, 1.5)
    assert rechecked.support_safety_factor == pytest.approx(1.5, rel=1e-9)


def test_row_spacing_never_exceeds_the_practice_limit():
    support = ng.Support(bolt_capacity_t=18.0, bolts_per_row=5, row_spacing_m=1.0)
    result = ng.design_opening("Gallery", 3.6, 2.0, 80, support, 1.5)
    assert result.max_row_spacing_m == ng.MAX_BOLT_SPACING_M


def test_bolt_length_covers_the_rock_load_band_plus_anchorage():
    support = ng.Support(bolt_length_m=1.5, anchorage_allowance_m=0.3)
    result = ng.design_opening("Gallery", 4.2, 2.3, 30, support, 1.5)
    assert result.bolt_length_required_m == pytest.approx(4.2 * ng.rock_load_coefficient(30) + 0.3)
    assert result.bolt_length_required_m > support.bolt_length_m
    assert not result.adequate


def test_bolt_length_never_drops_below_the_practical_minimum():
    support = ng.Support(bolt_length_m=1.8)
    result = ng.design_opening("Gallery", 4.2, 2.3, 88, support, 1.5)
    assert result.rock_load_height_m == 0.0
    assert result.bolt_length_required_m == ng.MIN_BOLT_LENGTH_M


# --------------------------------------------------------------------------
# Pillars
# --------------------------------------------------------------------------

def test_sheorey_strength_matches_a_hand_calculation():
    strength = ng.pillar_strength(coal_ucs_mpa=25.0, height_m=3.0, width_m=21.3, depth_m=150.0)
    expected = 0.27 * 25.0 * 3.0 ** -0.36 + (150.0 / 250.0) * (21.3 / 3.0 - 1.0)
    assert strength == pytest.approx(expected)


def test_tributary_stress_matches_a_hand_calculation():
    stress = ng.pillar_stress(depth_m=200.0, pillar_width_m=20.0, gallery_width_m=4.5)
    assert stress == pytest.approx(0.025 * 200.0 * (24.5 / 20.0) ** 2)


def test_stress_rises_with_depth_and_with_wider_galleries():
    base = ng.pillar_stress(150, 20, 4.2)
    assert ng.pillar_stress(300, 20, 4.2) == pytest.approx(2 * base)
    assert ng.pillar_stress(150, 20, 4.8) > base


def test_extraction_ratio():
    assert ng.extraction_ratio(20.0, 5.0) == pytest.approx(1 - (20 ** 2) / (25 ** 2))
    assert ng.extraction_ratio(0.0, 5.0) == pytest.approx(1.0)


def test_pillar_verdicts_step_through_the_thresholds():
    shallow = ng.design_pillar(30.0, 4.2, 3.0, 100.0, 25.0)
    assert shallow.factor_of_safety > 2.0
    assert shallow.verdict.startswith("Stable")

    deep = ng.design_pillar(20.0, 4.2, 3.0, 500.0, 20.0)
    assert deep.factor_of_safety < 1.5


# --------------------------------------------------------------------------
# Statutory reference data
# --------------------------------------------------------------------------

def test_statutory_pillar_centres_lookup():
    assert ng.statutory_pillar_centre(50.0, 4.2) == 18.0
    assert ng.statutory_pillar_centre(150.0, 4.2) == 22.5
    assert ng.statutory_pillar_centre(400.0, 4.8) == 58.0
    assert ng.statutory_pillar_centre(150.0, 5.5) is None   # outside the table


def test_pillar_centres_increase_with_depth_and_gallery_width():
    for column, width in enumerate(ng.GALLERY_WIDTH_COLUMNS):
        values = [row[column] for _, row in ng.PILLAR_CENTRES_TABLE]
        assert values == sorted(values)
    for _, row in ng.PILLAR_CENTRES_TABLE:
        assert list(row) == sorted(row)


# --------------------------------------------------------------------------
# Whole design
# --------------------------------------------------------------------------

def _fair_roof_design(**overrides):
    rmr = ng.RMR(14, 12, 10, 6, 5)          # basic 47
    geometry = ng.Geometry(**overrides.pop("geometry", {}))
    support = ng.Support(**overrides.pop("support", {}))
    return ng.analyse(rmr, geometry, support, **overrides)


def test_analyse_produces_both_openings_and_the_checks():
    result = _fair_roof_design()
    assert result.rmr_basic == 47
    assert result.gallery.name == "Gallery"
    assert result.junction.span_m > result.gallery.span_m
    assert result.junction.rock_load_t_m2 > result.gallery.rock_load_t_m2
    assert any(c.code == "CMR 111" for c in result.checks)


def test_wide_gallery_fails_the_statutory_check():
    result = _fair_roof_design(geometry={"gallery_width_m": 5.4, "crossing_width_m": 5.4})
    width_check = next(c for c in result.checks if c.title == "Gallery width")
    assert width_check.status == "fail"
    assert result.verdict == "Not adequate"


def test_a_strong_pattern_in_good_roof_passes_everything():
    rmr = ng.RMR(24, 20, 16, 12, 8)          # basic 80, no rock load band
    geometry = ng.Geometry(4.2, 3.0, 4.2, 120.0, 2.3)
    support = ng.Support(bolt_capacity_t=10.0, bolts_per_row=4,
                         row_spacing_m=1.0, bolt_length_m=1.8)
    result = ng.analyse(rmr, geometry, support, pillar_centre_m=30.0)
    assert result.rmr_class == "I"
    assert result.verdict == "Adequate"
    assert all(c.status == "pass" for c in result.checks)


def test_junction_gets_at_least_as_many_bolts_per_row_as_the_gallery():
    result = _fair_roof_design(support={"bolts_per_row": 3})
    assert result.junction.bolts_per_row >= result.gallery.bolts_per_row
    assert result.junction.span_m / result.junction.bolts_per_row <= ng.MAX_BOLT_SPACING_M


def test_recommended_pattern_is_compliant_when_applied():
    pattern = ng.recommended_pattern(span_m=4.2, density_t_m3=2.3, rmr=45,
                                     bolt_capacity_t=10.0, safety_factor=1.5)
    support = ng.Support(10.0, int(pattern["bolts_per_row"]),
                         pattern["row_spacing_m"], pattern["bolt_length_m"])
    result = ng.design_opening("Gallery", 4.2, 2.3, 45, support, 1.5)
    assert result.support_safety_factor >= 1.5
    assert support.bolt_length_m >= result.bolt_length_required_m


def test_json_round_trip_of_a_result():
    result = _fair_roof_design(pillar_centre_m=25.5)
    payload = result.as_dict()
    assert payload["rmr_class_label"] == "Fair"
    assert payload["pillar"]["factor_of_safety"] > 0
    assert isinstance(payload["checks"], list)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_unit_conversions():
    assert ng.to_imperial(1.0, "length") == pytest.approx(3.2808, rel=1e-4)
    assert ng.to_imperial(1.0, "pressure") == pytest.approx(145.038, rel=1e-4)
    assert ng.to_imperial(1.0, "density") == pytest.approx(62.428, rel=1e-4)


def test_cli_runs_and_reports(capsys):
    assert ng._cli(["--rmr", "45", "--width", "4.2", "--pillar-centre", "25.5"]) == 0
    out = capsys.readouterr().out
    assert "CMRI-ISM" in out
    assert "Verdict:" in out


def test_a_junction_may_carry_its_own_row_spacing():
    rmr = ng.RMR(14, 12, 10, 6, 5)
    geometry = ng.Geometry(4.2, 3.0, 4.2, 150.0, 2.3)
    support = ng.Support(10.0, 4, 1.2, 3.0)
    loose = ng.analyse(rmr, geometry, support)
    tight = ng.analyse(rmr, geometry, support, junction_row_spacing_m=0.6)
    assert tight.junction.row_spacing_m == 0.6
    assert loose.gallery.row_spacing_m == 1.2
    assert tight.junction.support_safety_factor > loose.junction.support_safety_factor


def test_impractical_junction_spacing_raises_a_supplementary_support_warning():
    rmr = ng.RMR(6, 5, 4, 3, 2)          # basic 20, very poor roof
    geometry = ng.Geometry(4.8, 3.0, 4.8, 300.0, 2.5)
    support = ng.Support(10.0, 4, 1.0, 3.0)
    result = ng.analyse(rmr, geometry, support)
    warning = next(c for c in result.checks if c.title == "Junction support")
    assert warning.status == "warn"
    assert "cable bolts" in warning.detail
