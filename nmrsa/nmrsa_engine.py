"""NMRSA calculation engine — Indian coal-mine roof support design.

Reference implementation of every number the browser tool (``index.html``)
shows.  The two implementations are kept deliberately in step so the JavaScript
can be regression-checked against a testable Python core (Phase 5 of the
proposal, "outputs match known, safe mathematical proofs").

Method basis — Indian practice
------------------------------
* **CMRI-ISM RMR** (Venkateswarlu, 1986) — five parameters summed to a basic
  rating out of 100, then multiplied by situation adjustment factors.
* **Rock load** ``P = W x g x (1.7 - 0.037 RMR + 0.0002 RMR^2)`` in t/m^2,
  where ``W`` is the span in m and ``g`` the roof rock density in t/m^3.
* **Junction** — the same expression evaluated on the diagonal span of the
  crossing, ``sqrt(W1^2 + W2^2)``.
* **Pillars** — CIMFR/Sheorey strength with tributary-area loading.

Every constant that comes from a statute or a published table is tagged in
``SOURCES`` so a reviewer can check it against the original document.  Nothing
here replaces a Systematic Support Rule approved under the Coal Mines
Regulations; it is a design aid.

Units are metric throughout (m, t, t/m^3, t/m^2, MPa).  ``convert`` handles the
Imperial display layer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

__version__ = "1.0.0"

G = 9.80665  # tonne-force -> kN

SOURCES = {
    "rmr": "Venkateswarlu, V. (1986) — CMRI-ISM Rock Mass Rating for Indian coal measures.",
    "rockload": "P = W.g.(1.7 - 0.037 RMR + 0.0002 RMR^2), t/m^2 — CMRI-ISM rock load "
                "estimation; see Paul et al. (2014), J. Mining 618719.",
    "pillar": "Sheorey / CIMFR pillar strength: S = 0.27 sc h^-0.36 + (H/250)(w/h - 1), MPa.",
    "stress": "Tributary area: sp = 0.025 H ((w+B)/w)^2 MPa, taking 25 kN/m^3 overburden.",
    "cmr": "Coal Mines Regulations 2017, Reg. 111 — gallery width/height limits and minimum "
           "pillar centres. REPRODUCED FOR REFERENCE — verify against the Gazette text.",
}

# --------------------------------------------------------------------------
# CMRI-ISM RMR
# --------------------------------------------------------------------------

#: parameter key -> (display name, maximum rating)
RMR_PARAMETERS: Dict[str, Tuple[str, int]] = {
    "layer_thickness": ("Layer thickness", 30),
    "structural_features": ("Structural features", 25),
    "weatherability": ("Weatherability", 20),
    "rock_strength": ("Strength of roof rock", 15),
    "groundwater": ("Ground water seepage", 10),
}

#: adjustment factors applied multiplicatively to the basic RMR.  Defaults are
#: 1.00 (no adjustment) — the site values belong to the geotechnical report,
#: they are not fixed by any regulation.
RMR_ADJUSTMENTS: Tuple[str, ...] = (
    "depth",             # situation of the working / depth of cover
    "lateral_stress",    # horizontal (tectonic) stress
    "induced_stress",    # stress induced by adjacent / superjacent workings
    "excavation",        # method of excavation (blasting damage)
    "span",              # gallery span relative to the rated condition
)

RMR_CLASSES: Tuple[Tuple[float, float, str, str], ...] = (
    (0, 20, "V", "Very poor"),
    (20, 40, "IV", "Poor"),
    (40, 60, "III", "Fair"),
    (60, 80, "II", "Good"),
    (80, 100.0001, "I", "Very good"),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class RMR:
    """CMRI-ISM rock mass rating for one roof horizon."""

    layer_thickness: float = 12.0
    structural_features: float = 12.0
    weatherability: float = 10.0
    rock_strength: float = 7.0
    groundwater: float = 5.0
    adjustments: Dict[str, float] = field(default_factory=dict)

    def basic(self) -> float:
        total = sum(getattr(self, key) for key in RMR_PARAMETERS)
        return clamp(total, 0.0, 100.0)

    def factor(self) -> float:
        product = 1.0
        for key in RMR_ADJUSTMENTS:
            product *= float(self.adjustments.get(key, 1.0))
        return product

    def adjusted(self) -> float:
        return clamp(self.basic() * self.factor(), 0.0, 100.0)

    def over_rated(self) -> List[str]:
        """Sub-ratings that exceed the CMRI-ISM maximum for that parameter."""
        bad = []
        for key, (name, cap) in RMR_PARAMETERS.items():
            if getattr(self, key) > cap:
                bad.append(f"{name} rated {getattr(self, key):g}, maximum is {cap}")
        return bad


def rmr_class(rmr: float) -> Tuple[str, str]:
    for low, high, numeral, label in RMR_CLASSES:
        if low <= rmr < high:
            return numeral, label
    return "I", "Very good"


# --------------------------------------------------------------------------
# Rock load
# --------------------------------------------------------------------------

def rock_load_coefficient(rmr: float) -> float:
    """The bracket of the CMRI-ISM rock load expression, clamped at zero.

    ``1.7 - 0.037 RMR + 0.0002 RMR^2`` has roots at RMR = 85 and RMR = 100 and
    dips marginally negative between them, so the value is floored at 0.  A
    rating above 85 therefore predicts no mobilised rock load and the support
    is set by the statutory minimum, not by this equation.
    """
    k = 1.7 - 0.037 * rmr + 0.0002 * rmr * rmr
    return max(k, 0.0)


def rock_load(span_m: float, density_t_m3: float, rmr: float) -> float:
    """Expected rock load in t/m^2 over an opening of the given span."""
    return span_m * density_t_m3 * rock_load_coefficient(rmr)


def rock_load_height(span_m: float, rmr: float) -> float:
    """Height of the de-stressed roof band in m (rock load / density)."""
    return span_m * rock_load_coefficient(rmr)


def junction_span(width_a_m: float, width_b_m: float) -> float:
    """Diagonal span of a four-way junction of two galleries."""
    return math.hypot(width_a_m, width_b_m)


# --------------------------------------------------------------------------
# Support design
# --------------------------------------------------------------------------

#: Indicative full-column-grouted bolt capacities used in Indian collieries.
#: ``capacity_t`` is the *design* anchorage capacity, i.e. what the pull-out
#: test is expected to hold, not the bar's ultimate yield.
BOLT_TYPES: Tuple[Dict[str, object], ...] = (
    {"id": "resin22", "label": "22 mm full-column resin-grouted, Fe 500", "capacity_t": 10.0},
    {"id": "resin20", "label": "20 mm full-column resin-grouted, Fe 500", "capacity_t": 8.0},
    {"id": "cement22", "label": "22 mm cement-capsule grouted", "capacity_t": 8.0},
    {"id": "mech", "label": "Point-anchored mechanical (expansion shell)", "capacity_t": 5.0},
    {"id": "cable", "label": "Flexible cable bolt, 15.2 mm strand", "capacity_t": 18.0},
)

MAX_BOLT_SPACING_M = 1.5   # across a row and between rows, common SSR practice
MIN_BOLT_LENGTH_M = 1.2    # shortest bolt normally accepted in a gallery
DEFAULT_ANCHORAGE_M = 0.3  # grout length required above the rock-load band


@dataclass
class Support:
    """A candidate bolting pattern."""

    bolt_capacity_t: float = 10.0
    bolts_per_row: int = 4
    row_spacing_m: float = 1.2
    bolt_length_m: float = 1.8
    anchorage_allowance_m: float = DEFAULT_ANCHORAGE_M

    def density(self, width_m: float) -> float:
        """Bolts per square metre of roof."""
        if width_m <= 0 or self.row_spacing_m <= 0:
            return 0.0
        return self.bolts_per_row / (width_m * self.row_spacing_m)

    def resistance(self, width_m: float) -> float:
        """Support resistance offered, t/m^2."""
        return self.density(width_m) * self.bolt_capacity_t


@dataclass
class Geometry:
    gallery_width_m: float = 4.2
    gallery_height_m: float = 3.0
    crossing_width_m: float = 4.2
    depth_m: float = 150.0
    roof_density_t_m3: float = 2.3


@dataclass
class OpeningResult:
    """Everything computed for one opening (a gallery or a junction)."""

    name: str
    span_m: float
    bolts_per_row: int
    row_spacing_m: float
    rock_load_t_m2: float
    rock_load_height_m: float
    safety_factor_required: float
    resistance_required_t_m2: float
    bolt_density_required: float
    max_row_spacing_m: float
    resistance_provided_t_m2: float
    support_safety_factor: float
    bolt_length_required_m: float
    adequate: bool

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def bolts_across(span_m: float, max_spacing_m: float = MAX_BOLT_SPACING_M) -> int:
    """Fewest bolts that span an opening without exceeding the spacing limit."""
    if span_m <= 0:
        return 0
    return max(2, math.ceil(span_m / max_spacing_m))


def design_opening(
    name: str,
    span_m: float,
    density_t_m3: float,
    rmr: float,
    support: Support,
    safety_factor: float,
    bolts_per_row: Optional[int] = None,
    row_spacing_m: Optional[float] = None,
) -> OpeningResult:
    """Design (and check) the bolting for a single span.

    ``bolts_per_row`` defaults to the pattern's own value; a junction is wider
    than the gallery it is formed from, so it is passed the larger number of
    bolts that its diagonal span will take at the 1.5 m spacing limit.
    ``row_spacing_m`` lets a junction carry its own, closer, row spacing — the
    way a Systematic Support Rule writes it.
    """
    n_row = support.bolts_per_row if bolts_per_row is None else bolts_per_row
    spacing = support.row_spacing_m if row_spacing_m is None else row_spacing_m
    load = rock_load(span_m, density_t_m3, rmr)
    height = rock_load_height(span_m, rmr)
    required = load * safety_factor
    n_required = required / support.bolt_capacity_t if support.bolt_capacity_t > 0 else 0.0
    if n_required > 0 and span_m > 0:
        max_row_spacing = n_row / (n_required * span_m)
    else:
        max_row_spacing = MAX_BOLT_SPACING_M
    max_row_spacing = min(max_row_spacing, MAX_BOLT_SPACING_M)

    provided = (n_row / (span_m * spacing)) * support.bolt_capacity_t \
        if span_m > 0 and spacing > 0 else 0.0
    ssf = provided / load if load > 0 else math.inf
    length_required = max(height + support.anchorage_allowance_m, MIN_BOLT_LENGTH_M)

    return OpeningResult(
        name=name,
        span_m=span_m,
        bolts_per_row=n_row,
        row_spacing_m=spacing,
        rock_load_t_m2=load,
        rock_load_height_m=height,
        safety_factor_required=safety_factor,
        resistance_required_t_m2=required,
        bolt_density_required=n_required,
        max_row_spacing_m=max_row_spacing,
        resistance_provided_t_m2=provided,
        support_safety_factor=ssf,
        bolt_length_required_m=length_required,
        adequate=(ssf >= safety_factor and support.bolt_length_m >= length_required),
    )


# --------------------------------------------------------------------------
# Pillars — CIMFR / Sheorey
# --------------------------------------------------------------------------

OVERBURDEN_MPA_PER_M = 0.025  # 25 kN/m^3


def pillar_strength(coal_ucs_mpa: float, height_m: float, width_m: float, depth_m: float) -> float:
    """Sheorey (CIMFR) in-situ pillar strength, MPa."""
    if height_m <= 0 or width_m <= 0:
        return 0.0
    return (0.27 * coal_ucs_mpa * height_m ** -0.36
            + (depth_m / 250.0) * (width_m / height_m - 1.0))


def pillar_stress(depth_m: float, pillar_width_m: float, gallery_width_m: float) -> float:
    """Tributary-area pillar stress, MPa."""
    if pillar_width_m <= 0:
        return math.inf
    ratio = (pillar_width_m + gallery_width_m) / pillar_width_m
    return OVERBURDEN_MPA_PER_M * depth_m * ratio * ratio


def extraction_ratio(pillar_width_m: float, gallery_width_m: float) -> float:
    centre = pillar_width_m + gallery_width_m
    if centre <= 0:
        return 0.0
    return 1.0 - (pillar_width_m ** 2) / (centre ** 2)


@dataclass
class PillarResult:
    pillar_width_m: float
    centre_distance_m: float
    strength_mpa: float
    stress_mpa: float
    factor_of_safety: float
    extraction_ratio: float
    verdict: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def design_pillar(
    centre_distance_m: float,
    gallery_width_m: float,
    working_height_m: float,
    depth_m: float,
    coal_ucs_mpa: float,
) -> PillarResult:
    width = centre_distance_m - gallery_width_m
    strength = pillar_strength(coal_ucs_mpa, working_height_m, width, depth_m)
    stress = pillar_stress(depth_m, width, gallery_width_m)
    fos = strength / stress if stress > 0 else 0.0
    if fos >= 2.0:
        verdict = "Stable for long-term standing"
    elif fos >= 1.5:
        verdict = "Adequate for development"
    elif fos >= 1.0:
        verdict = "Marginal — redesign or reduce gallery width"
    else:
        verdict = "Unsafe — pillar overloaded"
    return PillarResult(
        pillar_width_m=width,
        centre_distance_m=centre_distance_m,
        strength_mpa=strength,
        stress_mpa=stress,
        factor_of_safety=fos,
        extraction_ratio=extraction_ratio(width, gallery_width_m),
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# Statutory reference data — Coal Mines Regulations 2017
# --------------------------------------------------------------------------
# REPRODUCED FOR REFERENCE ONLY.  The Gazette text of Reg. 111 governs; check
# these figures against it (and against any DGMS circular in force) before a
# Systematic Support Rule is drawn up on them.

MAX_GALLERY_WIDTH_M = 4.8
MAX_GALLERY_HEIGHT_M = 3.0

#: depth band upper bound (m, ``None`` = no limit) -> minimum distance between
#: centres of adjacent pillars for gallery widths 3.0 / 3.6 / 4.2 / 4.8 m.
PILLAR_CENTRES_TABLE: Tuple[Tuple[Optional[float], Tuple[float, float, float, float]], ...] = (
    (60.0, (12.0, 15.0, 18.0, 19.5)),
    (90.0, (13.5, 16.5, 19.5, 21.0)),
    (150.0, (16.5, 19.0, 22.5, 25.5)),
    (240.0, (22.0, 25.5, 30.5, 34.5)),
    (360.0, (28.5, 34.0, 39.5, 45.0)),
    (None, (39.0, 45.0, 51.5, 58.0)),
)

GALLERY_WIDTH_COLUMNS = (3.0, 3.6, 4.2, 4.8)


def statutory_pillar_centre(depth_m: float, gallery_width_m: float) -> Optional[float]:
    """Minimum pillar centre distance from the CMR 2017 Reg. 111 table.

    Returns ``None`` when the gallery width exceeds the widest tabulated
    column, in which case the Regional Inspector's permission governs.
    """
    column = None
    for index, width in enumerate(GALLERY_WIDTH_COLUMNS):
        if gallery_width_m <= width + 1e-9:
            column = index
            break
    if column is None:
        return None
    for upper, row in PILLAR_CENTRES_TABLE:
        if upper is None or depth_m <= upper:
            return row[column]
    return None


@dataclass
class Check:
    code: str
    title: str
    status: str   # "pass" | "warn" | "fail"
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def compliance_checks(
    geometry: Geometry,
    support: Support,
    gallery: OpeningResult,
    junction: OpeningResult,
    pillar: Optional[PillarResult] = None,
) -> List[Check]:
    checks: List[Check] = []

    w = geometry.gallery_width_m
    if w <= MAX_GALLERY_WIDTH_M + 1e-9:
        checks.append(Check("CMR 111", "Gallery width", "pass",
                            f"{w:.2f} m is within the 4.8 m limit."))
    else:
        checks.append(Check("CMR 111", "Gallery width", "fail",
                            f"{w:.2f} m exceeds 4.8 m — written permission of the "
                            "Regional Inspector is required."))

    h = geometry.gallery_height_m
    if h <= MAX_GALLERY_HEIGHT_M + 1e-9:
        checks.append(Check("CMR 111", "Gallery height", "pass",
                            f"{h:.2f} m is within the 3.0 m limit."))
    else:
        checks.append(Check("CMR 111", "Gallery height", "fail",
                            f"{h:.2f} m exceeds 3.0 m — written permission is required."))

    spacing_across = w / gallery.bolts_per_row if gallery.bolts_per_row else math.inf
    if spacing_across <= MAX_BOLT_SPACING_M + 1e-9:
        checks.append(Check("SSR", "Bolt spacing across the row", "pass",
                            f"{spacing_across:.2f} m centre-to-centre."))
    else:
        checks.append(Check("SSR", "Bolt spacing across the row", "fail",
                            f"{spacing_across:.2f} m exceeds the 1.5 m normally accepted — "
                            "add a bolt to the row."))

    widest_rows = max(gallery.row_spacing_m, junction.row_spacing_m)
    if widest_rows <= MAX_BOLT_SPACING_M + 1e-9:
        checks.append(Check("SSR", "Row spacing", "pass",
                            f"{gallery.row_spacing_m:.2f} m in the gallery, "
                            f"{junction.row_spacing_m:.2f} m at the junction."))
    else:
        checks.append(Check("SSR", "Row spacing", "fail",
                            f"{widest_rows:.2f} m between rows exceeds 1.5 m."))

    for result in (gallery, junction):
        if result.support_safety_factor >= result.safety_factor_required:
            checks.append(Check("Design", f"Support safety factor — {result.name}", "pass",
                                f"SSF {result.support_safety_factor:.2f} against a required "
                                f"{result.safety_factor_required:.2f}."))
        else:
            checks.append(Check("Design", f"Support safety factor — {result.name}", "fail",
                                f"SSF {result.support_safety_factor:.2f} is below the required "
                                f"{result.safety_factor_required:.2f} — close the rows to "
                                f"{result.max_row_spacing_m:.2f} m or use a stronger bolt."))

    if junction.max_row_spacing_m < 0.75:
        checks.append(Check("Practice", "Junction support", "warn",
                            f"Rows at {junction.max_row_spacing_m:.2f} m are impractical to "
                            "install — carry the junction on cable bolts, W-straps or breaker "
                            "props in addition to the roof bolts."))

    longest = max(gallery.bolt_length_required_m, junction.bolt_length_required_m)
    if support.bolt_length_m >= longest:
        checks.append(Check("Design", "Bolt length", "pass",
                            f"{support.bolt_length_m:.2f} m anchors "
                            f"{support.bolt_length_m - junction.rock_load_height_m:.2f} m above "
                            "the junction rock-load band."))
    else:
        checks.append(Check("Design", "Bolt length", "fail",
                            f"{support.bolt_length_m:.2f} m is shorter than the "
                            f"{longest:.2f} m needed to anchor above the rock-load band."))

    if support.bolt_length_m > geometry.gallery_height_m:
        checks.append(Check("Practice", "Drilling clearance", "warn",
                            f"A {support.bolt_length_m:.2f} m bolt in a "
                            f"{geometry.gallery_height_m:.2f} m gallery needs a jointed rod or a "
                            "flexible bolt."))

    if pillar is not None:
        statutory = statutory_pillar_centre(geometry.depth_m, w)
        if statutory is None:
            checks.append(Check("CMR 111", "Minimum pillar centres", "warn",
                                "Gallery width is outside the tabulated columns — the Regional "
                                "Inspector's permission governs the pillar size."))
        elif pillar.centre_distance_m >= statutory - 1e-9:
            checks.append(Check("CMR 111", "Minimum pillar centres", "pass",
                                f"{pillar.centre_distance_m:.1f} m meets the tabulated "
                                f"{statutory:.1f} m for {geometry.depth_m:.0f} m of cover."))
        else:
            checks.append(Check("CMR 111", "Minimum pillar centres", "fail",
                                f"{pillar.centre_distance_m:.1f} m is below the tabulated "
                                f"{statutory:.1f} m for {geometry.depth_m:.0f} m of cover."))

        if pillar.factor_of_safety >= 1.5:
            checks.append(Check("Design", "Pillar factor of safety", "pass",
                                f"FoS {pillar.factor_of_safety:.2f} — {pillar.verdict.lower()}."))
        else:
            checks.append(Check("Design", "Pillar factor of safety", "fail",
                                f"FoS {pillar.factor_of_safety:.2f} — {pillar.verdict.lower()}."))

    return checks


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

UNIT_FACTORS = {
    # metric value -> imperial display value
    "length": (3.280839895, "m", "ft"),
    "pressure": (145.0377, "MPa", "psi"),
    "load": (0.204816, "t/m²", "kip/ft²"),   # 1 tf/m^2 = 0.2048 kip... see note
    "force": (2.204622622, "t", "kip"),
    "density": (62.42796 / 1.0, "t/m³", "lb/ft³"),
}


def to_imperial(value: float, kind: str) -> float:
    return value * UNIT_FACTORS[kind][0]


# --------------------------------------------------------------------------
# Whole-design entry point
# --------------------------------------------------------------------------

@dataclass
class DesignResult:
    rmr_basic: float
    rmr_adjusted: float
    rmr_class: str
    rmr_class_label: str
    gallery: OpeningResult
    junction: OpeningResult
    pillar: Optional[PillarResult]
    checks: List[Check]

    @property
    def verdict(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "Not adequate"
        if any(c.status == "warn" for c in self.checks):
            return "Adequate with cautions"
        return "Adequate"

    def as_dict(self) -> Dict[str, object]:
        return {
            "rmr_basic": self.rmr_basic,
            "rmr_adjusted": self.rmr_adjusted,
            "rmr_class": self.rmr_class,
            "rmr_class_label": self.rmr_class_label,
            "verdict": self.verdict,
            "gallery": self.gallery.as_dict(),
            "junction": self.junction.as_dict(),
            "pillar": self.pillar.as_dict() if self.pillar else None,
            "checks": [c.as_dict() for c in self.checks],
        }


def analyse(
    rmr: RMR,
    geometry: Geometry,
    support: Support,
    safety_factor_gallery: float = 1.5,
    safety_factor_junction: float = 2.0,
    pillar_centre_m: Optional[float] = None,
    working_height_m: Optional[float] = None,
    coal_ucs_mpa: float = 25.0,
    junction_bolts_per_row: Optional[int] = None,
    junction_row_spacing_m: Optional[float] = None,
) -> DesignResult:
    """Run the whole design: rock load, bolting, pillar and compliance."""
    adjusted = rmr.adjusted()
    numeral, label = rmr_class(adjusted)

    gallery = design_opening(
        "Gallery", geometry.gallery_width_m, geometry.roof_density_t_m3,
        adjusted, support, safety_factor_gallery,
    )
    span_j = junction_span(geometry.gallery_width_m, geometry.crossing_width_m)
    junction = design_opening(
        "Junction", span_j, geometry.roof_density_t_m3,
        adjusted, support, safety_factor_junction,
        bolts_per_row=junction_bolts_per_row or max(support.bolts_per_row, bolts_across(span_j)),
        row_spacing_m=junction_row_spacing_m,
    )

    pillar = None
    if pillar_centre_m:
        pillar = design_pillar(
            pillar_centre_m,
            geometry.gallery_width_m,
            working_height_m or geometry.gallery_height_m,
            geometry.depth_m,
            coal_ucs_mpa,
        )

    checks = compliance_checks(geometry, support, gallery, junction, pillar)
    return DesignResult(
        rmr_basic=rmr.basic(),
        rmr_adjusted=adjusted,
        rmr_class=numeral,
        rmr_class_label=label,
        gallery=gallery,
        junction=junction,
        pillar=pillar,
        checks=checks,
    )


def recommended_pattern(
    span_m: float,
    density_t_m3: float,
    rmr: float,
    bolt_capacity_t: float,
    safety_factor: float,
    bolts_per_row: Optional[int] = None,
) -> Dict[str, float]:
    """Smallest compliant pattern for a span: row spacing rounded down to 0.1 m."""
    bolts_per_row = bolts_per_row or bolts_across(span_m)
    load = rock_load(span_m, density_t_m3, rmr)
    required = load * safety_factor
    if required <= 0:
        spacing = MAX_BOLT_SPACING_M
    else:
        spacing = bolts_per_row * bolt_capacity_t / (required * span_m)
    spacing = min(spacing, MAX_BOLT_SPACING_M)
    spacing = math.floor(spacing * 10) / 10 if spacing > 0 else 0.0
    length = max(rock_load_height(span_m, rmr) + DEFAULT_ANCHORAGE_M, MIN_BOLT_LENGTH_M)
    return {
        "bolts_per_row": bolts_per_row,
        "row_spacing_m": spacing,
        "bolt_length_m": math.ceil(length * 10) / 10,
        "rock_load_t_m2": load,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nmrsa",
        description="Roof support design for Indian coal mines (CMRI-ISM RMR).",
    )
    parser.add_argument("--rmr", type=float, required=True, help="adjusted CMRI-ISM RMR, 0-100")
    parser.add_argument("--width", type=float, default=4.2, help="gallery width, m")
    parser.add_argument("--height", type=float, default=3.0, help="gallery height, m")
    parser.add_argument("--crossing", type=float, default=None, help="crossing gallery width, m")
    parser.add_argument("--depth", type=float, default=150.0, help="depth of cover, m")
    parser.add_argument("--density", type=float, default=2.3, help="roof density, t/m3")
    parser.add_argument("--bolt-capacity", type=float, default=10.0, help="per bolt, tonnes")
    parser.add_argument("--bolts-per-row", type=int, default=4)
    parser.add_argument("--row-spacing", type=float, default=1.2, help="m")
    parser.add_argument("--junction-row-spacing", type=float, default=None, help="m")
    parser.add_argument("--bolt-length", type=float, default=1.8, help="m")
    parser.add_argument("--pillar-centre", type=float, default=None, help="m, enables the pillar check")
    parser.add_argument("--coal-ucs", type=float, default=25.0, help="MPa, 25 mm cube")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args(argv)

    rmr = RMR(layer_thickness=args.rmr, structural_features=0, weatherability=0,
              rock_strength=0, groundwater=0)
    geometry = Geometry(args.width, args.height, args.crossing or args.width,
                        args.depth, args.density)
    support = Support(args.bolt_capacity, args.bolts_per_row, args.row_spacing, args.bolt_length)
    result = analyse(rmr, geometry, support, pillar_centre_m=args.pillar_centre,
                     coal_ucs_mpa=args.coal_ucs,
                     junction_row_spacing_m=args.junction_row_spacing)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print(f"NMRSA {__version__} — CMRI-ISM roof support design")
    print(f"  RMR (adjusted)        {result.rmr_adjusted:6.1f}   class {result.rmr_class} "
          f"({result.rmr_class_label})")
    for opening in (result.gallery, result.junction):
        print(f"\n  {opening.name} — span {opening.span_m:.2f} m, "
              f"{opening.bolts_per_row} bolts per row at {opening.row_spacing_m:.2f} m")
        print(f"    rock load           {opening.rock_load_t_m2:6.2f} t/m2 "
              f"({opening.rock_load_height_m:.2f} m of roof)")
        print(f"    resistance needed   {opening.resistance_required_t_m2:6.2f} t/m2 "
              f"at SF {opening.safety_factor_required:.1f}")
        print(f"    max row spacing     {opening.max_row_spacing_m:6.2f} m")
        print(f"    bolt length needed  {opening.bolt_length_required_m:6.2f} m")
        print(f"    support safety fac. {opening.support_safety_factor:6.2f}")
    if result.pillar:
        p = result.pillar
        print(f"\n  Pillar — {p.pillar_width_m:.1f} m wide at {p.centre_distance_m:.1f} m centres")
        print(f"    strength {p.strength_mpa:.2f} MPa / stress {p.stress_mpa:.2f} MPa "
              f"-> FoS {p.factor_of_safety:.2f}")
    print(f"\n  Verdict: {result.verdict}")
    for check in result.checks:
        mark = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[check.status]
        print(f"    [{mark}] {check.title}: {check.detail}")
    print("\n  A design aid only. The Systematic Support Rules approved under the Coal Mines")
    print("  Regulations govern what is installed underground.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
