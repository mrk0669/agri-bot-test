# NMRSA — Mine Roof Support Analyser

A roof support design tool for Indian underground coal mines, built around the
**CMRI-ISM rock mass rating** and checked against the **Coal Mines Regulations 2017**.

Open `index.html` in any browser. There is nothing to install, no server, and no network
call — which is the point, because the people who need it are working where there is no
signal. Projects and the roof register are kept in the browser's own storage.

```
nmrsa/
  index.html              the tool itself, self-contained
  nmrsa_engine.py         the same calculations in Python, with a CLI
  test_nmrsa_engine.py    42 unit tests over the engine
```

## What it computes

| Step | Expression | Units |
|---|---|---|
| Basic rating | layer thickness (30) + structural features (25) + weatherability (20) + rock strength (15) + ground water (10) | — |
| Adjusted rating | basic × situation adjustment factors | — |
| Rock load | `P = W · γ · (1.7 − 0.037·RMR + 0.0002·RMR²)` | t/m² |
| Rock load height | `P / γ` | m |
| Junction span | `√(W₁² + W₂²)` — the diagonal, not the width | m |
| Support resistance | `(bolts per row × capacity) / (span × row spacing)` | t/m² |
| Support safety factor | resistance ÷ rock load | — |
| Bolt length needed | rock load height + 0.3 m of grout, never below 1.2 m | m |
| Pillar strength | `S = 0.27·σc·h^−0.36 + (H/250)·(w/h − 1)` — Sheorey / CIMFR | MPa |
| Pillar stress | `σp = 0.025·H·((w + B)/w)²` — tributary area at 25 kN/m³ | MPa |

The rock-load bracket has roots at RMR 85 and RMR 100 and dips marginally negative between
them, so it is floored at zero. Above RMR 85 the method predicts no mobilised load and the
statutory minimum governs — the tool says so rather than reporting a meaningless number.

## What it checks

- Gallery width 4.8 m and height 3.0 m — Coal Mines Regulations 2017, Reg. 111.
- Minimum distance between pillar centres for the depth of cover and gallery width.
- Bolt spacing across the row and between rows against the 1.5 m normally accepted in a
  Systematic Support Rule.
- Support safety factor: 1.5 in the gallery, 2.0 at the junction, both editable.
- Bolt length against the rock load band, and drilling clearance against gallery height.
- Pillar factor of safety, 1.5 for development and 2.0 to stand long term.

Where a junction would need rows closer than 0.75 m — which happens quickly in weak roof —
the tool says to carry it on cable bolts, W-straps or breaker props instead of pretending a
bolt pattern alone will do it.

## The tool

**Support design** — the rating on five sliders with a plain-language band for each, the
geometry, the pattern, and the result: rock load, required and offered resistance, the
safety factor against its threshold, a cross-section drawn to scale showing the rock-load
band and the bolts, and a junction plan showing the two patterns. *Suggest a pattern* solves
for the loosest compliant spacing and fills it in.

**Design charts** — rock load against rating, largest permissible row spacing against
rating, and rating against depth of cover with the three support regimes and the boundaries
that matter for this working.

**Pillars** — Sheorey strength against tributary stress, factor of safety across a range of
centre distances, and the Reg. 111 table with the applicable row highlighted.

**Roof register** — a local log of falls, cavities and tell-tale movements against the
conditions that produced them, plotted on rating and span. It ships empty on purpose: no
case data is invented here, and after a few months your own records describe your seam
better than any published table. Export copies it as JSON.

**Method & codes** — every formula, what it came from, and what it does not cover.

Metric throughout, with an imperial display toggle. Print produces the whole thing as a
report with a sign-off block.

## Python engine

Same arithmetic, testable and scriptable:

```bash
python nmrsa/nmrsa_engine.py --rmr 45 --width 4.2 --depth 150 --pillar-centre 25.5
python nmrsa/nmrsa_engine.py --rmr 45 --json          # machine-readable
python -m pytest nmrsa -q                             # 42 tests
```

The tests are hand-computed from the published expressions, so a failure means the engine
drifted from the method rather than that a threshold moved. The browser tool and the Python
engine were cross-checked value-for-value across 21 cases.

## Limits — read before using a number from here

NMRSA is a **design aid**. What is installed underground is governed by the Systematic
Support Rules framed under the Coal Mines Regulations 2017 and by the DGMS circulars in
force, approved by the manager and the Regional Inspector.

- The statutory figures reproduced here — the width and height limits and the pillar centre
  table of Reg. 111 — are **reproduced for convenience and must be verified against the
  Gazette text** before any statutory plan is drawn on them.
- Bolt anchorage capacities in the picker are indicative. Use your own pull-out test results.
- RMR situation adjustment factors default to 1.00. The site values belong to the
  geotechnical report; the tool does not invent them.
- The empirical expressions were fitted to Indian bord-and-pillar development in the
  3.6–4.8 m width range. Outside it, treat the output as indicative only.

## Sources

- Venkateswarlu, V. (1986) — *Geomechanics classification of coal measure rocks vis-à-vis
  roof supports*, CMRI, Dhanbad. The five-parameter rating and its weighting.
- Paul, A. et al. (2014) — *Rock load estimation in development galleries and junctions for
  underground coal mines: a CMRI-ISM rock mass rating approach*, Journal of Mining 618719.
- Sheorey, P. R. (1987) — *Coal pillar strength estimation from failed and stable cases*,
  Int. J. Rock Mech. Min. Sci.
- Coal Mines Regulations 2017 (DGMS), Reg. 111 — development work.
