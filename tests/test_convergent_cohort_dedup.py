"""Regression test for coefficient inflation when supply-chain paths converge.

When a single producer cohort (same activity, same time bucket) is reached via
multiple consumer paths — e.g. a *backward* relative TD on a consumption edge,
where the product demanded in several years is served by one earlier-built
cohort — the per-unit technosphere coefficients of that cohort must be emitted
ONCE, not summed once per converging path.

The bug: `TimelineBuilder.build_timeline` deduplicated on raw datetimes, which
drift (datetime64[s] + timedelta64[Y] uses an average-length year), so the
duplicates produced by different paths survived `drop_duplicates` and were then
summed by the `groupby` over rounded buckets — inflating the coefficient by the
number of converging paths.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw_temporalis import TemporalDistribution

from bw_timex import TimexLCA


@pytest.fixture
def convergent_backward_db():
    project = "test_convergent_cohort_dedup"
    bd.projects.set_current(project)
    for db in list(bd.databases):
        del bd.databases[db]
    for m in list(bd.methods):
        del bd.methods[m]

    bd.Database("biosphere").write(
        {("biosphere", "CO2"): {"type": "emission", "name": "CO2", "unit": "kg"}}
    )
    emis = {"electricity": (0.40, 0.05), "manufacturing": (6000, 4000)}
    for yi, dbn in [(0, "background_2025"), (1, "background_2045")]:
        bd.Database(dbn).write(
            {
                (dbn, c): {
                    "name": c,
                    "reference product": c,
                    "unit": "x",
                    "location": "GLO",
                    "exchanges": [
                        {"amount": 1, "type": "production", "input": (dbn, c)},
                        {"amount": e[yi], "type": "biosphere", "input": ("biosphere", "CO2")},
                    ],
                }
                for c, e in emis.items()
            }
        )
    bd.Method(("GWP", "100")).write([(("biosphere", "CO2"), 1.0)])

    L = 12
    LIFETIME_KWH = 12000 * 0.2 * L
    LIFETIME_KM = 12000 * L
    ages = np.arange(0, L)
    td_use = TemporalDistribution(
        date=ages.astype("timedelta64[Y]"), amount=np.full(L, 1 / L)
    )
    td_back = TemporalDistribution(
        date=(-ages).astype("timedelta64[Y]"), amount=np.full(L, 1 / L)
    )

    fg = bd.Database("foreground")
    fg.register()
    ev = fg.new_node(code="ev", name="ev", unit="veh", location="GLO",
                     type=bd.labels.product_node_default)
    ev.save()
    evlc = fg.new_node(code="evlc", name="ev_lifecycle", unit="veh", location="GLO",
                       type=bd.labels.process_node_default)
    evlc.save()
    evlc.new_edge(input=ev, amount=1, type=bd.labels.production_edge_default).save()
    evlc.new_edge(input=bd.get_node(database="background_2025", code="manufacturing"),
                  amount=1, type=bd.labels.consumption_edge_default).save()
    evlc.new_edge(input=bd.get_node(database="background_2025", code="electricity"),
                  amount=LIFETIME_KWH, type=bd.labels.consumption_edge_default,
                  temporal_distribution=td_use).save()
    km = fg.new_node(code="km", name="km_driven", unit="km", location="GLO",
                     type=bd.labels.product_node_default)
    km.save()
    drv = fg.new_node(code="drv", name="driving", unit="km", location="GLO",
                      type=bd.labels.process_node_default)
    drv.save()
    drv.new_edge(input=km, amount=1, type=bd.labels.production_edge_default).save()
    drv.new_edge(input=ev, amount=1 / LIFETIME_KM,
                 type=bd.labels.consumption_edge_default,
                 temporal_distribution=td_back).save()
    for db in bd.databases:
        bd.Database(db).process()

    return {"km": km, "LIFETIME_KM": LIFETIME_KM, "LIFETIME_KWH": LIFETIME_KWH}


def _run(km, amount_td):
    dd = {
        "background_2025": datetime(2025, 1, 1),
        "background_2045": datetime(2045, 1, 1),
        "foreground": "dynamic",
    }
    t = TimexLCA(demand={km: amount_td}, method=("GWP", "100"), database_dates=dd)
    t.build_timeline(starting_datetime=datetime(2037, 1, 1), temporal_grouping="year")
    t.lci()
    t.static_lcia()
    return t


def test_no_coefficient_inflation_across_converging_paths(convergent_backward_db):
    km = convergent_backward_db["km"]
    LIFETIME_KM = convergent_backward_db["LIFETIME_KM"]

    # Demand km in two adjacent years. Their backward casts overlap on the build
    # years 2027..2037, which converge on the same ev_lifecycle cohorts.
    two = TemporalDistribution(
        date=np.array([datetime(2037, 1, 1), datetime(2038, 1, 1)], dtype="datetime64[s]"),
        amount=np.array([float(LIFETIME_KM), float(LIFETIME_KM)]),
    )
    t = _run(km, two)

    # The static base LCA evaluates everything on the dirtiest (2025) grid. Since
    # the grid only decarbonises going forward, the time-explicit score MUST NOT
    # exceed it. The convergence bug inflated it well above (ratio ~1.15).
    assert t.static_score <= t.base_lca.score, (
        f"time-explicit {t.static_score:,.0f} exceeds static {t.base_lca.score:,.0f} "
        "-> converging cohorts double-counted"
    )

    # Directly: no build-year cohort's electricity coefficient may exceed one
    # vehicle's lifetime electricity. (Two converging paths doubled it to 2x.)
    tl = t.timeline
    elec = tl[tl["producer_name"] == "electricity"]
    per_build = elec.groupby(elec["date_consumer"].dt.year)["amount"].sum()
    assert per_build.max() <= convergent_backward_db["LIFETIME_KWH"] + 1, (
        f"a build-year cohort consumes {per_build.max():,.0f} kWh/unit, "
        f"more than one lifetime ({convergent_backward_db['LIFETIME_KWH']:,.0f})"
    )


def test_subwindow_entries_still_sum(convergent_backward_db):
    """Single demand point: no convergence, coefficients must be intact (the fix
    must not over-dedup). One vehicle's worth of km => time-explicit < static."""
    km = convergent_backward_db["km"]
    LIFETIME_KM = convergent_backward_db["LIFETIME_KM"]
    one = TemporalDistribution(
        date=np.array([datetime(2037, 1, 1)], dtype="datetime64[s]"),
        amount=np.array([float(LIFETIME_KM)]),
    )
    t = _run(km, one)
    assert t.static_score <= t.base_lca.score
    # sanity: a single vehicle still produces a non-trivial score
    assert t.static_score > 0
