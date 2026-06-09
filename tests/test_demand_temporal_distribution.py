from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw_temporalis import TemporalDistribution

from bw_timex import TimexLCA


def _setup_heat_demand_project():
    bd.projects.set_current("__test_demand_temporal_distribution__")
    for db in list(bd.databases):
        del bd.databases[db]
    for m in list(bd.methods):
        del bd.methods[m]

    bd.Database("bio").write(
        {("bio", "co2"): {"name": "CO2", "unit": "kg", "type": "emission"}}
    )
    for db_name, co2 in [("elec_2025", 0.40), ("elec_2035", 0.10)]:
        bd.Database(db_name).write(
            {
                (db_name, "elec"): {
                    "name": "electricity market",
                    "reference product": "electricity",
                    "unit": "kWh",
                    "location": "DE",
                    "exchanges": [
                        {"input": (db_name, "elec"), "amount": 1, "type": "production"},
                        {"input": ("bio", "co2"), "amount": co2, "type": "biosphere"},
                    ],
                }
            }
        )
    bd.Method(("m",)).write([(("bio", "co2"), 1.0)])

    bd.Database("fg").write(
        {
            ("fg", "heat"): {
                "name": "heat",
                "type": "product",
                "unit": "kWh",
                "location": "DE",
                "exchanges": [],
            },
            ("fg", "hp_operation"): {
                "name": "heat_pump_operation",
                "type": "process",
                "unit": "kWh",
                "location": "DE",
                "exchanges": [
                    {
                        "input": ("fg", "heat"),
                        "amount": 1,
                        "type": "production",
                    },
                    {
                        "input": ("elec_2025", "elec"),
                        "amount": 0.33,
                        "type": "technosphere",
                    },
                ],
            },
        }
    )
    for db in bd.databases:
        bd.Database(db).process()


def test_demand_with_absolute_td_distributes_cohorts():
    _setup_heat_demand_project()
    heat = bd.get_node(database="fg", code="heat")

    demand_td = TemporalDistribution(
        date=np.array(
            [datetime(2025, 1, 1), datetime(2035, 1, 1)],
            dtype="datetime64[s]",
        ),
        amount=np.array([600.0, 400.0]),
    )

    tlca = TimexLCA(
        demand={heat: demand_td},
        method=("m",),
        database_dates={
            "elec_2025": datetime(2025, 1, 1),
            "elec_2035": datetime(2035, 1, 1),
            "fg": "dynamic",
        },
    )
    tlca.build_timeline(starting_datetime=datetime(2025, 1, 1), temporal_grouping="year")
    tlca.lci()
    tlca.static_lcia()

    fu_rows = tlca.timeline[tlca.timeline["consumer"] == -1]
    fu_by_year = {row.date_producer.year: float(row.amount) for row in fu_rows.itertuples()}
    assert fu_by_year == pytest.approx({2025: 600.0, 2035: 400.0})

    elec = tlca.timeline[tlca.timeline["producer_name"] == "electricity market"]
    elec_years = {row.date_producer.year for row in elec.itertuples()}
    assert elec_years == {2025, 2035}

    # 600 kWh heat at 2025 (0.33 kWh elec/kWh heat, 0.4 kg CO2/kWh)
    # + 400 kWh heat at 2035 (0.33 kWh elec/kWh heat, 0.1 kg CO2/kWh)
    expected = 600 * 0.33 * 0.40 + 400 * 0.33 * 0.10
    assert tlca.static_score == pytest.approx(expected)


def test_demand_td_must_be_absolute_datetime():
    _setup_heat_demand_project()
    heat = bd.get_node(database="fg", code="heat")
    rel_td = TemporalDistribution(
        date=np.array([0, 10], dtype="timedelta64[Y]"),
        amount=np.array([0.6, 0.4]),
    )
    with pytest.raises(ValueError, match="absolute"):
        TimexLCA(
            demand={heat: rel_td},
            method=("m",),
            database_dates={
                "elec_2025": datetime(2025, 1, 1),
                "elec_2035": datetime(2035, 1, 1),
                "fg": "dynamic",
            },
        )
