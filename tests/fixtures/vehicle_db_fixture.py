import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution


@pytest.fixture
@bw2test
def vehicle_db():
    bd.Database("bio").write(
        {
            ("bio", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        },
    )

    bd.Database("db_2020").write(
        {
            ("db_2020", "glider"): {
                "name": "market for glider, passenger car",
                "location": "somewhere",
                "reference product": "market for glider, passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "glider"),
                    },
                    {
                        "amount": 6.29,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2020", "powertrain"): {
                "name": "market for powertrain, for electric passenger car",
                "location": "somewhere",
                "reference product": "market for powertrain, for electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "powertrain"),
                    },
                    {
                        "amount": 17.89,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2020", "battery"): {
                "name": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "location": "somewhere",
                "reference product": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "battery"),
                    },
                    {
                        "amount": 8.23,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2020", "electricity"): {
                "name": "market group for electricity, low voltage",
                "location": "somewhere",
                "reference product": "market group for electricity, low voltage",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "electricity"),
                    },
                    {
                        "amount": 0.73,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2020", "dismantling"): {
                "name": "market for manual dismantling of used electric passenger car",
                "location": "somewhere",
                "reference product": "market for manual dismantling of used electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "dismantling"),
                    },
                    {
                        "amount": 0.0091,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2020", "battery_recycling"): {
                "name": "market for used Li-ion battery",
                "location": "somewhere",
                "reference product": "market for used Li-ion battery",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "battery_recycling"),
                    },
                    {
                        "amount": -1.18,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
        }
    )

    bd.Database("db_2030").write(
        {
            ("db_2030", "glider"): {
                "name": "market for glider, passenger car",
                "location": "somewhere",
                "reference product": "market for glider, passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "glider"),
                    },
                    {
                        "amount": 4.37,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2030", "powertrain"): {
                "name": "market for powertrain, for electric passenger car",
                "location": "somewhere",
                "reference product": "market for powertrain, for electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "powertrain"),
                    },
                    {
                        "amount": 11.90,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2030", "battery"): {
                "name": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "location": "somewhere",
                "reference product": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "battery"),
                    },
                    {
                        "amount": 5.26,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2030", "electricity"): {
                "name": "market group for electricity, low voltage",
                "location": "somewhere",
                "reference product": "market group for electricity, low voltage",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "electricity"),
                    },
                    {
                        "amount": 0.23,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2030", "dismantling"): {
                "name": "market for manual dismantling of used electric passenger car",
                "location": "somewhere",
                "reference product": "market for manual dismantling of used electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "dismantling"),
                    },
                    {
                        "amount": 0.008,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2030", "battery_recycling"): {
                "name": "market for used Li-ion battery",
                "location": "somewhere",
                "reference product": "market for used Li-ion battery",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "battery_recycling"),
                    },
                    {
                        "amount": -0.60,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
        }
    )

    bd.Database("db_2040").write(
        {
            ("db_2040", "glider"): {
                "name": "market for glider, passenger car",
                "location": "somewhere",
                "reference product": "market for glider, passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "glider"),
                    },
                    {
                        "amount": 3.71,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2040", "powertrain"): {
                "name": "market for powertrain, for electric passenger car",
                "location": "somewhere",
                "reference product": "market for powertrain, for electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "powertrain"),
                    },
                    {
                        "amount": 9.79,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2040", "battery"): {
                "name": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "location": "somewhere",
                "reference product": "battery production, Li-ion, LiMn2O4, rechargeable, prismatic",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "battery"),
                    },
                    {
                        "amount": 4.25,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2040", "electricity"): {
                "name": "market group for electricity, low voltage",
                "location": "somewhere",
                "reference product": "market group for electricity, low voltage",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "electricity"),
                    },
                    {
                        "amount": 0.067,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2040", "dismantling"): {
                "name": "market for manual dismantling of used electric passenger car",
                "location": "somewhere",
                "reference product": "market for manual dismantling of used electric passenger car",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "dismantling"),
                    },
                    {
                        "amount": 0.0077,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
            ("db_2040", "battery_recycling"): {
                "name": "market for used Li-ion battery",
                "location": "somewhere",
                "reference product": "market for used Li-ion battery",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2040", "battery_recycling"),
                    },
                    {
                        "amount": -0.40,  # aggregated LCI of CO2-eq
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                    },
                ],
            },
        }
    )

    # parameters for EV:
    ELECTRICITY_CONSUMPTION = 0.2  # kWh/km
    MILEAGE = 150_000  # km
    LIFETIME = 16  # years

    # Overall mass: 1200 kg
    MASS_GLIDER = 840  # kg
    MASS_POWERTRAIN = 80  # kg
    MASS_BATTERY = 280  # kg

    bd.Database("foreground").write(
        {
            ("foreground", "EV"): {
                "name": "electric vehicle life cycle",
                "location": "somewhere",
                "reference product": "electric vehicle life cycle",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "EV"),
                    },
                    {
                        "amount": MASS_GLIDER,
                        "type": "technosphere",
                        "input": ("db_2020", "glider"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(
                                [-2, -1], dtype="timedelta64[Y]"
                            ),  # 40% of production consumption in year -1, 60% in year -2
                            amount=np.array([0.6, 0.4]),
                        ),
                    },
                    {
                        "amount": MASS_POWERTRAIN,
                        "type": "technosphere",
                        "input": ("db_2020", "powertrain"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(
                                [-2, -1], dtype="timedelta64[Y]"
                            ),  # 40% of production consumption in year -1, 60% in year -2
                            amount=np.array([0.6, 0.4]),
                        ),
                    },
                    {
                        "amount": MASS_BATTERY,
                        "type": "technosphere",
                        "input": ("db_2020", "battery"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(
                                [-2, -1], dtype="timedelta64[Y]"
                            ),  # 40% of production consumption in year -1, 60% in year -2
                            amount=np.array([0.6, 0.4]),
                        ),
                    },
                    {
                        "amount": ELECTRICITY_CONSUMPTION * MILEAGE,
                        "type": "technosphere",
                        "input": ("db_2020", "electricity"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(
                                [int(LIFETIME / 2)], dtype="timedelta64[Y]"
                            ),  # all electricity consumption in year 8, to simplify tests
                            amount=np.array([1]),
                        ),
                    },
                    {
                        "amount": MASS_GLIDER,
                        "type": "technosphere",
                        "input": ("db_2020", "dismantling"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([LIFETIME + 1], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                    },
                    {
                        "amount": -MASS_BATTERY,
                        "type": "technosphere",
                        "input": ("db_2020", "battery_recycling"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([LIFETIME + 1], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                    },
                ],
            },
        }
    )

    bd.Method(("GWP", "example")).write(
        [
            (("bio", "CO2"), 1),
        ]
    )
