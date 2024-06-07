import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution


@pytest.fixture
@bw2test
def point_emission_in_ten_years_db():
    bd.Database("temporalis-bio").write(
        {
            ("temporalis-bio", "HFC23"): {  # only biosphere flow is CH4
                "type": "emission",
                "name": "HFC23",
                "temporalis code": "HFC23",
            },
        }
    )

    bd.Database("test").write(  # dummy system containing 1 activity
        {
            ("test", "A"): {
                "name": "A",
                "location": "somewhere",
                "reference product": "a",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("test", "A"),
                    },
                    {
                        "amount": 1,
                        "type": "biosphere",
                        "input": ("temporalis-bio", "HFC23"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([10], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),  # emission of CH4 10 years after execution of process A
                    },
                ],
            },
        }
    )

    bd.Method(("GWP", "example")).write(
        [
            (("temporalis-bio", "HFC23"), 1530),  # GWP100 from IPCC AR6
        ]
    )
