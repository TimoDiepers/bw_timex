import bw2data as bd
import numpy as np
import pandas as pd
import pytest

from timex_lca import TimexLCA


def test_custom_dynamic_characterization(point_emission_in_ten_years_db):
    demand = {("test", "A"): 1}
    gwp = ("GWP", "example")
    tlca = TimexLCA(demand, gwp)
    tlca.build_timeline()
    tlca.lci()

    characterization_function_dict_HFC23 = {
        bd.get_node(code="HFC23").id: characterize_HFC23,
    }
    tlca.dynamic_lcia(
        metric="radiative_forcing",
        time_horizon=100,
        characterization_function_dict=characterization_function_dict_HFC23,
    )

    sum_of_forcing = tlca.dynamic_score.sum()
    AGWP100_CO2 = 0.0895e-12
    GWP100_HFC23 = 14600
    AGWP100_HFC23 = 1310e-12

    assert sum_of_forcing == pytest.approx(
        AGWP100_HFC23, abs=2e-11
    )  # allow for some rounding errors
    assert sum_of_forcing / AGWP100_CO2 == pytest.approx(
        GWP100_HFC23, abs=5e-15
    )  # allow for some rounding errors

    tlca.dynamic_lcia(
        metric="GWP",
        time_horizon=100,
        characterization_function_dict=characterization_function_dict_HFC23,
    )
    assert tlca.dynamic_score == pytest.approx(
        AGWP100_HFC23, abs=5e-15
    )  # allow for some rounding errors


def characterize_HFC23(
    series,
    period: int = 100,
    cumulative=False,
) -> pd.DataFrame:
    """
    This is not the same function as in the dynamic_characterization module. This does not include indirect effects to be more comparable to standard GWP values.
    """

    # functional variables and units (from publications listed in docstring)
    radiative_efficiency_ppb = 0.191  # W/m2/ppb; 2019 background cch4 concentration; IPCC AR6 Table 7.15. This number includes indirect effects.

    # for conversion from ppb to kg-CH4
    M_HFC23 = 0.07002e3  # g/mol
    M_air = 28.97  # g/mol, dry air
    m_atmosphere = 5.135e18  # kg [Trenberth and Smith, 2005]

    radiative_efficiency_kg = (
        radiative_efficiency_ppb * M_air / M_HFC23 * 1e9 / m_atmosphere
    )  # W/m2/kg-CH4

    tau = 228  # Lifetime (years)

    date_beginning: np.datetime64 = series["date"].to_numpy()
    date_characterized: np.ndarray = date_beginning + np.arange(
        start=0, stop=period, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")

    decay_multipliers: list = np.array(
        [
            radiative_efficiency_kg * tau * (1 - np.exp(-year / tau))
            for year in range(period)
        ]
    )

    forcing = pd.Series(data=series.amount * decay_multipliers, dtype="float64")

    if not cumulative:
        forcing = forcing.diff(periods=1).fillna(0)

    return pd.DataFrame(
        {
            "date": pd.Series(data=date_characterized, dtype="datetime64[s]"),
            "amount": forcing,
            "flow": series.flow,
            "activity": series.activity,
        }
    )
