import json
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bw2data.backends import ActivityDataset as AD
from bw2data.backends.proxies import Exchange
from bw2data.backends.schema import ExchangeDataset
from bw2data.errors import MultipleResults, UnknownObject
from bw_temporalis import TemporalDistribution, easy_timedelta_distribution
from IPython.display import Javascript, display
from ipywidgets import (
    Button,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    IntText,
    Label,
    Layout,
    Output,
    Textarea,
    ToggleButtons,
    VBox,
)
from loguru import logger

time_res_mapping_strftime = {
    "year": "%Y",
    "month": "%Y%m",
    "day": "%Y%m%d",
    "hour": "%Y%m%d%H",
}


def extract_date_as_integer(dt_obj: datetime, time_res: Optional[str] = "year") -> int:
    """
    Converts a datetime object to an integer for a given temporal resolution `time_res`

    Parameters
    ----------

    dt_obj : Datetime object.
        Datetime object to be converted to an integer.

    time_res : str, optional
        time resolution to be returned: year=YYYY, month=YYYYMM, day=YYYYMMDD, hour=YYYYMMDDHH

    Returns
    -------
    date_as_integer : int
        Datetime object converted to an integer in the format of time_res

    """
    if time_res not in time_res_mapping_strftime:
        available = list(time_res_mapping_strftime.keys())
        raise ValueError(
            f"Invalid time_res: '{time_res}'. Please choose from: {available}."
        )
    formatted_date = dt_obj.strftime(time_res_mapping_strftime[time_res])
    date_as_integer = int(formatted_date)

    return date_as_integer


def extract_date_as_string(timestamp: datetime, temporal_grouping: str) -> str:
    """
    Extracts the grouping date as a string from a datetime object, based on the chosen temporal
    grouping. E.g. for `temporal_grouping` = 'month', and `timestamp` = 2023-03-29T01:00:00, it
    extracts the string '202303'.


    Parameters
    ----------
    timestamp : Datetime object
        Datetime object to be converted to a string.
    temporal_grouping : str
        Temporal grouping for the date string. Options are: 'year', 'month', 'day', 'hour'


    Returns
    -------
    date_as_string: str
        Date as a string in the format of the chosen temporal grouping.
    """

    if temporal_grouping not in time_res_mapping_strftime.keys():
        raise ValueError(
            f'temporal_grouping: {temporal_grouping} is not a valid option. Please \
            choose from: {list(time_res_mapping_strftime.keys())}, defaulting to "year"',
        )
    return timestamp.strftime(time_res_mapping_strftime[temporal_grouping])


def convert_date_string_to_datetime(temporal_grouping, date_string) -> datetime:
    """
    Converts the string of a date to datetime object.
    e.g. for `temporal_grouping` = 'month', and `date_string` = '202303', it extracts 2023-03-01

    Parameters
    ----------
    temporal_grouping : str
        Temporal grouping for the date string. Options are: 'year', 'month', 'day', 'hour'
    date_string : str
        Date as a string

    Returns
    -------
    datetime
        Datetime object of the date string at the chosen temporal resolution.
    """
    time_res_dict = {
        "year": "%Y",
        "month": "%Y%m",
        "day": "%Y%m%d",
        "hour": "%Y%m%d%H",
    }

    if temporal_grouping not in time_res_dict.keys():
        raise ValueError(
            f'temporal grouping: {temporal_grouping} is not a valid option. Please \
            choose from: {list(time_res_dict.keys())}, defaulting to "year"',
        )
    return datetime.strptime(date_string, time_res_dict[temporal_grouping])


def round_datetime(date: datetime, resolution: str) -> datetime:
    """
    Round a datetime object based on a given resolution

    Parameters
    ----------
    date : datetime
        datetime object to be rounded
    resolution: str
        Temporal resolution to round the datetime object to. Options are: 'year', 'month', 'day' and
        'hour'.

    Returns
    -------
    datetime
        rounded datetime object
    """
    if resolution == "year":
        mid_year = pd.Timestamp(f"{date.year}-07-01")
        return (
            pd.Timestamp(f"{date.year+1}-01-01")
            if date >= mid_year
            else pd.Timestamp(f"{date.year}-01-01")
        )

    if resolution == "month":
        start_of_month = pd.Timestamp(f"{date.year}-{date.month}-01")
        next_month = start_of_month + pd.DateOffset(months=1)
        mid_month = start_of_month + (next_month - start_of_month) / 2
        return next_month if date >= mid_month else start_of_month

    if resolution == "day":
        start_of_day = datetime(date.year, date.month, date.day)
        mid_day = start_of_day + timedelta(hours=12)
        return start_of_day + timedelta(days=1) if date >= mid_day else start_of_day

    if resolution == "hour":
        start_of_hour = datetime(date.year, date.month, date.day, date.hour)
        mid_hour = start_of_hour + timedelta(minutes=30)
        return start_of_hour + timedelta(hours=1) if date >= mid_hour else start_of_hour

    raise ValueError("Resolution must be one of 'year', 'month', 'day', or 'hour'.")


def add_flows_to_characterization_functions(
    flows: Union[str, List[str]],
    func: Callable,
    characterization_functions: Optional[dict] = None,
) -> dict:
    """
    Add a new flow or a list of flows to the available characterization functions.

    Parameters
    ----------
    flows : Union[str, List[str]]
        Flow or list of flows to be added to the characterization function dictionary.
    func : Callable
        Dynamic characterization function for flow.
    characterization_functions : dict, optional
        Dictionary of flows and their corresponding characterization functions. Default is an empty
        dictionary.

    Returns
    -------
    dict
        Updated characterization function dictionary with the new flow(s) and function(s).
    """

    if characterization_functions is None:
        characterization_functions = {}

    # Check if the input is a single flow (str) or a list of flows (List[str])
    if isinstance(flows, str):
        # It's a single flow, add it directly
        characterization_functions[flows] = func
    elif isinstance(flows, list):
        # It's a list of flows, iterate and add each one
        for flow in flows:
            characterization_functions[flow] = func

    return characterization_functions


def resolve_temporalized_node_name(code: str) -> str:
    """
    Getting the name of a node based on the code only.
    Works for non-unique codes if the name is the same across all databases.

    Parameters
    ----------
    code: str
        Code of the node to resolve.

    Returns
    -------
    str
        Name of the node.
    """
    qs = AD.select().where(AD.code == code)
    names = set([obj.name for obj in qs])
    if len(qs) > 1:
        if len(names) > 1:
            raise ValueError(
                "Found multiple names for the given code: {}".format(names)
            )
    elif not qs:
        raise UnknownObject
    return names.pop()


def plot_characterized_inventory_as_waterfall(
    lca_obj,
    static_scores=None,
    prospective_scores=None,
    order_stacked_activities=None,
):
    """
    Plot a stacked waterfall chart of characterized inventory data. As comparison,
    static and prospective scores can be added. Only works for metric GWP at the moment.

    Parameters
    ----------
    lca_obj : TimexLCA
        LCA object with characterized inventory data.
    static_scores : dict, optional
        Dictionary of static scores. Default is None.
    prospective_scores : dict, optional
        Dictionary of prospective scores. Default is None.
    order_stacked_activities : list, optional
        List of activities to order the stacked bars in the waterfall plot. Default is None.

    Returns
    -------
    None
        plots the waterfall chart.

    """
    if not hasattr(lca_obj, "characterized_inventory"):
        raise ValueError("LCA object does not have characterized inventory data.")

    if not hasattr(lca_obj, "activity_time_mapping"):
        raise ValueError("Make sure to pass an instance of a TimexLCA.")

    time_res_dict = {
        "year": "%Y",
        "month": "%Y-%m",
        "day": "%Y-%m-%d",
        "hour": "%Y-%m-%d %H",
    }

    plot_data = lca_obj.characterized_inventory.copy()

    plot_data["year"] = plot_data["date"].dt.strftime(
        time_res_dict[lca_obj.temporal_grouping]
    )  # TODO make temporal resolution flexible

    # Optimized activity label fetching using the TimexLCA's built-in method
    unique_activities = plot_data["activity"].unique()
    activity_labels = {
        idx: lca_obj.get_activity_name_from_time_mapped_id(idx)
        for idx in unique_activities
    }
    plot_data["activity_label"] = plot_data["activity"].map(activity_labels)

    plot_data = plot_data.groupby(["year", "activity_label"], as_index=False)[
        "amount"
    ].sum()
    pivoted_data = plot_data.pivot(
        index="year", columns="activity_label", values="amount"
    )

    combined_data = []
    # Adding exchange_scores as a static column
    if static_scores:
        static_data = pd.DataFrame(
            static_scores.items(), columns=["activity_label", "amount"]
        )
        static_data["year"] = "static"
        pivoted_static_data = static_data.pivot(
            index="year", columns="activity_label", values="amount"
        )
        combined_data.append(pivoted_static_data)

    combined_data.append(pivoted_data)  # making sure the order is correct

    # Adding exchange_scores as a prospective column
    if prospective_scores:
        prospective_data = pd.DataFrame(
            prospective_scores.items(), columns=["activity_label", "amount"]
        )
        prospective_data["year"] = "prospective"
        pivoted_prospective_data = prospective_data.pivot(
            index="year", columns="activity_label", values="amount"
        )
        combined_data.append(pivoted_prospective_data)

    combined_df = pd.concat(combined_data, axis=0)

    if order_stacked_activities:
        combined_df = combined_df[
            order_stacked_activities
        ]  # change order of activities in the stacked bars of the waterfall

    # Calculate the bottom for only the dynamic data
    dynamic_bottom = pivoted_data.sum(axis=1).cumsum().shift(1).fillna(0)

    if static_scores and prospective_scores:
        bottom = pd.concat([pd.Series([0]), dynamic_bottom, pd.Series([0])])
    elif static_scores:
        bottom = pd.concat([pd.Series([0]), dynamic_bottom])
    elif prospective_scores:
        bottom = pd.concat([dynamic_bottom, pd.Series([0])])
    else:
        bottom = dynamic_bottom

    # Plotting
    ax = combined_df.plot(
        kind="bar",
        stacked=True,
        bottom=bottom,
        figsize=(14, 6),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("GWP [kg CO2-eq]")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")

    if static_scores:
        ax.axvline(x=0.5, color="black", linestyle="--", lw=1)
    if prospective_scores:
        ax.axvline(x=len(combined_df) - 1.5, color="black", linestyle="--", lw=1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # x=1.02 moves it outside, y=0.5 centers vertically
        fontsize="small",
    )
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.show()


def get_exchange(**kwargs) -> Exchange:
    """
    Get an exchange from the database.

    Parameters
    ----------
    **kwargs :
        Arguments to specify an exchange.
            - input_node: Input node object
            - input_code: Input node code
            - input_database: Input node database
            - output_node: Output node object
            - output_code: Output node code
            - output_database: Output node database

    Returns
    -------
    Exchange
        The exchange object matching the criteria.

    Raises
    ------
    MultipleResults
        If multiple exchanges match the criteria.
    UnknownObject
        If no exchange matches the criteria.
    """

    # Process input_node if present
    input_node = kwargs.pop("input_node", None)
    if input_node:
        kwargs["input_code"] = input_node["code"]
        kwargs["input_database"] = input_node["database"]

    # Process output_node if present
    output_node = kwargs.pop("output_node", None)
    if output_node:
        kwargs["output_code"] = output_node["code"]
        kwargs["output_database"] = output_node["database"]

    # Map kwargs to database fields
    mapping = {
        "input_code": ExchangeDataset.input_code,
        "input_database": ExchangeDataset.input_database,
        "output_code": ExchangeDataset.output_code,
        "output_database": ExchangeDataset.output_database,
    }

    # Build query filters
    filters = []
    for key, value in kwargs.items():
        field = mapping.get(key)
        if field is not None:
            filters.append(field == value)

    # Execute query with filters
    qs = ExchangeDataset.select().where(*filters)
    candidates = [Exchange(obj) for obj in qs]
    num_candidates = len(candidates)

    if num_candidates > 1:
        raise MultipleResults(
            f"Found {num_candidates} results for the given search. "
            "Please be more specific or double-check your system model for duplicates."
        )
    if num_candidates == 0:
        raise UnknownObject("No exchange found matching the criteria.")

    return candidates[0]


def add_temporal_distribution_to_exchange(
    temporal_distribution: TemporalDistribution, **kwargs
):
    """
    Adds a temporal distribution to an exchange specified by kwargs.

    Parameters
    ----------
    temporal_distribution : TemporalDistribution
        TemporalDistribution to be added to the exchange.
    **kwargs :
        Arguments to specify an exchange.
            - input_node: Input node object
            - input_id: Input node database ID
            - input_code: Input node code
            - input_database: Input node database
            - output_node: Output node object
            - output_id: Output node database ID
            - output_code: Output node code
            - output_database: Output node database

    Returns
    -------
    None
        The exchange is saved with the temporal distribution.
    """
    from .validation import TemporalDistributionExchangeInputs

    TemporalDistributionExchangeInputs(temporal_distribution=temporal_distribution)
    exchange = get_exchange(**kwargs)
    exchange["temporal_distribution"] = temporal_distribution
    exchange.save()
    logger.info(f"Added temporal distribution to exchange {exchange}.")


def add_temporal_evolution_to_exchange(
    temporal_evolution_factors: dict = None,
    temporal_evolution_amounts: dict = None,
    temporal_evolution_reference: str = "producer",
    **kwargs,
):
    """Add temporal evolution data to an exchange specified by kwargs.

    Parameters
    ----------
    temporal_evolution_factors : dict, optional
        Dictionary mapping datetime keys to scaling factors.
    temporal_evolution_amounts : dict, optional
        Dictionary mapping datetime keys to absolute amounts.
    temporal_evolution_reference : {"producer", "consumer"}, optional
        Whether temporal evolution is evaluated at producer or consumer timestamps.
    **kwargs :
        Arguments to specify an exchange (same as get_exchange).

    Returns
    -------
    None
        The exchange is saved with the temporal evolution data.
    """
    from .validation import TemporalEvolutionExchangeInputs

    TemporalEvolutionExchangeInputs(
        temporal_evolution_factors=temporal_evolution_factors,
        temporal_evolution_amounts=temporal_evolution_amounts,
        temporal_evolution_reference=temporal_evolution_reference,
    )
    exchange = get_exchange(**kwargs)
    if temporal_evolution_factors is not None:
        exchange["temporal_evolution_factors"] = temporal_evolution_factors
    if temporal_evolution_amounts is not None:
        exchange["temporal_evolution_amounts"] = temporal_evolution_amounts
    exchange["temporal_evolution_reference"] = temporal_evolution_reference
    exchange.save()
    logger.info(f"Added temporal evolution to exchange {exchange}.")


def interactive_td_widget():
    """
    Create an interactive ipywidget for drafting and visualizing temporal distributions and copying
    them to the clipboard.

    For use in jupyter notebooks.

    Returns
    -------
    ipywidgets.VBox
        Interactive widget for drafting temporal distributions.
    """
    # ---------- Controls ----------
    mode = ToggleButtons(
        options=["Generator", "Manual"], value="Generator", description="Mode"
    )

    # Generator controls
    start = IntText(value=0, description="start")
    end = IntText(value=10, description="end")
    resolution = Dropdown(
        options=[("Years", "Y"), ("Months", "M"), ("Days", "D")],
        value="Y",
        description="resolution",
    )
    steps = IntSlider(
        value=11, min=2, max=11, step=1, description="steps", continuous_update=False
    )
    kind = ToggleButtons(
        options=["uniform", "triangular", "normal"], value="uniform", description="kind"
    )
    # Give wide initial bounds; we'll override on kind changes
    param = FloatSlider(
        value=1.0,
        min=0.01,
        max=50.0,
        step=0.01,
        description="param",
        disabled=True,
        continuous_update=False,
    )

    # Manual controls
    manual_unit = Dropdown(
        options=[
            ("Years", "Y"),
            ("Months", "M"),
            ("Days", "D"),
            ("Hours", "h"),
            ("Minutes", "m"),
            ("Seconds", "s"),
        ],
        value="Y",
        description="resolution",
    )
    dates_text = Textarea(
        value="0, 2, 4, 6, 8, 10",
        description="dates",
        layout=Layout(width="100%", min_height="70px"),
    )
    amounts_text = Textarea(
        value="0.1, 0.1, 0.2, 0.2, 0.2, 0.2",
        description="amounts",
        layout=Layout(width="100%", min_height="70px"),
    )

    for widget in (start, end, resolution, steps, param, manual_unit):
        widget.style.description_width = "initial"

    steps.layout = Layout(width="220px")
    param.layout = Layout(width="220px")
    start.layout = Layout(width="160px")
    end.layout = Layout(width="160px")
    resolution.layout = Layout(width="180px")
    manual_unit.layout = Layout(width="220px")

    copy_btn = Button(description="Copy TD code", button_style="success")
    copy_import_btn = Button(description="Copy TD + imports", button_style="")
    copy_btn.layout = Layout(width="160px")
    copy_import_btn.layout = Layout(width="200px")
    status = Label(value="")
    status.layout = Layout(margin="0 0 0 8px")
    plot_out = Output(layout=Layout(width="100%"))

    # ---------- Helpers ----------
    def _parse_num_list(txt: str, label: str) -> List[float]:
        parts = [p for p in txt.replace(",", " ").split() if p]
        if not parts:
            raise ValueError(f"{label} cannot be empty.")

        values = []
        for p in parts:
            try:
                if ("." in p) or ("e" in p.lower()):
                    values.append(float(p))
                else:
                    values.append(int(p))
            except ValueError as exc:
                raise ValueError(f"Could not parse '{p}' in {label}.") from exc
        return values

    def _format_number(value: float) -> str:
        as_float = float(value)
        if np.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
        return (f"{as_float:.6f}").rstrip("0").rstrip(".")

    def _make_td_generator():
        return easy_timedelta_distribution(
            start=min(start.value, end.value),
            end=max(start.value, end.value),
            resolution=resolution.value,
            steps=int(steps.value),
            kind=kind.value,
            param=None if param.disabled else float(param.value),
        )

    def _make_td_manual():
        d = _parse_num_list(dates_text.value, "dates")
        a = _parse_num_list(amounts_text.value, "amounts")
        if len(d) != len(a):
            raise ValueError("dates and amounts must have the same length.")
        if not d:
            raise ValueError("Provide at least one date and amount.")
        date = np.array(d, dtype=f"timedelta64[{manual_unit.value}]")
        amount = np.array(a, dtype=float)
        if np.any(np.isnan(amount)):
            raise ValueError("Amounts must be numeric values.")
        return TemporalDistribution(date=date, amount=amount)

    def _current_td():
        return _make_td_generator() if mode.value == "Generator" else _make_td_manual()

    def _current_resolution_for_graph():
        return resolution.value if mode.value == "Generator" else manual_unit.value

    def _draw_graph(td: TemporalDistribution):
        with plot_out:
            plot_out.clear_output(wait=True)
            fig = plt.figure(figsize=(7, 3))
            td.graph(style="default", resolution=_current_resolution_for_graph())
            plt.show()
            plt.close(fig)
        status.value = (
            f"OK · steps={len(td.amount)} · sum(amount)={float(np.sum(td.amount)):.6f}"
        )

    def refresh_preview(*_):
        try:
            td = _current_td()
            _draw_graph(td)
        except Exception as exc:
            with plot_out:
                plot_out.clear_output(wait=True)
            status.value = f"Error: {exc}"

    # --- robust param updater (avoid value snapping back to 1.0) ---
    def _with_param_unobserved(fn):
        try:
            param.unobserve(refresh_preview, names="value")
            fn()
        finally:
            param.observe(refresh_preview, names="value")

    def _reset_param_for_kind():
        def _apply():
            s, e = sorted([start.value, end.value])
            span = abs(e - s)

            def _set_slider_value(target: float) -> None:
                bounded = min(max(target, param.min), param.max)
                step = param.step or 0
                if step <= 0:
                    param.value = bounded
                    return
                base = param.min
                ticks = round((bounded - base) / step)
                param.value = base + ticks * step

            if kind.value == "uniform":
                param.description = "param"
                param.disabled = True
                param.min = 0.1
                param.max = 50.0
                param.step = 0.1
                param.value = 1.0
                param.layout.display = "none"
                return

            if kind.value == "triangular":
                param.description = "mode"
                if s == e:
                    param.disabled = True
                    exact_value = float(s)
                    param.min = exact_value
                    param.max = exact_value
                    param.step = 1.0
                    param.value = exact_value
                else:
                    param.disabled = False
                    param.min = float(s)
                    param.max = float(e)
                    param.step = max((param.max - param.min) / 20.0, 0.01)
                    _set_slider_value((param.min + param.max) / 2.0)
                param.layout.display = ""
                return

            # normal
            param.description = "std dev"
            param.disabled = False
            span = max(span, 1)
            param.min = 0.02
            param.max = 1.0
            param.step = 0.01
            _set_slider_value(0.15)
            param.layout.display = ""

        _with_param_unobserved(_apply)

    def _code_generator():
        s, e = sorted([start.value, end.value])
        k = kind.value
        p = None if param.disabled else float(param.value)
        code = (
            "td = easy_timedelta_distribution(\n"
            f"    start={s},\n"
            f"    end={e},\n"
            f"    resolution='{resolution.value}',\n"
            f"    steps={int(steps.value)},\n"
            f"    kind='{k}'"
        )
        if p is not None:
            code += f",\n    param={_format_number(p)}"
        code += "\n)"
        return code

    def _code_manual():
        d = _parse_num_list(dates_text.value, "dates")
        a = _parse_num_list(amounts_text.value, "amounts")
        unit = manual_unit.value
        d_str = ", ".join(str(int(x)) for x in d)
        a_str = ", ".join(_format_number(x) for x in a)
        return (
            f"date = np.array([{d_str}], dtype='timedelta64[{unit}]')\n"
            f"amount = np.array([{a_str}], dtype=float)\n"
            "td = TemporalDistribution(date=date, amount=amount)"
        )

    def _build_code(include_imports: bool = False) -> str:
        body = _code_generator() if mode.value == "Generator" else _code_manual()
        if not include_imports:
            return body

        if mode.value == "Generator":
            imports = [
                "from bw_temporalis import easy_timedelta_distribution",
            ]
        else:
            imports = [
                "import numpy as np",
                "from bw_temporalis import TemporalDistribution",
            ]

        return "\n".join(imports + ["", body])

    def _copy_code(include_imports: bool) -> None:
        try:
            code = _build_code(include_imports=include_imports)
            display(Javascript(f"navigator.clipboard.writeText({json.dumps(code)})"))
            suffix = " + imports" if include_imports else ""
            status.value = f"✅ Code{suffix} copied to clipboard!"
        except Exception as exc:
            status.value = f"Error: {exc}"

    # ---------- Updates ----------
    def _on_kind_change(_):
        _reset_param_for_kind()
        refresh_preview()

    def _on_start_end_change(_):
        _reset_param_for_kind()
        refresh_preview()

    for w in (start, end):
        w.observe(_on_start_end_change, names="value")
    kind.observe(_on_kind_change, names="value")
    for w in (resolution, steps):
        w.observe(refresh_preview, names="value")
    param.observe(
        refresh_preview, names="value"
    )  # reattached in _with_param_unobserved
    for w in (mode, manual_unit, dates_text, amounts_text):
        w.observe(refresh_preview, names="value")
    copy_btn.on_click(lambda _: _copy_code(include_imports=False))
    copy_import_btn.on_click(lambda _: _copy_code(include_imports=True))

    # Initial state
    _reset_param_for_kind()
    refresh_preview()

    # ---------- UI ----------
    gen_box = VBox(
        [
            HBox([start, end, resolution], layout=Layout(gap="10px")),
            HBox([steps, param], layout=Layout(gap="10px")),
        ],
        layout=Layout(gap="10px"),
    )
    man_box = VBox(
        [manual_unit, dates_text, amounts_text],
        layout=Layout(gap="8px", width="100%"),
    )

    # Keep steps.max synced to end for nicer defaults
    def _sync_steps_max(_=None):
        new_max = abs(end.value - start.value) + 1
        if steps.max != new_max:
            steps.max = new_max

    _sync_steps_max()
    end.observe(_sync_steps_max, names="value")
    start.observe(_sync_steps_max, names="value")

    buttons_box = HBox(
        [copy_btn, copy_import_btn, status],
        layout=Layout(align_items="center", gap="10px"),
    )

    def _layout_children():
        if mode.value == "Generator":
            return [
                mode,
                kind,
                gen_box,
                buttons_box,
                plot_out,
            ]
        return [
            mode,
            man_box,
            buttons_box,
            plot_out,
        ]

    container = VBox(_layout_children(), layout=Layout(gap="12px", width="100%"))

    def _mode_refresh(_):
        container.children = _layout_children()
        if mode.value == "Generator":
            _sync_steps_max()
            _reset_param_for_kind()
        refresh_preview()

    mode.observe(_mode_refresh, names="value")

    return container


def get_temporal_evolution_factor(
    temporal_evolution: dict,
    target_date: datetime,
) -> float:
    """Linearly interpolate a scaling factor for a given date from a temporal evolution dict.

    Parameters
    ----------
    temporal_evolution : dict or None
        Dictionary mapping datetime keys to float scaling factors.
        If None or empty, returns 1.0 (no scaling).
    target_date : datetime
        The calendar date to look up the factor for.

    Returns
    -------
    float
        The interpolated scaling factor. Clamped to the nearest boundary
        value for dates outside the specified range.
    """
    if not temporal_evolution:
        return 1.0

    sorted_dates = sorted(temporal_evolution.keys())

    if len(sorted_dates) == 1:
        return temporal_evolution[sorted_dates[0]]

    # Clamp: below minimum
    if target_date <= sorted_dates[0]:
        return temporal_evolution[sorted_dates[0]]

    # Clamp: above maximum
    if target_date >= sorted_dates[-1]:
        return temporal_evolution[sorted_dates[-1]]

    # Find surrounding dates and interpolate
    for i in range(len(sorted_dates) - 1):
        lower = sorted_dates[i]
        upper = sorted_dates[i + 1]
        if lower <= target_date <= upper:
            weight = (target_date - lower).total_seconds() / (
                upper - lower
            ).total_seconds()
            return (
                temporal_evolution[lower] * (1 - weight)
                + temporal_evolution[upper] * weight
            )

    return 1.0  # fallback
