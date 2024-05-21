import bw2data as bd
import warnings
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sb
from peewee import fn
from collections import defaultdict
from scipy import sparse

from datetime import datetime
from typing import Optional, Callable
from bw2data import (  # for prepare_timex_lca_inputs
    Database,
    Method,
    Normalization,
    Weighting,
    databases,
    methods,
    normalizations,
    projects,
    weightings,
)
from bw2data.backends.schema import ActivityDataset as AD
from bw2data.backends.schema import get_id
from bw2data.errors import Brightway2Project
from bw2calc import LCA
from .timeline_builder import TimelineBuilder
from .matrix_modifier import MatrixModifier
from .dynamic_biosphere_builder import DynamicBiosphereBuilder
from .dynamic_characterization import DynamicCharacterization
from .remapping import TimeMappingDict
from .utils import extract_date_as_integer


class TimexLCA:
    """
    Class to perform time-explicit LCA calculations.

    A TimexLCA retrieves the LCI of processes occuring at explicit points in time and relinks their technosphere
    exchanges to match the technology landscape at that point in time, while keeping track of the timing of the
    resulting emissions. As such, it combines prospective and dynamic LCA approaches.

    TimexLCA first calculates a static LCA, which informs a priority-first graph traversal. From the graph traversal,
    temporal relationships between exchanges and processes are derived. Based on the timing of the processes, timex_lca
    matches the processes at the intersection between foreground and background to the best available background
    databases. This temporal relinking is achieved by using datapackages to add new time-specific processes. The new
    processes and their exchanges to other technosphere processes or biosphere flows extent the technopshere and
    biosphere matrices.
    
    Temporal information of both processes and biosphere flows are retained, allowing for dynamic LCIA.

    TimexLCA calculates:
     1) a static LCA score (`TimexLCA.static_lca.score`, same as `bw2calc.lca.score`),
     2) a static time-explicit LCA score (`TimexLCA.score`), which links LCIs to the respective background databases but without additional temporal dynamics of the biosphere flows,
     3) a dynamic time-explicit LCA score (`TimexLCA.dynamic_score`), with dynamic inventory and dynamic charaterization factors. These are provided for radiative forcing and GWP but can also be user-defined.

    Example
    -------
    >>> demand = {('my_foreground_database', 'my_process'): 1} #replace here with your functinal unit
    >>> method = ("some_method_family", "some_category", "some_method")    #replace here with your method   
    >>> database_date_dict = {'my_database': datetime.strptime("2020", "%Y"),
                              'my_foreground_database':'dynamic'} #replace here with your database dates
    >>> timex_lca = TimexLCA(demand, method, database_date_dict) 
    >>> timex_lca.build_timeline() # you can pass many optional arguments here, also for the graph traversal
    >>> timex_lca.lci()
    >>> timex_lca.static_lcia()
    >>> timex_lca.static_score
    >>> timex_lca.dynamic_lcia(metric="radiative_forcing") # different metrics can be used, e.g. "GWP", "radiative_forcing"
    >>> timex_lca.dynamic_score

    """ """"""

    def __init__(
        self,
        demand,
        method,
        database_date_dict: dict = None,
    ) -> None:
        """
        Instantiating a `TimexLCA` object calculates a static LCA, initializes time mapping dicts for activities and biosphere flows, and stores useful subsets of ids in the node_id_collection_dict.

        Parameters
        ----------
        demand : dict[object: float]
                The demand for which the LCA will be calculated. The keys can be Brightway `Node`
                instances, `(database, code)` tuples, or integer ids.
        method : tuple
                Tuple defining the LCIA method, such as `('foo', 'bar')` or default methods, such as `("EF v3.1", "climate change", "global warming potential (GWP100)")`
        database_date_dict : dict, optional
                Dictionary mapping database names to dates.
        """

        self.demand = demand
        self.method = method
        self.database_date_dict = database_date_dict

        if not self.database_date_dict:
            warnings.warn(
                "No database_date_dict provided. Treating the databases containing the functional unit as dynamic. No remapping to time explicit databases will be done."
            )
            self.database_date_dict = {key[0]: "dynamic" for key in demand.keys()}

        # Create static_only dict that excludes dynamic processes that will be exploded later. This way we only have the "background databases" that we can later link to from the dates of the timeline.
        self.database_date_dict_static_only = {
            k: v for k, v in self.database_date_dict.items() if type(v) == datetime
        }

        # Create some collections of nodes that will be useful down the line, e.g. all nodes from the background databases that link to foregroud nodes.
        self.create_node_id_collection_dict()

        # Calculate static LCA results using a custom prepare_lca_inputs function that includes all background databases in the LCA. We need all the IDs for the time mapping dict.
        fu, data_objs, remapping = self.prepare_static_lca_inputs(
            demand=self.demand, method=self.method
        )
        self.static_lca = LCA(fu, data_objs=data_objs, remapping_dicts=remapping)
        self.static_lca.lci()
        self.static_lca.lcia()

        # Create a time mapping dict that maps each activity to a activity_time_mapping_id in the format (('database', 'code'), datetime_as_integer): time_mapping_id)
        self.activity_time_mapping_dict = TimeMappingDict(
            start_id=bd.backends.ActivityDataset.select(fn.MAX(AD.id)).scalar() + 1
        )  # making sure to use unique ids for the time mapped processes by counting up from the highest current activity id

        # Create a similar dict for the biosphere flows. This is populated by the dynamic_biosphere_builder
        self.biosphere_time_mapping_dict = TimeMappingDict(start_id=0)

    ########################################
    # Main functions to be called by users #
    ########################################

    def build_timeline(
        self,
        temporal_grouping: str = "year",
        interpolation_type: str = "linear",
        edge_filter_function: Callable = None,
        cutoff: float = 1e-9,
        max_calc: float = 1e4,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Creates a `TimelineBuilder` instance that does the graph traversal (similar to bw_temporalis) and extracts all
        edges with their temporal information. Creates the `TimexLCA.timeline` of exchanges.

        Parameters
        ----------
        temporal_grouping : str, optional
            Time resolution for grouping exchanges over time in the timeline. Default is 'year', other options are
            'month', 'day', 'hour'.
        interpolation_type : str, optional
            Type of interpolation when sourcing the new producers in the time-explicit background databases. Default is
            'linear', which means linear interpolation between the closest 2 databases, other options are 'closest',
            which selects only the closest database.
        edge_filter_function : Callable, optional
            Function to skip edges in the graph traversal. Default is to skip all edges within background databases.
        cutoff: float, optional
            The cutoff value for the graph traversal. Default is 1e-9.
        max_calc: float, optional
            The maximum number of calculations to be performed by the graph traversal. Default is 1e4.
        *args : iterable
            Positional arguments for the graph traversal. for `bw_temporalis.TemporalisLCA` passed to the
            `EdgeExtractor` class, which inherits from `TemporalisLCA`.
            See `bw_temporalis` documentation for more information.
        **kwargs : dict
            Additional keyword arguments for `bw_temporalis.TemporalisLCA` passed to the EdgeExtractor class, which
            inherits from TemporalisLCA.
            See bw_temporalis documentation for more information.

        Returns
        -------
        pandas.DataFrame:
            A DataFrame containing the timeline of exchanges
            
        See also
        --------
        timex_lca.timeline_builder.TimelineBuilder: Class that builds the timeline.

        """
        if not edge_filter_function:
            warnings.warn(
                "No edge filter function provided. Skipping all edges within background databases."
            )
            skippable = []
            for db in self.database_date_dict_static_only.keys():
                skippable.extend([node.id for node in bd.Database(db)])
            self.edge_filter_function = lambda x: x in skippable

        self.temporal_grouping = temporal_grouping
        self.interpolation_type = interpolation_type
        self.cutoff = cutoff
        self.max_calc = max_calc

        self.add_static_activities_to_time_mapping_dict()  # pre-populate the activity time mapping dict with the static activities. Doing this here because we need the temporal grouping for consistent times resolution.

        # Create timeline builder that does the graph traversal (similar to bw_temporalis) and extracts all edges with their temporal information. Can later be used to build a timeline with the TimelineBuilder.build_timeline() method.
        self.timeline_builder = TimelineBuilder(
            self.static_lca,
            self.edge_filter_function,
            self.database_date_dict,
            self.database_date_dict_static_only,
            self.activity_time_mapping_dict,
            self.node_id_collection_dict,
            self.temporal_grouping,
            self.interpolation_type,
            self.cutoff,
            self.max_calc,
            *args,
            **kwargs,
        )

        self.timeline = self.timeline_builder.build_timeline()
        return self.timeline[
            [
                "date_producer",
                "producer_name",
                "date_consumer",
                "consumer_name",
                "amount",
                "interpolation_weights",
            ]
        ]

    def lci(self, build_dynamic_biosphere: Optional[bool] = True) -> None:
        """
        Calculates the time-explicit LCI.

        Generates the biosphere and technosphere modifications via the `MatrixModifier` class by calling `TimexLCA.
        build_datapackage()`. Optionally, the dynamic biosphere matrix and dynamic inventory is calculated via
        `TimexLCA.calculate_dynamic_inventory()`. Set `build_dynamic_biosphere` to False if you only want to get a new
        overall score and don't care about the timing
        of the emissions. This saves time and memory.

        Parameters
        ----------
        build_dynamic_biosphere: bool
            if True, build the dynamic biosphere matrix and calculate the dynamic LCI. Default is True.

        Returns
        -------
        None, but calls LCI calculations from bw2calc and calculates the dynamic inventory, if `build_dynamic_biosphere` is True.
        
        See also
        --------
        build_datapackage: Method to create the datapackages that contain the modifications to the technopshere and biosphere matrix using the `MatrixModifier` class.
        calculate_dynamic_inventory: Method to calculate the dynamic inventory if `build_dynamic_biosphere` is True.
        """

        if not hasattr(self, "timeline"):
            warnings.warn(
                "Timeline not yet built. Call TimexLCA.build_timeline() first."
            )
            return

        self.datapackage = (
            self.build_datapackage()
        )  # this contains the matrix modifications

        self.fu, self.data_objs, self.remapping = self.prepare_timex_lca_inputs(
            demand=self.demand,
            method=self.method,
            demand_timing_dict=self.demand_timing_dict,
        )

        self.lca = LCA(
            self.fu,
            data_objs=self.data_objs
            + self.datapackage,  # here we include the datapackage
            remapping_dicts=self.remapping,
        )

        if build_dynamic_biosphere:
            self.lca.lci(factorize=True)
            self.calculate_dynamic_inventory()
            self.lca.redo_lci(
                self.fu
            )  # to get back the original LCI - necessary because we do some redo_lci's in the dynamic inventory calculation
        else:
            self.lca.lci()

    def static_lcia(self) -> None:
        """
        Calculates static LCIA using time-explicit LCIs with the standard static characterization factors of the selected LCIA method using `bw2calc.lcia()`.

        Parameters
        ----------
        None

        Returns
        -------
        None, but stores the static score in the attribute `static_score`.
        """
        if not hasattr(self, "lca"):
            warnings.warn("LCI not yet calculated. Call TimexLCA.lci() first.")
            return
        self.lca.lcia()
        self.static_score = self.lca.score

    def dynamic_lcia(
        self,
        metric: str | None = "GWP",
        time_horizon: int | None = 100,
        fixed_time_horizon: bool | None = False,
        characterization_function_dict: dict = None,
        cumsum: bool | None = True,
    ) -> pd.DataFrame:
        """
        Calculates dynamic LCIA with the `DynamicCharacterization` class using the dynamic inventory and dynamic
        characterization functions.

        Dynamic characterization functions in the form of a dictionary {biosphere_flow_database_id:
        characterization_function} can be given by the user.
        If none are given, a set of default dynamic characterization functions based on IPCC AR6 are provided.
        These are mapped to the biosphere3 flows of the chosen static climate change impact category.
        If there is no characterization function for a biosphere flow, it will be ignored.

        Two dynamic climate change metrics are provided: "GWP" and "radiative_forcing".
        The time horizon for the impact assessment can be set with the `time_horizon` parameter, defaulting to 100 years.
        The `fixed_time_horizon` parameter determines whether the emission time horizon for all emissions is calculated from the
        functional unit (`fixed_time_horizon=True`) or from the time of the emission (`fixed_time_horizon=False`).
        The former is the implementation of the Levasseur approach(https://doi.org/10.1021/es9030003), while the latter is how conventional LCA is done.

        Parameters
        ----------
        metric : str, optional
            the metric for which the dynamic LCIA should be calculated. Default is "GWP". Available: "GWP" and "radiative_forcing"
        time_horizon: int, optional
            the time horizon for the impact assessment. Default is 100.
        fixed_time_horizon: bool, optional
            Whether the emission time horizon for all emissions is calculated from the functional unit (fixed_time_horizon=True) or from the time of the emission (fixed_time_horizon=False). Default is False.
        characterization_function_dict: dict, optional
            Dict of the form {biosphere_flow_database_id: characterization_function}. Default is None, which triggers the use of the provided dynamic characterization functions based on IPCC AR6 Chapter 7.
        cumsum: bool, optional
            Whether to calculate the cumulative sum of the characterization results. Default is True.

        Returns
        -------
        pandas.DataFrame
            A Dataframe with the characterized inventory for the chosen metric and parameters. Also stores the sum as attribute `dynamic_score`.
            
        See also
        --------
        timex_lca.dynamic_characterization.DynamicCharacterization: Class that characterizes the dynamic inventory.
        """

        if not hasattr(self, "dynamic_inventory"):
            warnings.warn(
                "Dynamic lci not yet calculated. Call TimexLCA.calculate_dynamic_lci() first."
            )

        self.metric = metric
        self.time_horizon = time_horizon
        self.fixed_time_horizon = fixed_time_horizon

        self.dynamic_characterizer = DynamicCharacterization(
            self.dynamic_inventory_df,
            self.dicts.activity,
            self.dicts.biosphere,
            self.activity_time_mapping_dict_reversed,
            self.biosphere_time_mapping_dict_reversed,
            self.demand_timing_dict,
            self.temporal_grouping,
            self.method,
            characterization_function_dict,
        )

        self.characterized_inventory = (
            self.dynamic_characterizer.characterize_dynamic_inventory(
                metric,
                time_horizon,
                fixed_time_horizon,
                cumsum,
            )
        )

        self.dynamic_score = self.characterized_inventory["amount"].sum()

        return self.characterized_inventory

    ###############################################
    # Other core functions for the inner workings #
    ###############################################

    def build_datapackage(self) -> list:
        """
        Creates the datapackages that contain the modifications to the technopshere and biosphere matrix using the `MatrixModifier` class.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of datapackages that contain the modifications to the technopshere and biosphere matrix
            
        See also
        --------
        timex_lca.matrix_modifier.MatrixModifier: Class that handles the technosphere and biosphere matrix modifications.
        """
        # mapping of the demand id to demand time
        self.demand_timing_dict = self.create_demand_timing_dict()

        # Create matrix modifier that creates the new datapackages with the exploded processes and new links to background databases.
        self.matrix_modifier = MatrixModifier(
            self.timeline, self.database_date_dict_static_only, self.demand_timing_dict
        )
        self.node_id_collection_dict["temporal_markets"] = (
            self.matrix_modifier.temporal_market_ids
        )
        self.node_id_collection_dict["temporalized_processes"] = (
            self.matrix_modifier.temporalized_process_ids
        )
        return self.matrix_modifier.create_datapackage()

    def calculate_dynamic_inventory(
        self,
    ) -> None:
        """
        Calculates the dynamic inventory, by first creating a dynamic biosphere matrix using the `DynamicBiosphereBuilder`
        class and then multiplying it with the dynamic supply array. The dynamic inventory matrix is stored in the attribute `dynamic_inventory`.
        It is also converted to a DataFrame and stored in the attribute `dynamic_inventory_df`.

        Parameters
        ----------
        None

        Returns
        -------
        None, but calculates the dynamic inventory and stores it in the attribute `dynamic_inventory` as a matrix and in `dynamic_inventory_df` as a DataFrame.
        
        See also
        --------
        timex_lca.dynamic_biosphere_builder.DynamicBiosphereBuilder: Class for creating the dynamic biosphere matrix and inventory.
        """

        if not hasattr(self, "lca"):
            warnings.warn(
                "Static Timex LCA has not been run. Call TimexLCA.lci() first."
            )
            return

        self.len_technosphere_dbs = sum(
            [len(bd.Database(db)) for db in self.database_date_dict.keys()]
        )

        self.dynamic_biosphere_builder = DynamicBiosphereBuilder(
            self.lca,
            self.activity_time_mapping_dict,
            self.biosphere_time_mapping_dict,
            self.demand_timing_dict,
            self.node_id_collection_dict,
            self.temporal_grouping,
            self.database_date_dict,
            self.database_date_dict_static_only,
        )
        self.dynamic_biomatrix = (
            self.dynamic_biosphere_builder.build_dynamic_biosphere_matrix()
        )

        # Build the dynamic inventory
        count = len(self.activity_time_mapping_dict)
        diagonal_supply_array = sparse.spdiags(
            [self.dynamic_supply_array], [0], count, count
        )  # diagnolization of supply array keeps the dimension of the process, which we want to pass as additional information to the dynamic inventory dict
        self.dynamic_inventory = self.dynamic_biomatrix @ diagonal_supply_array

        self.biosphere_time_mapping_dict_reversed = {
            v: k for k, v in self.biosphere_time_mapping_dict.items()
        }

        self.activity_time_mapping_dict_reversed = {
            v: k for k, v in self.activity_time_mapping_dict.items()
        }
        self.dynamic_inventory_df = self.create_dynamic_inventory_dataframe()

    def create_dynamic_inventory_dataframe(self) -> pd.DataFrame:
        """Brings the dynamic inventory from its matrix form in `dynamic_inventory` into the the format
        of a pandas.DataFrame, with the right structure to later apply dynamic characterization functions.

        Format needs to be:

        +------------+--------+------+----------+
        |   date     | amount | flow | activity |
        +============+========+======+==========+
        |  datetime  |   33   |  1   |    2     |
        +------------+--------+------+----------+
        |  datetime  |   32   |  1   |    2     |
        +------------+--------+------+----------+
        |  datetime  |   31   |  1   |    2     |
        +------------+--------+------+----------+

        - date: datetime, e.g. '2024-01-01 00:00:00'
        - flow: flow id
        - activity: activity id

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame, dynamic inventory in DataFrame format

        """

        dataframe_rows = []
        for i in range(self.dynamic_inventory.shape[0]):
            row_start = self.dynamic_inventory.indptr[i]
            row_end = self.dynamic_inventory.indptr[i + 1]
            for j in range(row_start, row_end):
                row = i
                col = self.dynamic_inventory.indices[j]
                value = self.dynamic_inventory.data[j]

                col_database_id = self.activity_dict.reversed[col]

                bioflow_node, date = self.biosphere_time_mapping_dict_reversed[
                    row
                ]  # indices are already the same as in the matrix, as we create an entirely new biosphere instead of adding new entries (like we do with the technosphere matrix)
                emitting_process_key, _ = self.activity_time_mapping_dict_reversed[
                    col_database_id
                ]

                dataframe_rows.append(
                    (
                        date,
                        value,
                        bioflow_node.id,
                        bd.get_activity(emitting_process_key).id,
                    )
                )

        df = pd.DataFrame(
            dataframe_rows, columns=["date", "amount", "flow", "activity"]
        )

        return df.sort_values(by=["date", "amount"], ascending=[True, False])

    #############
    # For setup #
    #############

    def prepare_static_lca_inputs(
        self,
        demand=None,
        method=None,
        weighting=None,
        normalization=None,
        demands=None,
        remapping=True,
        demand_database_last=True,
    ) -> tuple:
        """
        Prepare LCA input arguments in Brightway2.5 style.

        Adapted bw2data.compat.py

        The difference to the original method is that we load all available databases into the matrices instead of just the ones depending on the demand.
        We need this for the creation of the time mapping dict that creates a mapping between the producer id and the reference timing of the databases in the `database_date_dict`.

        Parameters
        ----------
        demand : dict[object: float]
            The demand for which the LCA will be calculated. The keys can be Brightway `Node`
            instances, `(database, code)` tuples, or integer ids.
        method : tuple
            Tuple defining the LCIA method, such as `('foo', 'bar')`. Only needed if not passing
            `data_objs`.
        weighting : tuple
            Tuple defining the LCIA weighting, such as `('foo', 'bar')`. Only needed if not passing
            `data_objs`.
        normalization: str
        demands: list of dicts of demands
        remapping: bool
            If True, remap dictionaries
        demand_database_last: bool
            If True, add the demand databases last in the list `database_names`.

        Returns
        -------
        tuple
            Indexed demand, data objects, and remapping dictionaries
            
        See also
        --------
        bw2data.compat.prepare_lca_inputs: Original code this function is adapted from (https://github.com/brightway-lca/brightway2-data/blob/main/bw2data/compat.py).

        """
        if not projects.dataset.data.get("25"):
            raise Brightway2Project(
                "Please use `projects.migrate_project_25` before calculating using Brightway 2.5"
            )

        databases.clean()
        data_objs = []
        remapping_dicts = None

        demand_database_names = [
            db_label for db_label in self.database_date_dict.keys()
        ]  # Load all databases that could lateron be linked to

        if demand_database_names:
            database_names = set.union(
                *[
                    Database(db_label).find_graph_dependents()
                    for db_label in demand_database_names
                ]
            )

            if demand_database_last:
                database_names = [
                    x for x in database_names if x not in demand_database_names
                ] + demand_database_names

            data_objs.extend([Database(obj).datapackage() for obj in database_names])

            if remapping:
                # This is technically wrong - we could have more complicated queries
                # to determine what is truly a product, activity, etc.
                # However, for the default database schema, we know that each node
                # has a unique ID, so this won't produce incorrect responses,
                # just too many values. As the dictionary only exists once, this is
                # not really a problem.
                reversed_mapping = {
                    i: (d, c)
                    for d, c, i in AD.select(AD.database, AD.code, AD.id)
                    .where(AD.database << database_names)
                    .tuples()
                }
                remapping_dicts = {
                    "activity": reversed_mapping,
                    "product": reversed_mapping,
                    "biosphere": reversed_mapping,
                }

        if method:
            assert method in methods
            data_objs.append(Method(method).datapackage())
        if weighting:
            assert weighting in weightings
            data_objs.append(Weighting(weighting).datapackage())
        if normalization:
            assert normalization in normalizations
            data_objs.append(Normalization(normalization).datapackage())

        if demands:
            indexed_demand = [{get_id(k): v for k, v in dct.items()} for dct in demands]
        elif demand:
            indexed_demand = {get_id(k): v for k, v in demand.items()}
        else:
            indexed_demand = None

        return indexed_demand, data_objs, remapping_dicts

    def prepare_timex_lca_inputs(
        self,
        demand=None,
        method=None,
        demand_timing_dict=None,
        weighting=None,
        normalization=None,
        demands=None,
        remapping=True,
        demand_database_last=True,
    ) -> tuple:
        """
        Prepare LCA input arguments in Brightway 2.5 style.

        ORIGINALLY FROM bw2data.compat.py

        Changes include:
        - always load all databases in demand_database_names
        - indexed_demand has the id of the new consumer_id of the "exploded" demand

        Parameters
        ----------
        demand : dict[object: float]
            The demand for which the LCA will be calculated. The keys can be Brightway `Node`
            instances, `(database, code)` tuples, or integer ids.
        method : tuple
            Tuple defining the LCIA method, such as `('foo', 'bar')`. Only needed if not passing
            `data_objs`.
        demand_timing_dict: dict
            Dictionary mapping demand ids to their timing.
        weighting : tuple
            Tuple defining the LCIA weighting, such as `('foo', 'bar')`. Only needed if not passing
            `data_objs`.
        normalization: str
        demands: list of dicts of demands
        remapping: bool
            If True, remap dictionaries
        demand_database_last: bool
            If True, add the demand databases last in the list `database_names`.

        Returns
        -------
        tuple
            Indexed demand, data objects, and remapping dictionaries
            
        See also
        --------
        bw2data.compat.prepare_lca_inputs: Original code this function is adapted from (https://github.com/brightway-lca/brightway2-data/blob/main/bw2data/compat.py).
        """

        if not projects.dataset.data.get("25"):
            raise Brightway2Project(
                "Please use `projects.migrate_project_25` before calculating using Brightway 2.5"
            )

        databases.clean()
        data_objs = []
        remapping_dicts = None

        demand_database_names = [
            db_label for db_label in self.database_date_dict.keys()
        ]  # Load all databases that could lateron be linked to

        if demand_database_names:
            database_names = set.union(
                *[
                    Database(db_label).find_graph_dependents()
                    for db_label in demand_database_names
                ]
            )

            if demand_database_last:
                database_names = [
                    x for x in database_names if x not in demand_database_names
                ] + demand_database_names

            data_objs.extend([Database(obj).datapackage() for obj in database_names])

            if remapping:
                # This is technically wrong - we could have more complicated queries
                # to determine what is truly a product, activity, etc.
                # However, for the default database schema, we know that each node
                # has a unique ID, so this won't produce incorrect responses,
                # just too many values. As the dictionary only exists once, this is
                # not really a problem.
                reversed_mapping = {
                    i: (d, c)
                    for d, c, i in AD.select(AD.database, AD.code, AD.id)
                    .where(AD.database << database_names)
                    .tuples()
                }
                remapping_dicts = {
                    "activity": reversed_mapping,
                    "product": reversed_mapping,
                    "biosphere": reversed_mapping,
                }

        if method:
            assert method in methods
            data_objs.append(Method(method).datapackage())
        if weighting:
            assert weighting in weightings
            data_objs.append(Weighting(weighting).datapackage())
        if normalization:
            assert normalization in normalizations
            data_objs.append(Normalization(normalization).datapackage())

        if demands:
            indexed_demand = [
                {
                    self.activity_time_mapping_dict[
                        (
                            bd.get_node(id=bd.get_id(k)).key,
                            self.demand_timing_dict[bd.get_id(k)],
                        )
                    ]: v
                    for k, v in dct.items()
                }
                for dct in demands
            ]
        elif demand:
            indexed_demand = {
                self.activity_time_mapping_dict[
                    (
                        bd.get_node(id=bd.get_id(k)).key,
                        self.demand_timing_dict[bd.get_id(k)],
                    )
                ]: v
                for k, v in demand.items()
            }
        else:
            indexed_demand = None

        return indexed_demand, data_objs, remapping_dicts

    def create_node_id_collection_dict(self) -> None:
        """
        Creates a dict of collections of nodes that will be useful down the line, e.g. to determine static nodes for the graph traversal or create the dynamic biosphere matrix.
        Available collections are:

        - ``demand_database_names``: set of database names of the demand processes
        - ``demand_dependent_database_names``: set of database names of all processes that depend on the demand processes
        - ``demand_dependent_background_database_names``: set of database names of all processes that depend on the demand processes and are in the background databases
        - ``demand_dependent_background_node_ids``: set of node ids of all processes that depend on the demand processes and are in the background databases
        - ``foreground_node_ids``: set of node ids of all processes that are not in the background databases
        - ``first_level_background_node_ids_static``: set of node ids of all processes that are in the background databases and are directly linked to the demand processes
        - ``first_level_background_node_ids_all``: like first_level_background_node_ids_static, but includes first level background processes from other time explicit databases.

        Parameters
        ----------
            None

        Returns
        -------
            None, but adds the `node_id_collection_dict containing` the above-mentioned collections.
        """
        self.node_id_collection_dict = {}

        # Original variable names preserved, set types for performance and uniqueness
        demand_database_names = {
            db
            for db in self.database_date_dict.keys()
            if db not in self.database_date_dict_static_only.keys()
        }
        self.node_id_collection_dict["demand_database_names"] = demand_database_names

        demand_dependent_database_names = set()
        for db in demand_database_names:
            demand_dependent_database_names.update(bd.Database(db).find_dependents())
        self.node_id_collection_dict["demand_dependent_database_names"] = (
            demand_dependent_database_names
        )

        demand_dependent_background_database_names = (
            demand_dependent_database_names & self.database_date_dict_static_only.keys()
        )
        self.node_id_collection_dict["demand_dependent_background_database_names"] = (
            demand_dependent_background_database_names
        )

        demand_dependent_background_node_ids = {
            node.id
            for db in demand_dependent_background_database_names
            for node in bd.Database(db)
        }
        self.node_id_collection_dict["demand_dependent_background_node_ids"] = (
            demand_dependent_background_node_ids
        )

        foreground_node_ids = {
            node.id
            for db in self.database_date_dict.keys()
            if db not in self.database_date_dict_static_only.keys()
            for node in bd.Database(db)
        }
        self.node_id_collection_dict["foreground_node_ids"] = foreground_node_ids

        first_level_background_node_ids_static = set()
        first_level_background_node_ids_all = set()
        for node_id in foreground_node_ids:
            node = bd.get_node(id=node_id)
            for exc in node.technosphere():
                if exc.input["database"] in self.database_date_dict_static_only.keys():
                    first_level_background_node_ids_static.add(exc.input.id)
                    for background_db in self.database_date_dict_static_only.keys():
                        try:
                            other_node = bd.get_node(
                                **{
                                    "database": background_db,
                                    "name": exc.input["name"],
                                    "product": exc.input["reference product"],
                                    "location": exc.input["location"],
                                }
                            )
                            first_level_background_node_ids_all.add(other_node.id)
                        except Exception as e:
                            warnings.warn(
                                f"Failed to find process in database {background_db} for name='{exc.input['name']}', reference product='{exc.input['reference product']}', location='{exc.input['location']}': {e}"
                            )
                            pass

        self.node_id_collection_dict["first_level_background_node_ids_static"] = (
            first_level_background_node_ids_static
        )
        self.node_id_collection_dict["first_level_background_node_ids_all"] = (
            first_level_background_node_ids_all
        )

    def add_static_activities_to_time_mapping_dict(self) -> None:
        """
        Adds all activities from the static LCA to `activity_time_mapping_dict`, an instance of `TimeMappingDict`.
        This gives a unique mapping in the form of (('database', 'code'), datetime_as_integer): time_mapping_id) that is later used to uniquely identify time-resolved processes. This is the pre-population of the time mapping dict with the static activities. Further time-explicit activities (from other temporalized background databases) are added lateron by in the TimelineBuilder.

        Parameters
        ----------
        None

        Returns
        -------
        None but adds the activities to the `activity_time_mapping_dict`
        """
        for id in self.static_lca.dicts.activity.keys():  # activity ids
            key = self.static_lca.remapping_dicts["activity"][
                id
            ]  # ('database', 'code')
            time = self.database_date_dict[
                key[0]
            ]  # datetime (or 'dynamic' for foreground processes)
            if type(time) == str:  # if 'dynamic', just add the string
                self.activity_time_mapping_dict.add((key, time), unique_id=id)
            elif type(time) == datetime:
                self.activity_time_mapping_dict.add(
                    (key, extract_date_as_integer(time, self.temporal_grouping)),
                    unique_id=id,
                )  # if datetime, map to the date as integer
            else:
                warnings.warn(f"Time of activity {key} is neither datetime nor str.")

    def create_demand_timing_dict(self) -> dict:
        """
        Generate a dictionary that maps producer (key) to timing (value) for the demands in the product system.
        It searches the timeline for those rows that contain the functional units (demand-processes as producer and -1 as consumer) and returns the time of the demand as an integer.
        Time of demand can have flexible resolution (year=YYYY, month=YYYYMM, day=YYYYMMDD, hour=YYYYMMDDHH) defined in `temporal_grouping`.

        Parameters
        ----------
        None

        Returns
        -------
        Dictionary mapping producer ids to reference timing for the specified demands.
        """
        demand_ids = [bd.get_activity(key).id for key in self.demand.keys()]
        demand_rows = self.timeline[
            self.timeline["producer"].isin(demand_ids)
            & (self.timeline["consumer"] == -1)
        ]
        self.demand_timing_dict = {
            row.producer: row.hash_producer for row in demand_rows.itertuples()
        }
        return self.demand_timing_dict

    ######################################
    # For creating human-friendly output #
    ######################################

    def create_labelled_technosphere_dataframe(self) -> pd.DataFrame:
        """
        Returns the technosphere matrix as a dataframe with comprehensible labels instead of ids.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame, technosphere matrix as a pandas.DataFrame with comprehensible labels instead of ids.
        """

        if not hasattr(self, "activity_time_mapping_labels_dict"):
            self.activity_time_mapping_labels_dict = self.activity_time_mapping_dict.reversed()
            
            for ((database, code), time), idx in self.activity_time_mapping_dict.items():
                database_time = self.database_date_dict[database]
                if type(database_time) != str:
                    if extract_date_as_integer(database_time, self.temporal_grouping) != time:
                        self.activity_time_mapping_labels_dict[idx] = (("temporal market", code), time)
        
        df = pd.DataFrame(self.technosphere_matrix.toarray())
        df.rename(  # from matrix id to activity id
            index=self.dicts.activity.reversed,
            columns=self.dicts.activity.reversed,
            inplace=True,
        )
        df.rename(  # from activity id to ((database, code), time)
            index=self.activity_time_mapping_labels_dict,
            columns=self.activity_time_mapping_labels_dict,
            inplace=True,
        )
        return df

    def create_labelled_biosphere_dataframe(self) -> pd.DataFrame:
        """
        Returns the biosphere matrix as a pandas.DataFrame with comprehensible labels instead of ids.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame, biosphere matrix as a pandas.DataFrame with comprehensible labels instead of ids.
        """
        
        if not hasattr(self, "activity_time_mapping_labels_dict"):
            self.activity_time_mapping_labels_dict = self.activity_time_mapping_dict.reversed()
            
            for ((database, code), time), idx in self.activity_time_mapping_dict.items():
                database_time = self.database_date_dict[database]
                if type(database_time) != str:
                    if extract_date_as_integer(database_time, self.temporal_grouping) != time:
                        self.activity_time_mapping_labels_dict[idx] = (("temporal market", code), time)
                        
        df = pd.DataFrame(self.biosphere_matrix.toarray())
        df.rename(  # from matrix id to activity id
            index=self.dicts.biosphere.reversed,
            columns=self.dicts.activity.reversed,
            inplace=True,
        )
        df.rename(
            index=self.remapping_dicts["biosphere"],  # from activity id to bioflow name
            columns=self.activity_time_mapping_labels_dict,  # from activity id to ((database, code), time)
            inplace=True,
        )

        return df

    def create_labelled_dynamic_biosphere_dataframe(self) -> pd.DataFrame:
        """
        Returns the dynamic biosphere matrix as a dataframe with comprehensible labels instead of ids.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame, dynamic biosphere matrix as a pandas.DataFrame with comprehensible labels instead of ids.
        """
        df = pd.DataFrame(self.dynamic_biomatrix.toarray())
        df.rename(  # from matrix id to activity id
            index=self.biosphere_time_mapping_dict_reversed,
            columns=self.dicts.activity.reversed,
            inplace=True,
        )
        df.rename(  # from activity id to ((database, code), time)
            columns=self.activity_time_mapping_dict.reversed(),
            inplace=True,
        )

        df = df.loc[(df != 0).any(axis=1)]  # For readablity, remove all-zero rows

        return df

    def plot_dynamic_inventory(self, bio_flows, cumulative=False) -> None:
        """
        Simple plot of dynamic inventory of a biosphere flow over time, with optional cumulative plotting.

        Parameters
        ----------
        bio_flows : list of int
            database ids of the biosphere flows to plot.
        cumulative : bool
            if True, plot cumulative amounts over time

        Returns
        -------
        none, but shows a plot
        """
        plt.figure(figsize=(14, 6))

        filtered_df = self.dynamic_inventory_df[
            self.dynamic_inventory_df["flow"].isin(bio_flows)
        ]
        aggregated_df = filtered_df.groupby("date").sum()["amount"].reset_index()

        if cumulative:
            aggregated_df["amount"] = np.cumsum(aggregated_df["amount"])

        plt.plot(
            aggregated_df["date"], aggregated_df["amount"], marker="o", linestyle="none"
        )

        plt.ylim(bottom=0)
        plt.xlabel("time")
        plt.ylabel("amount [kg]")
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to make room for the rotated date labels
        plt.show()

    def plot_dynamic_characterized_inventory(
        self,
        cumsum: bool = False,
        sum_emissions_within_activity: bool = False,
        sum_activities: bool = False,
    ) -> None:
        """
        Plot the characterized inventory of the dynamic LCI in a very simple plot.
        Legend and title are selcted automatically based on the chosen metric.

        Parameters
        ----------
        cumsum : bool
            if True, plot cumulative amounts over time
        sum_emissions_within_activity : bool
            if True, sum emissions within each activity over time
        sum_activities : bool
            if True, sum emissions over all activities over time

        Returns
        -------
        none, but shows a plot
        """
        if cumsum:
            amount = "amount_sum"
        else:
            amount = "amount"

        if not hasattr(self, "characterized_inventory"):
            warnings.warn(
                "Characterized inventory not yet calculated. Call TimexLCA.characterize_dynamic_lci() first."
            )
            return

        # Fetch the inventory to use in plotting, modify based on flags
        plot_data = self.characterized_inventory.copy()

        if sum_emissions_within_activity:
            plot_data = plot_data.groupby(["date", "activity_name"]).sum().reset_index()
            plot_data["amount_sum"] = plot_data["amount"].cumsum()

        if sum_activities:
            plot_data = plot_data.groupby("date").sum().reset_index()
            plot_data["amount_sum"] = plot_data["amount"].cumsum()
            plot_data["activity_label"] = "All activities"

        else:
            plot_data["activity_label"] = plot_data.apply(
                lambda row: bd.get_activity(row["activity_name"])["name"], axis=1
            )
        # Plotting
        plt.figure(figsize=(14, 6))
        axes = sb.scatterplot(x="date", y=amount, hue="activity_label", data=plot_data)

        # Determine the plotting labels and titles based on the characterization method
        if self.metric == "radiative_forcing":
            label_legend = "Radiative forcing [W/m2]"
            title = "Radiative forcing"
        elif self.metric == "GWP":
            label_legend = "GWP [kg CO2-eq]"
            title = "GWP"

        if self.fixed_time_horizon:
            suptitle = f" \nTH of {self.time_horizon} years starting at FU,"
        else:
            suptitle = f" \nTH of {self.time_horizon} years starting at each emission,"

        suptitle += f" temporal resolution of inventories: {self.temporal_grouping}"

        # Determine y-axis limit flexibly
        if plot_data[amount].min() < 0:
            ymin = plot_data[amount].min() * 1.1
        else:
            ymin = 0

        axes.set_title(title)
        axes.set_axisbelow(True)
        axes.set_ylim(bottom=ymin)
        axes.set_ylabel(label_legend)
        axes.set_xlabel("Time")

        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles[::-1], labels[::-1])
        plt.title(title, fontsize=16, y=1.05)
        plt.suptitle(suptitle, fontsize=12, y=0.95)
        plt.grid()
        plt.show()

    #########
    # Other #
    #########

    def remap_inventory_dicts(self) -> None:
        """
        Give a warning if users want to use bw25's original mapping function.

        Parameters
        ----------
        None

        Returns
        -------
        None, but gives a warning

        """
        warnings.warn(
            "bw25's original mapping function doesn't work with our new time-mapped matrix entries. The Timex mapping can be found in acvitity_time_mapping_dict and biosphere_time_mapping_dict."
        )
        return

    def __getattr__(self, name):
        """
        Delegate attribute access to the self.lca object if the attribute
        is not found in the TimexLCA instance itself, excluding special attributes.
        """
        if name.startswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if hasattr(self.lca, name):
            return getattr(self.lca, name)
        elif hasattr(self.dynamic_biosphere_builder, name):
            return getattr(self.dynamic_biosphere_builder, name)
        else:
            raise AttributeError(
                f"'TimexLCA' object and its 'lca'- and dynamic_biosphere_builder- attributes have no attribute '{name}'"
            )
