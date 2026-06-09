from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Callable, KeysView, Union

import bw2data as bd
import numpy as np
import pandas as pd
from bw2calc import LCA
from bw2data.configuration import labels
from loguru import logger

from .edge_extractor import Edge, EdgeExtractor, EdgeExtractorBFS
from .utils import (
    convert_date_string_to_datetime,
    extract_date_as_integer,
    extract_date_as_string,
    round_datetime,
)


class TimelineBuilder:
    """
    Class for building a process timeline based on the temporal distributions of their exchanges.

    First, the `EdgeExtractor` does a priority-first graph traversal and extracts a timeline of
    exchanges (edge_timeline) with temporal information. Identical edges within temporal grouping
    (e.g. year, month, day, hour) are then grouped and the amount of exchanges is summed up.
    """

    def __init__(
        self,
        base_lca: LCA,
        starting_datetime: datetime,
        edge_filter_function: Callable,
        database_dates: dict,
        database_dates_static: dict,
        activity_time_mapping: dict,
        node_collections: dict,
        nodes: dict,
        temporal_grouping: str = "year",
        interpolation_type: str = "linear",
        cutoff: float = 1e-9,
        max_calc: int = 2000,
        graph_traversal: str = "priority",
        demand_temporal_distributions: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        base_lca: LCA
            A static LCA object.
        starting_datetime: datetime | str, optional
            Point in time when the demand occurs.
        edge_filter_function: Callable
            A callable that filters edges. If not provided, a function that always returns False is used.
        database_dates: dict
            A dictionary mapping databases to dates.
        database_dates_static: dict
            same as database_dates, but excluding the "dynamic" foreground databases.
        activity_time_mapping: dict
          A dictionary to map processes to specific times.
        node_collections: dict
            A dictionary collecting useful subsets of node ids.
        nodes: dict
            A dictionary {node_id: 'bw2data.backends.proxies.Activity'} for all nodes.
        temporal_grouping: str, optional
            The temporal grouping to be used. Default is "year".
        interpolation_type: str, optional
            The type of interpolation to be used to select the background databases. Default is "linear".
        cutoff:
            The cutoff value for the graph traversal. Default is 1e-9.
        max_calc:
            The maximum number of calculations to be performed by the graph traversal. Default is 2000.
        args:   Variable length argument list
            Keyword arguments passed to the EdgeExtractor which inherits from TemporalisLCA. Here, things like the further settings for graph traversal can be set. For details, see bw_temporalis.TemporalisLCA.
        kwargs: Arbitrary keyword arguments
            Keyword arguments passed to the EdgeExtractor which inherits from TemporalisLCA.
        """
        self.base_lca = base_lca
        self.starting_datetime = starting_datetime
        self.edge_filter_function = edge_filter_function
        self.database_dates = database_dates
        self.database_dates_static = database_dates_static
        self.activity_time_mapping = activity_time_mapping
        self.node_collections = node_collections
        self.nodes = nodes
        self.temporal_grouping = temporal_grouping
        self.interpolation_type = interpolation_type
        self.cutoff = cutoff
        self.max_calc = max_calc
        self._logged_reference_date_below_range = False
        self._logged_reference_date_above_range = False

        # Finding indices of activities from the connected background databases that are known to be static, i.e. have no temporal distributions connecting to them.
        # These will be be skipped in the graph traversal.
        static_background_activity_ids = {
            node_id
            for node_id in self.node_collections["background"]
            if node_id not in self.node_collections["first_level_background_static"]
        }

        logger.info("Traversing supply chain graph...")
        self.demand_temporal_distributions = demand_temporal_distributions or {}
        if graph_traversal == "bfs":
            self.edge_extractor = EdgeExtractorBFS(
                lca_object=base_lca,
                starting_datetime=self.starting_datetime,
                edge_filter_function=edge_filter_function,
                cutoff=self.cutoff,
                static_activity_indices=set(static_background_activity_ids),
                demand_temporal_distributions=self.demand_temporal_distributions,
            )
        elif graph_traversal == "priority":
            self.edge_extractor = EdgeExtractor(
                base_lca,
                starting_datetime=self.starting_datetime,
                *args,
                edge_filter_function=edge_filter_function,
                cutoff=self.cutoff,
                max_calc=self.max_calc,
                static_activity_indices=set(static_background_activity_ids),
                demand_temporal_distributions=self.demand_temporal_distributions,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown graph_traversal '{graph_traversal}'. Use 'priority' or 'bfs'."
            )
        self.edge_timeline = self.edge_extractor.build_edge_timeline()

    def build_timeline(self) -> pd.DataFrame:
        """
        Create a DataFrame with grouped, time-explicit edges and, for each grouped edge,
        interpolate to the database with the closest time of representativeness.

        It uses the edge_timeline, an output from the graph traversal in EdgeExtractor.
        Edges from same producer to same consumer that occur at different times within
        the same time window (temporal_grouping) are grouped together.
        Possible temporal groupings are "year", "month", "day" and "hour".

        For edges between foreground and background system, the column "temporal_market_shares"
        assigns the ratio [0-1] of the edge's amount to be taken from the database with the closest
        time of representativeness. If a process is in the foreground system only, the interpolation weight is set to None.

        Available interpolation types are:

        - "linear": linear interpolation between the two closest databases, based on temporal distance.

        - "closest": closest database is assigned 1

        Parameters
        ----------
        None
            (all are already passed during instantiation)

        Returns
        -------
        pd.DataFrame
            A timeline with grouped, time-explicit edges and temporal_market_shares to background databases.
        """
        logger.info("Building timeline...")

        # check if database names match with databases in BW project
        self.check_database_names()

        # Check if temporal_grouping is a valid value
        valid_temporal_groupings = ["year", "month", "day", "hour"]
        if self.temporal_grouping not in valid_temporal_groupings:
            raise ValueError(
                f"Invalid value for 'temporal_grouping'. Allowed values are {valid_temporal_groupings}."
            )

        # Extract edge data into a list of dictionaries
        edges_data = [self.extract_edge_data(edge) for edge in self.edge_timeline]

        # Convert list of dictionaries to dataframe
        edges_df = pd.DataFrame(edges_data)

        # adjust the sign for substitution exchanges:
        edges_df["amount"] = edges_df["amount"] * edges_df["edge_type"].apply(
            lambda x: self.adjust_sign_of_amount_based_on_edge_type(x)
        )

        # Explode datetime and amount columns: each row with multiple dates and amounts is exploded into multiple rows with one date and one amount
        edges_df = edges_df.explode(["consumer_date", "producer_date", "amount"])

        # Create a hashable key from temporal_evolution dicts for dedup and groupby
        # (dicts are unhashable, so we need a hashable proxy column)
        edges_df["_te_key"] = edges_df["temporal_evolution"].apply(
            lambda d: tuple(sorted(d.items())) if isinstance(d, dict) else None
        )

        dedup_cols = [c for c in edges_df.columns if c != "temporal_evolution"]
        edges_df.drop_duplicates(subset=dedup_cols, inplace=True)
        edges_df = edges_df[edges_df["amount"] != 0]

        # For the Functional Unit: set consumer date = producer date as it occurs at the same time
        edges_df.loc[edges_df["consumer"] == -1, "consumer_date"] = edges_df.loc[
            edges_df["consumer"] == -1, "producer_date"
        ]

        edges_df["rounded_consumer_date"] = edges_df["consumer_date"].apply(
            lambda x: round_datetime(x, self.temporal_grouping)
        )
        edges_df["rounded_producer_date"] = edges_df["producer_date"].apply(
            lambda x: round_datetime(x, self.temporal_grouping)
        )

        # extract grouping time of consumer and producer: processes occuring at different times within in the same time window of grouping get the same grouping time
        edges_df["consumer_grouping_time"] = edges_df["rounded_consumer_date"].apply(
            lambda x: extract_date_as_string(x, self.temporal_grouping)
        )
        edges_df["producer_grouping_time"] = edges_df["rounded_producer_date"].apply(
            lambda x: extract_date_as_string(x, self.temporal_grouping)
        )

        # Collapse coefficients that converge on the same time-mapped edge via
        # different supply-chain paths. The per-unit coefficient of a time-mapped
        # edge must be emitted once per (consumer cohort bucket, producer bucket),
        # but when several consumer paths reach one producer cohort (e.g. a
        # backward relative TD where a product demanded in several years is served
        # by one earlier-built cohort) the same coefficient is replicated once per
        # path. The `drop_duplicates` above keys on raw datetimes, which drift
        # (datetime64[s] + timedelta64[Y] uses an average-length year), so those
        # duplicates survive and would be summed by the groupby below.
        #
        # The relative offset (producer - consumer) IS the edge's own TD entry: it
        # is path-independent and exact, so it separates "same edge entry reached
        # by multiple paths" (drop here) from "distinct sub-window entries that
        # round into the same bucket" (different offset -> kept and summed below).
        edges_df["relative_offset"] = pd.to_datetime(
            edges_df["producer_date"], errors="coerce"
        ) - pd.to_datetime(edges_df["consumer_date"], errors="coerce")
        edges_df.drop_duplicates(
            subset=[
                "producer",
                "consumer",
                "consumer_grouping_time",
                "producer_grouping_time",
                "relative_offset",
                "amount",
                "edge_type",
                "_te_key",
                "temporal_evolution_reference",
            ],
            inplace=True,
        )

        # group unique pair of consumer and producer with the same grouping times
        # _te_key ensures exchanges with different temporal_evolution dicts stay separate
        grouped_edges = (
            edges_df.groupby(
                [
                    "producer_grouping_time",
                    "consumer_grouping_time",
                    "producer",
                    "consumer",
                    "_te_key",
                    "temporal_evolution_reference",
                ],
                dropna=False,
            )
            .agg({"amount": "sum"})
            .reset_index()
        )
        # Reconstruct temporal_evolution dicts from the hashable _te_key
        grouped_edges["temporal_evolution"] = grouped_edges["_te_key"].apply(
            lambda k: dict(k) if isinstance(k, tuple) else None
        )
        grouped_edges.drop(columns=["_te_key"], inplace=True)

        # Convert grouping times to datetime with a unique-value cache to avoid repeated parsing
        unique_grouping_strings = set(grouped_edges["producer_grouping_time"]).union(
            set(grouped_edges["consumer_grouping_time"])
        )
        datetime_cache = {
            value: convert_date_string_to_datetime(self.temporal_grouping, value)
            for value in unique_grouping_strings
        }

        grouped_edges["date_producer"] = grouped_edges["producer_grouping_time"].map(
            datetime_cache
        )
        grouped_edges["date_consumer"] = grouped_edges["consumer_grouping_time"].map(
            datetime_cache
        )

        # add dates as integers as hashes to the DataFrame
        hash_cache = {
            dt: extract_date_as_integer(dt, time_res=self.temporal_grouping)
            for dt in datetime_cache.values()
        }
        grouped_edges["hash_producer"] = grouped_edges["date_producer"].map(hash_cache)
        grouped_edges["hash_consumer"] = grouped_edges["date_consumer"].map(hash_cache)

        # add new processes to activity_time_mapping
        for row in grouped_edges.itertuples():
            self.activity_time_mapping.add(
                (
                    ("temporalized", self.nodes[row.producer]["code"]),
                    row.hash_producer,
                )
            )

        # store the ids from the time_mapping in DataFrame
        grouped_edges["time_mapped_producer"] = [
            self.get_time_mapping_key(producer, hash_producer)
            for producer, hash_producer in zip(
                grouped_edges["producer"], grouped_edges["hash_producer"]
            )
        ]

        grouped_edges["time_mapped_consumer"] = [
            self.get_time_mapping_key(consumer, hash_consumer)
            if consumer != -1
            else -1
            for consumer, hash_consumer in zip(
                grouped_edges["consumer"], grouped_edges["hash_consumer"]
            )
        ]

        # Add temporal_market_shares to background databases to the DataFrame
        grouped_edges = self.add_column_temporal_market_shares_to_timeline(
            grouped_edges,
            interpolation_type=self.interpolation_type,
        )

        # Retrieve producer and consumer names
        grouped_edges["producer_name"] = [
            self.nodes[producer]["name"] for producer in grouped_edges["producer"]
        ]
        grouped_edges["consumer_name"] = [
            self.get_consumer_name(consumer) for consumer in grouped_edges["consumer"]
        ]

        # Reorder columns
        grouped_edges = grouped_edges[
            [
                "hash_producer",
                "time_mapped_producer",
                "date_producer",
                "producer",
                "producer_name",
                "hash_consumer",
                "time_mapped_consumer",
                "date_consumer",
                "consumer",
                "consumer_name",
                "amount",
                "temporal_market_shares",
                "temporal_evolution",
                "temporal_evolution_reference",
            ]
        ]

        return grouped_edges

    ###################################################
    # underlying functions called by build_timeline() #
    ###################################################

    def check_database_names(self) -> None:
        """
        Check that the strings of the databases exist in the databases of the Brightway project.

        """
        for db in self.database_dates_static.keys():
            assert (
                db in bd.databases
            ), f"{db} is not in your Brightway project databases."

    def extract_edge_data(self, edge: Edge) -> dict:
        """
        Stores the attributes of an Edge instance in a dictionary.

        Parameters
        ----------
        edge: Edge
            Edge instance

        Returns
        -------
        dict
            Dictionary with the attributes of the edge instance.
        """
        try:
            consumer_date = edge.abs_td_consumer.date
            consumer_date = np.array(
                [consumer_date for i in range(len(edge.td_producer))]
            ).T.flatten()
        except AttributeError:
            consumer_date = None

        return {
            "producer": edge.producer,
            "consumer": edge.consumer,
            "consumer_date": consumer_date,
            "producer_date": edge.abs_td_producer.date,
            "amount": edge.abs_td_producer.amount,
            "edge_type": edge.edge_type,
            "temporal_evolution": edge.temporal_evolution,
            "temporal_evolution_reference": edge.temporal_evolution_reference,
        }

    def adjust_sign_of_amount_based_on_edge_type(self, edge_type):
        """
        It checks if the an exchange is of type substitution or a technosphere exchange,
        based on bw2data labelling convention, and adjusts the amount accordingly.
        Flips the sign of the amount value in the timeline for substitution (positive technosphere) exchanges.

        Parameters
        ----------
        edge_type: str
            Type of the edge, as defined in the exchange data.

        Returns
        -------
        int
            Multiplier for the amount value, 1 for technosphere exchanges, -1 for substitution exchanges.

        """

        if edge_type in labels.technosphere_negative_edge_types:
            return 1  # Variants of technosphere labels

        if edge_type in labels.technosphere_positive_edge_types:
            return -1 if edge_type in labels.substitution_edge_types else 1

        raise TypeError(f"Unrecognized type in this edge: {edge_type}")

    def get_time_mapping_key(self, node_id: int, node_hash: int) -> int:
        """
        Returns the time_mapping_id (key) from the activity_time_mapping for a given node.

        Parameters
        ----------
        node_id: int
            database id of the node.
        node_hash: int
            datetime_as_integer of the node.

        Returns
        -------
        int
            time_mapping_id (key) of the corresponding time-mapped activity.

        """
        try:
            return self.activity_time_mapping[
                (("temporalized", self.nodes[node_id]["code"]), node_hash)
            ]
        except KeyError:
            return self.activity_time_mapping[((self.nodes[node_id].key), node_hash)]

    def add_column_temporal_market_shares_to_timeline(
        self,
        tl_df: pd.DataFrame,
        interpolation_type: str = "linear",
    ) -> pd.DataFrame:
        """
        Add a column to a timeline with the weights for an interpolation between
        the two nearest dates, from the list of dates of the available databases.

        Parameters
        ----------
        tl_df: pd.DataFrame
            Timeline as a DataFrame.
        interpolation_type: str, optional
            Type of interpolation between the nearest lower and higher dates.
            Available options: "linear" and "nearest", defaulting to "linear".

        Returns
        -------
        pd.DataFrame
            Timeline as a DataFrame with a column 'temporal_market_shares' added,
            this column looks like {database_name: weight, database_name: weight}.
        """
        if not self.database_dates_static:
            tl_df["temporal_market_shares"] = None
            logger.info(
                "No time-explicit databases are provided. Mapping to time-explicit databases is not possible.",
            )
            return tl_df

        dates_list = [
            date
            for date in self.database_dates_static.values()
            if isinstance(date, datetime)
        ]
        if "date_producer" not in list(tl_df.columns):
            raise ValueError("The timeline does not contain dates.")

        sorted_dates = tuple(sorted(dates_list))

        # create reversed dict {date: database} with only static "background" db's
        self.reversed_database_dates = {
            v: k
            for k, v in self.database_dates_static.items()
            if isinstance(v, datetime)
        }

        unique_producer_dates = tl_df["date_producer"].unique()

        if interpolation_type == "nearest":
            interpolation_weights = {
                date: self.find_closest_date(date, sorted_dates)
                for date in unique_producer_dates
            }

        elif interpolation_type == "linear":
            interpolation_weights = {
                date: self.get_weights_for_interpolation_between_nearest_years(
                    date, sorted_dates, interpolation_type
                )
                for date in unique_producer_dates
            }

        else:
            raise ValueError(
                f"Sorry, but {interpolation_type} interpolation is not available yet."
            )

        first_level_background_static = self.node_collections[
            "first_level_background_static"
        ]
        remapped_interpolation_weights = {
            producer_date: {
                self.reversed_database_dates[date]: share
                for date, share in weights.items()
            }
            for producer_date, weights in interpolation_weights.items()
        }

        tl_df["temporal_market_shares"] = [
            remapped_interpolation_weights[producer_date]
            if producer in first_level_background_static
            else None
            for producer, producer_date in zip(tl_df["producer"], tl_df["date_producer"])
        ]

        return tl_df

    def find_closest_date(self, target: datetime, dates: tuple[datetime, ...]) -> dict:
        """
        Find the closest date to the target in the dates list.

        Parameters
        ----------
        target : datetime.datetime
            Target datetime object.
        dates : KeysView[datetime]
            List of datetime.datetime objects.

        Returns
        -------
        dict
            Dictionary with the key as the closest datetime.datetime object from the list and a value of 1.
        """

        # If the list is empty, return None
        if not dates:
            return None

        position = bisect_left(dates, target)
        if position == 0:
            closest = dates[0]
        elif position == len(dates):
            closest = dates[-1]
        else:
            lower_date = dates[position - 1]
            higher_date = dates[position]
            if abs(target - lower_date) <= abs(higher_date - target):
                closest = lower_date
            else:
                closest = higher_date

        return {closest: 1}

    def get_weights_for_interpolation_between_nearest_years(
        self,
        reference_date: datetime,
        dates_list: tuple[datetime, ...],
        interpolation_type: str | None = None,
    ) -> dict:
        """
        Find the nearest dates (lower and higher) for a given date from a list of dates
        and calculate the interpolation weights based on temporal proximity.

        Parameters
        ----------
        reference_date : datetime
            Target date.
        dates_list : KeysView[datetime]
            List of datetime objects representing the temporal representativeness of the available databases.
        interpolation_type : str, optional
            Type of interpolation between the nearest lower and higher dates. For now, only "linear" is available.

        Returns
        -------
        dict
            Dictionary with datetimes of the available closest databases as keys and the weights for interpolation as values.
        """
        interpolation_type = interpolation_type or self.interpolation_type
        position = bisect_left(dates_list, reference_date)

        if position < len(dates_list) and dates_list[position] == reference_date:
            return {dates_list[position]: 1}

        if position == 0:
            if not getattr(self, "_logged_reference_date_below_range", False):
                logger.info(
                    "Reference date {} is lower than all provided dates. Data will be taken from the closest higher year.",
                    reference_date,
                )
                self._logged_reference_date_below_range = True
            return {dates_list[0]: 1}

        if position == len(dates_list):
            if not getattr(self, "_logged_reference_date_above_range", False):
                logger.info(
                    "Reference date {} is higher than all provided dates. Data will be taken from the closest lower year.",
                    reference_date,
                )
                self._logged_reference_date_above_range = True
            return {dates_list[-1]: 1}

        closest_lower = dates_list[position - 1]
        closest_higher = dates_list[position]

        if interpolation_type == "linear":
            weight = int((reference_date - closest_lower).total_seconds()) / int(
                (closest_higher - closest_lower).total_seconds()
            )
        else:
            raise ValueError(
                f"Sorry, but {interpolation_type} interpolation is not available yet."
            )
        return {closest_lower: round(1 - weight, 3), closest_higher: round(weight, 3)}

    def add_interpolation_weights_at_intersection_to_background(
        self, row
    ) -> Union[dict, None]:
        """
        returns the interpolation weights to background databases only for those exchanges,
        where the producing process actually comes from a background database (temporal markets).

        Only these processes are receiving inputs from the background databases.
        All other process in the timeline are not directly linked to the background,
        so the interpolation weight info is not needed and set to None

        Parameters
        ----------
        row : pd.Series
            Row of the timeline DataFrame
        Returns
        -------
        dict
            Dictionary with the name of databases and interpolation weights.
        """

        if row["producer"] in self.node_collections["first_level_background_static"]:
            return {
                self.reversed_database_dates[x]: v
                for x, v in row["temporal_market_shares"].items()
            }
        return None

    def get_consumer_name(self, idx: int) -> str:
        """
        Returns the name of consumer node.
        If consuming node is the functional unit, returns -1.

        Parameters
        ----------
        id : int
            Id of node.

        Returns
        -------
        str
            Name of the node or -1
        """
        try:
            return self.nodes[idx]["name"]
        except KeyError:
            return "-1"  # functional unit
