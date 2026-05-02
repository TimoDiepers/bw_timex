import bw2data as bd
import numpy as np
import pandas as pd
from bw2calc import LCA
from bw_temporalis import TemporalDistribution
from scipy import sparse as sp

from .helper_classes import SetList
from .utils import convert_date_string_to_datetime, get_temporal_evolution_factor


class DynamicBiosphereBuilder:
    """
    Class for building a dynamic biosphere matrix with dimensions (biosphere flow at a specific point in time) x (processes)
    """

    def __init__(
        self,
        lca_obj: LCA,
        activity_time_mapping: dict,
        biosphere_time_mapping: dict,
        demand_timing: dict,
        node_collections: dict,
        temporal_grouping: str,
        database_dates: dict,
        database_dates_static: dict,
        timeline: pd.DataFrame,
        interdatabase_activity_mapping: SetList,
        expand_technosphere: bool = True,
    ) -> None:
        """
        Initializes the DynamicBiosphereBuilder object.

        Parameters
        ----------
        lca_obj : LCA object
            instance of the bw2calc LCA class, e.g. TimexLCA.lca
        activity_time_mapping : dict
            A dictionary mapping activity to their respective timing in the format
            (('database', 'code'), datetime_as_integer): time_mapping_id)
        biosphere_time_mapping : dict
            A dictionary mapping biosphere flows to their respective timing in the format
            (('database', 'code'), datetime_as_integer): time_mapping_id), empty at this point.
        demand_timing : dict
            A dictionary mapping of the demand to demand time
        node_collections : dict
            A dictionary containing lists of node ids for different node subsets
        temporal_grouping : str
            A string indicating the temporal grouping of the processes, e.g. 'year', 'month',
            'day', 'hour'
        database_dates : dict
            A dictionary mapping database names to their respective date
        database_dates_static : dict
            A dictionary mapping database names to their respective date, but only containing
            static databases, which are the background databases.
        timeline: pd.DataFrame
            The edge timeline, created from TimexLCA.build_timeline()
        interdatabase_activity_mapping : SetList
            A list of sets, where each set contains the activity ids of the same activity in
            different databases
        expand_technosphere : bool, optional
            A boolean indicating if the dynamic biosphere matrix is built via expanded matrices or directly from the timeline.
            Default is True.

        Returns
        -------
        None

        """

        self._time_res_mapping = {
            "year": "datetime64[Y]",
            "month": "datetime64[M]",
            "day": "datetime64[D]",
            "hour": "datetime64[h]",
        }

        self.lca_obj = lca_obj

        if expand_technosphere:
            self.technosphere_matrix = (
                lca_obj.technosphere_matrix.tocsc()
            )  # convert to csc as this is only used for column slicing
            self.dynamic_supply_array = lca_obj.supply_array
            self.activity_dict = lca_obj.dicts.activity
        else:
            self.dynamic_supply_array = timeline.amount.values.astype(
                float
            )  # get the supply vector directly from the timeline

        self.activity_time_mapping = activity_time_mapping
        self.biosphere_time_mapping = biosphere_time_mapping
        self.demand_timing = demand_timing
        self.node_collections = node_collections
        self.time_res = self._time_res_mapping[temporal_grouping]
        self.temporal_grouping = temporal_grouping
        self.database_dates = database_dates
        self.database_dates_static = database_dates_static
        self.timeline = timeline
        self.interdatabase_activity_mapping = interdatabase_activity_mapping
        self.rows = []
        self.cols = []
        self.values = []
        self.unique_rows_cols = set()  # To keep track of (row, col) pairs
        self.temporal_market_cols = []  # To keep track of temporal market columns

    def build_dynamic_biosphere_matrix(
        self,
        expand_technosphere: bool = True,
    ):
        """
        This function creates a separate biosphere matrix, with the dimensions
        (bio_flows at a specific time step) x (processes).

        Every temporally resolved biosphere flow has its own row in the matrix, making it highly
        sparse. The timing of the emitting process and potential additional temporal information of
        the biosphere flow (e.g. delay of emission compared to the timing of the process) are considered.

        Absolute Temporal Distributions for biosphere exchanges are dealt with as a look up
        function: If an activity happens at timestamp X and the biosphere exchange has an
        absolute temporal distribution (ATD), it looks up the amount from the ATD corresponding
        to timestamp X. E.g.: X = 2024, TD=(data=[2020,2021,2022,2023,2024,.....,2120],
        amount=[3,4,4,5,6,......,3]), it will look up the value 6 corresponding 2024. If timestamp X
        does not exist, it finds the nearest timestamp available (if two timestamps are equally close,
        it will take the first in order of appearance (see numpy.argmin() for this behavior).

        Parameters
        ----------
        expand_technosphere : bool, optional
            A boolean indicating if the dynamic biosphere matrix is built via expanded matrices
            or directly from the timeline. Default is via expanded matrices.

        Returns
        -------
        dynamic_biosphere_matrix : scipy.sparse.csr_matrix
            A sparse matrix with the dimensions (bio_flows at a specific time step) x (processes).
        temporal_market_lcis : dict
            A dictionary containing the disaggregated LCI's of the temporal markets,
            with the time-mapped-activity id as key.
        """

        lci_dict = {}
        temporal_market_lcis = {}

        for row in self.timeline.itertuples():
            idx = row.time_mapped_producer

            if expand_technosphere:
                process_col_index = self.activity_dict[
                    idx
                ]  # get the matrix column index
            else:  # from timeline
                process_col_index = row.Index  # start a new matrix

            (
                (original_db, original_code),
                time,
            ) = self.activity_time_mapping.reversed[idx]

            if idx in self.node_collections["temporalized_processes"]:

                time_in_datetime = convert_date_string_to_datetime(
                    self.temporal_grouping, str(time)
                )  # now time is a datetime

                td_producer = TemporalDistribution(
                    date=np.array([time_in_datetime], dtype=self.time_res),
                    amount=np.array([1]),
                ).date
                date = td_producer[0]

                # Get temporal evolution factor for this timestamp
                temporal_evolution_factor = 1.0
                if hasattr(row, "temporal_evolution") and row.temporal_evolution is not None:
                    reference = getattr(row, "temporal_evolution_reference", "producer")
                    reference_time = (
                        row.date_consumer if reference == "consumer" else time_in_datetime
                    )
                    temporal_evolution_factor = get_temporal_evolution_factor(
                        row.temporal_evolution, reference_time
                    )

                if original_db == "temporalized":
                    act = bd.get_node(code=original_code)
                else:
                    act = bd.get_node(database=original_db, code=original_code)

                for exc in act.biosphere():
                    if exc.get("temporal_distribution"):
                        td_dates = exc["temporal_distribution"].date
                        td_values = exc["temporal_distribution"].amount
                        # If the biosphere flows have an absolute TD, this means we have to look up
                        # the biosphere flow for the activity time (td_producer)
                        if isinstance(td_dates[0], np.datetime64):
                            dates = td_producer  # datetime array, same time as producer
                            values = [
                                exc["amount"]
                                * temporal_evolution_factor
                                * td_values[
                                    np.argmin(
                                        np.abs(
                                            td_dates.astype(self.time_res)
                                            - td_producer.astype(self.time_res)
                                        )
                                    )
                                ]
                            ]  # look up the value correponding to the absolute producer time
                        else:
                            # we can add a datetime of len(1) to a timedelta of len(N) easily
                            dates = td_producer + td_dates
                            values = exc["amount"] * temporal_evolution_factor * td_values

                    else:  # exchange has no TD
                        dates = td_producer  # datetime array, same time as producer
                        values = [exc["amount"] * temporal_evolution_factor]

                    # Add entries to dynamic bio matrix
                    for date, amount in zip(dates, values):

                        # first create a row index for the tuple (bioflow_id, date)
                        time_mapped_matrix_idx = self.biosphere_time_mapping.add(
                            (exc.input.id, date)
                        )

                        # populate lists with which sparse matrix is constructed
                        self.add_matrix_entry_for_biosphere_flows(
                            row=time_mapped_matrix_idx,
                            col=process_col_index,
                            amount=amount,
                        )

            elif idx in self.node_collections["temporal_markets"]:
                self.temporal_market_cols.append(process_col_index)
                (
                    (original_db, original_code),
                    time,
                ) = self.activity_time_mapping.reversed[idx]

                if expand_technosphere:
                    demand = self.demand_from_technosphere(idx, process_col_index)
                else:
                    demand = self.demand_from_timeline(row)

                if demand:
                    for act, amount in demand.items():
                        # check if lci already calculated for this activity
                        if not act in lci_dict.keys():
                            self.lca_obj.redo_lci({act: 1})
                            #  biosphere flows by activity of background supply chain emissions.
                            # Rows are bioflows. Columns are activities.
                            # save for reuse in dict
                            lci_dict[act] = self.lca_obj.inventory
                        # add lci of both background activities of the temporal market and save total lci
                        if idx not in temporal_market_lcis.keys():
                            temporal_market_lcis[idx] = lci_dict[act] * amount
                        else:
                            temporal_market_lcis[idx] += lci_dict[act] * amount

                    aggregated_inventory = temporal_market_lcis[idx].sum(axis=1)

                    # multiply LCI with supply of temporal market
                    temporal_market_lcis[idx] *= self.dynamic_supply_array[
                        process_col_index
                    ]

                    for row_idx, amount in enumerate(aggregated_inventory.A1):
                        bioflow = self.lca_obj.dicts.biosphere.reversed[row_idx]
                        ((_, _), time) = self.activity_time_mapping.reversed[idx]

                        time_in_datetime = convert_date_string_to_datetime(
                            self.temporal_grouping, str(time)
                        )  # now time is a datetime

                        td_producer = TemporalDistribution(
                            date=np.array([str(time_in_datetime)], dtype=self.time_res),
                            amount=np.array([1]),
                        ).date
                        date = td_producer[0]

                        time_mapped_matrix_idx = self.biosphere_time_mapping.add(
                            (bioflow, date)
                        )

                        self.add_matrix_entry_for_biosphere_flows(
                            row=time_mapped_matrix_idx,
                            col=process_col_index,
                            amount=amount,
                        )

        # now build the dynamic biosphere matrix
        if expand_technosphere:
            ncols = len(self.activity_time_mapping)
        else:
            ncols = len(self.timeline)

        shape = (max(self.rows) + 1, ncols)
        dynamic_biosphere_matrix = sp.coo_matrix(
            (self.values, (self.rows, self.cols)), shape
        )
        dynamic_biosphere_matrix = dynamic_biosphere_matrix.tocsr()

        return dynamic_biosphere_matrix, temporal_market_lcis

    def demand_from_timeline(self, row):
        """
        Returns a demand dict directly from the timeline row
        and its temporal_market_shares.

        Parameters:
        -----------
        row: pd.Series
            A row of the timeline DataFrame

        Returns
        -------
        demand: dict
            A demand-dictionary with as keys the ids of the time-mapped activities
            and as values the share.


        """
        demand = {}
        for db, amount in row.temporal_market_shares.items():
            timed_act_id = self.interdatabase_activity_mapping.find_match(
                row.producer, db
            )
            demand[timed_act_id] = amount
        return demand

    def demand_from_technosphere(self, idx, process_col_index):
        """
        Returns a demand dict of background processes based on the technosphere column.
        Foreground exchanges are skipped as these are added separately.

        Parameters:
        -----------
        idx: int
            The time-mapped-activity id of the producer
        process_col_index: int
            The technosphere matrix id of the producer

        Returns
        -------
        demand: dict
            A demand-dictionary with as keys the brightway ids of the consumed background
            activities and as values their consumed amount.
        """
        col = self.technosphere_matrix[:, process_col_index]  # Sparse column
        activity_row = self.activity_dict[idx]  # Producer's row index
        foreground_nodes = self.node_collections["foreground"]

        demand = {
            self.activity_dict.reversed[row_idx]: -amount
            for row_idx, amount in zip(col.nonzero()[0], col.data)
            if row_idx != activity_row  # Skip production exchange
            and self.activity_dict.reversed[row_idx]
            not in foreground_nodes  # Only background
        }

        return demand

    def add_matrix_entry_for_biosphere_flows(self, row, col, amount):
        """
        Adds an entry to a list of row, col and values, which are then used to construct the
        dynamic biosphere matrix. Only unique entries are added, i.e. if the same row and
        col index already exists, the value is not added again.

        Parameters
        ----------
        row : int
            A row index of a new element to the dynamic biosphere matrix
        col: int
            A column index of a new element to the dynamic biosphere matrix
        amount: float
            The amount of the new element to the dynamic biosphere matrix

        Returns
        -------
        None
            the lists of row, col and values are updated

        """

        if (row, col) not in self.unique_rows_cols:
            self.rows.append(row)
            self.cols.append(col)
            self.values.append(amount)

            self.unique_rows_cols.add((row, col))
