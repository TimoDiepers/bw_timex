import uuid
from typing import Optional

import bw2data as bd
import bw_processing as bwp
import numpy as np
import pandas as pd

from .helper_classes import InterDatabaseMapping
from .utils import get_temporal_evolution_factor


class MatrixModifier:
    """
    Class for adding and re-linking time-explicit processes in the LCA matrices.

    This is done by creating datapackages that add or change matrix entries in the technosphere and
    biosphere matrices, based on a process timeline (from TimelineBuilder.build_timeline()).
    "Temporal markets" are created for processes that are linked to background databases, temporally
    distributing the amounts to time-explicit background databases.
    """

    def __init__(
        self,
        timeline: pd.DataFrame,
        database_dates_static: dict,
        demand_timing: dict,
        nodes: dict,
        interdatabase_activity_mapping: InterDatabaseMapping,
        name: Optional[str] = None,
    ) -> None:
        """
        Initializes the MatrixModifier object and creates empty sets to collect
        the ids of temporalized processes and temporal markets.

        Parameters
        ----------
        timeline : pd.DataFrame
            A DataFrame of the timeline of exchanges
        database_dates_static : dict
            A dictionary mapping the static background databases to dates.
        demand_timing : dict
            A dictionary mapping the demand to its timing.
        name : str, optional
            An optional name for the MatrixModifier instance. Default is None.

        """

        self.timeline = timeline
        self.database_dates_static = database_dates_static
        self.demand_timing = demand_timing
        self.nodes = nodes
        self.interdatabase_activity_mapping = interdatabase_activity_mapping
        self.name = name
        self.temporalized_process_ids = set()
        self.temporal_market_ids = set()

    def create_datapackage(self) -> None:
        """
        Creates a list of datapackages for the technosphere and biosphere matrices,
        by calling the respective functions.

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of the technosphere and biosphere datapackages.
        """
        technosphere_datapackage = self.create_technosphere_datapackage()
        biosphere_datapackage = self.create_biosphere_datapackage()

        return [technosphere_datapackage, biosphere_datapackage]

    def create_technosphere_datapackage(self) -> bwp.Datapackage:
        """
        Creates the modifications to the technosphere matrix in form of a datapackage.
        Datapackages add or overwrite data points in the LCA matrices before LCA calculations.
        The technosphere datapackage adds the temporalized processes from the timeline
        to the technosphere matrix.

        The heavy lifting of this method happens in the method `add_row_to_technosphere_datapackage()`.
        Here, each node with a temporal distribution is "exploded", which means each occurrence
        of this node (e.g. steel production on 2020-01-01 and steel production on 2015-01-01)
        becomes a separate, time-explicit new node, by adding the new elements to the technosphere matrix.
        For processes at the intersection with background databases, the timing of the exchanges
        determines which background database to link to in so called "Temporal Markets".

        Parameters
        ----------
        None

        Returns
        -------
        bwp.Datapackage
            A datapackage containing the modifications for the technosphere matrix.
        """
        datapackage_technosphere = bwp.create_datapackage(
            sum_inter_duplicates=False
        )  # 'sum_inter_duplicates=False': If the same market is used by multiple foreground processes, the market gets created again, inputs should not be summed.

        new_nodes = set()

        for row in self.timeline.iloc[::-1].itertuples():
            self.add_row_to_technosphere_datapackage(
                row,
                datapackage_technosphere,
                new_nodes,
            )

        # Adding the production exchanges for new nodes
        for node_id, production_amount in new_nodes:
            datapackage_technosphere.add_persistent_vector(
                matrix="technosphere_matrix",
                name=uuid.uuid4().hex,
                data_array=np.array([production_amount], dtype=float),
                indices_array=np.array([(node_id, node_id)], dtype=bwp.INDICES_DTYPE),
            )

        return datapackage_technosphere

    def create_biosphere_datapackage(self) -> bwp.Datapackage:
        """
        Creates the modifications to the biosphere matrix in form of a datapackage.
        Datapackages add or overwrite data points in the LCA matrices before LCA calculations.
        It adds the biosphere flows to the exploded technosphere processes.

        This function iterates over each unique producer, and for each biosphere exchange
        of the original activity, it creates a new biosphere exchange for the new "temporalized" node.

        Temporal markets have no biosphere exchanges, as they only divide the amount of
        a technosphere exchange between the different databases.

        Parameters
        ----------
        None

        Returns
        -------
        bwp.Datapackage
            A datapackage containing the modifications for the biosphere matrix.
        """
        unique_producers = (
            self.timeline.groupby(["producer", "time_mapped_producer"])
            .count()
            .index.values
        )  # array of unique ((original) producer_id, (new) time_mapped_producer_id) tuples

        datapackage_biosphere = bwp.create_datapackage(sum_inter_duplicates=False)

        for producer in unique_producers:
            original_producer_node = self.nodes[producer[0]]
            if (
                original_producer_node["database"]
                not in self.database_dates_static.keys()  # skip temporal markets
            ):
                new_producer_id = producer[1]
                indices = (
                    []
                )  # list of (biosphere, technosphere) indices for the biosphere flow exchanges
                amounts = []  # list of amounts corresponding to the bioflows
                for exc in original_producer_node.biosphere():
                    indices.append(
                        (exc.input.id, new_producer_id)
                    )  # directly build a list of tuples to pass into the datapackage, the new_producer_id is the new column index
                    amounts.append(exc.amount)

                datapackage_biosphere.add_persistent_vector(
                    matrix="biosphere_matrix",
                    name=uuid.uuid4().hex,
                    data_array=np.array(amounts, dtype=float),
                    indices_array=np.array(
                        indices,
                        dtype=bwp.INDICES_DTYPE,
                    ),
                    flip_array=np.array([False], dtype=bool),
                )
        return datapackage_biosphere

    def add_row_to_technosphere_datapackage(
        self,
        row: pd.core.frame,
        datapackage: bwp.Datapackage,
        new_nodes: set,
    ) -> None:
        """
        This adds the modifications to the technosphere matrix for each time-dependent exchange
        as datapackage elements to a given `bwp.Datapackage`.


        Modifications include:

        1) Exploded processes: new matrix elements for time-explicit consumer and time-explicit
        producer, representing the temporal edge between them.

        2) Temporal markets: new matrix entries for "temporal markets" and links to the producers
        in temporally matching background databases. Processes in the background databases are
        matched on name, reference product and location.

        3) Diagonal entries: ones on the diagonal for new nodes.

        This function also collects the ids of new nodes, temporalized nodes and temporal markets.

        Parameters
        ----------
        row : pd.core.frame
            A row of the timeline DataFrame representing an temporalized edge
        datapackage : bwp.Datapackage
            Append to this datapackage, if available. Otherwise create a new datapackage.
        new_nodes : set
            Set of tuples (node_id, production_amount) to which new node ids are added.

        Returns
        -------
        None
            Adds elements for this edge to the bwp.Datapackage and stores the ids of new nodes, temporalized nodes and temporal markets.
        """

        if row.consumer == -1:  # functional unit
            new_producer_id = row.time_mapped_producer
            fu_production_amount = self.nodes[row.producer].rp_exchange().amount
            new_nodes.add((new_producer_id, fu_production_amount))
            self.temporalized_process_ids.add(
                new_producer_id
            )  # comes from foreground, so it is a temporalized process
            return

        new_consumer_id = row.time_mapped_consumer
        new_producer_id = row.time_mapped_producer

        previous_producer_id = row.producer
        previous_producer_node = self.nodes[previous_producer_id]

        production_exchange_amount = self.nodes[row.consumer].rp_exchange().amount
        scaled_amount = row.amount * abs(
            production_exchange_amount
        )  # abs value used for scaling to preserve the sign of the exchange

        # Apply temporal evolution scaling if present
        if hasattr(row, "temporal_evolution") and row.temporal_evolution is not None:
            reference = getattr(row, "temporal_evolution_reference", "producer")
            reference_date = (
                row.date_consumer if reference == "consumer" else row.date_producer
            )
            factor = get_temporal_evolution_factor(
                row.temporal_evolution, reference_date
            )
            scaled_amount *= factor

        # Add entry between exploded consumer and exploded producer (not in background database)
        datapackage.add_persistent_vector(
            matrix="technosphere_matrix",
            name=uuid.uuid4().hex,
            data_array=np.array([scaled_amount], dtype=float),
            indices_array=np.array(
                [(new_producer_id, new_consumer_id)],
                dtype=bwp.INDICES_DTYPE,
            ),
            flip_array=np.array([True], dtype=bool),
        )

        # Check if previous producer comes from background database -> temporal market
        if previous_producer_node["database"] in self.database_dates_static.keys():
            # Create new edges based on temporal_market_shares from the row
            if not row.temporal_market_shares:
                raise ValueError(
                    f"Row {row} has no temporal market shares, but the previous producer {previous_producer_node['name']} is from a background database."
                )
            for database, db_share in row.temporal_market_shares.items():
                # Get the producer activity in the corresponding background database
                try:
                    producer_id_in_background_db = (
                        self.interdatabase_activity_mapping.find_match(
                            previous_producer_id, database
                        )
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Producer with ID {previous_producer_id} not found in database {database}."
                    ) from exc
                except Exception as e:
                    raise RuntimeError(
                        f"Error finding producer {previous_producer_id} in database {database}: {e}"
                    ) from e

                # Add entry between exploded producer and producer in background database
                # -->("Temporal Market")
                datapackage.add_persistent_vector(
                    matrix="technosphere_matrix",
                    name=uuid.uuid4().hex,
                    data_array=np.array(
                        [db_share], dtype=float
                    ),  # temporal markets produce 1, so shares divide amount between dbs
                    indices_array=np.array(
                        [(producer_id_in_background_db, new_producer_id)],
                        dtype=bwp.INDICES_DTYPE,
                    ),
                    flip_array=np.array([True], dtype=bool),
                )
                self.temporal_market_ids.add(new_producer_id)
                producer_production_amount = (
                    1  # Shares sum up to 1, so production amount is 1
                )

        else:  # comes from foreground, so it is a temporalized process
            self.temporalized_process_ids.add(new_producer_id)

            # Get the production amount of the previous producer if it's not a temporal market - needed for diagonal matrix entry
            if isinstance(
                previous_producer_node, bd.backends.iotable.proxies.IOTableActivity
            ):
                if len(previous_producer_node.production()) == 1:
                    producer_production_amount = list(
                        previous_producer_node.production()
                    )[0].amount
                else:
                    raise ValueError(
                        "The producer activity is of type IOTableActivity, but has more than one production exchange. This is currently not supported."
                    )
            elif isinstance(previous_producer_node, bd.backends.proxies.Activity):
                producer_production_amount = (
                    self.nodes[row.producer].rp_exchange().amount
                )
            else:
                raise ValueError(
                    f"Can't determine the production amount of the producer activity {previous_producer_node['name']} , as it's of an unknown type."
                )

        # Add newly created producing process to new_nodes
        new_nodes.add((new_producer_id, producer_production_amount))
