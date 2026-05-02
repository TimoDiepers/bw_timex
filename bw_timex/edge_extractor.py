import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from numbers import Number
from typing import Callable

import numpy as np
from bw2data.backends.schema import ActivityDataset as AD
from bw2data.backends.schema import ExchangeDataset as ED
from bw_temporalis import TemporalDistribution, TemporalisLCA, loader_registry

datetime_type = np.dtype("datetime64[s]")
timedelta_type = np.dtype("timedelta64[s]")


@dataclass
class Edge:
    """
    Class for storing a temporal edge with source and target.

    Leaf edges link to a source process which is a leaf in
    our graph traversal (either through cutoff or a filter
    function).

    """

    edge_type: str
    distribution: TemporalDistribution
    leaf: bool
    consumer: int
    producer: int
    td_producer: TemporalDistribution
    td_consumer: TemporalDistribution
    abs_td_producer: TemporalDistribution = None
    abs_td_consumer: TemporalDistribution = None
    temporal_evolution: dict = None
    temporal_evolution_reference: str = "producer"


class EdgeExtractor(TemporalisLCA):
    """
    Child class of TemporalisLCA that traverses the supply chain just as the parent class but can create a timeline of edges, in addition timeline of flows or nodes.

    The edge timeline is then used to match the timestamp of edges to that of background
    databases and to replace these edges with edges from these background databases
    using Brightway Datapackages.
    """

    def __init__(self, *args, edge_filter_function: Callable = None, **kwargs) -> None:
        """
        Initialize the EdgeExtractor class and traverses the supply chain using
        functions of the parent class TemporalisLCA.

        Parameters
        ----------
        *args : Variable length argument list
        edge_filter_function : Callable, optional
            A callable that filters edges. If not provided, a function that always
            returns False is used.
        **kwargs : Arbitrary keyword arguments

        Returns
        -------
        None
            stores the output of the TemporalisLCA graph traversal (incl. relation of edges (edge_mapping) and nodes (node_mapping) in the instance of the class.

        """
        super().__init__(*args, **kwargs)  # use __init__ of TemporalisLCA

        if edge_filter_function:
            self.edge_ff = edge_filter_function
        else:
            self.edge_ff = lambda x: False

    def build_edge_timeline(self) -> list:
        """
        Creates a timeline of the edges from the output of the graph traversal.
        Starting from the edges of the functional unit node, it goes through
        each node using a heap, selecting the node with the highest impact first.
        It, then, propagates the TemporalDistributions of the edges from node to
        node through time using convolution-operators. It stops in case the current edge
        is known to have no temporal distribution (=leaf) (e.g. part of background database).

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of Edge instances with timestamps and amounts, and ids of its producing
            and consuming node.

        """

        heap = []
        timeline = []

        for edge in self.edge_mapping[
            self.unique_id
        ]:  # starting at the edges of the functional unit
            node = self.nodes[edge.producer_unique_id]
            heappush(
                heap,
                (
                    1 / node.cumulative_score,
                    self.t0 * edge.amount,
                    self.t0,
                    self.t0,
                    node,
                ),
            )

            timeline.append(
                Edge(
                    edge_type="production",  # FU exchange always type production (?)
                    distribution=self.t0 * edge.amount,
                    leaf=False,
                    consumer=self.unique_id,
                    producer=node.activity_datapackage_id,
                    td_producer=edge.amount,
                    td_consumer=self.t0,
                    abs_td_producer=self.t0,
                )
            )

        while heap:
            _, td, td_parent, abs_td, node = heappop(heap)

            for edge in self.edge_mapping[node.unique_id]:
                row_id = self.nodes[edge.producer_unique_id].activity_datapackage_id
                col_id = node.activity_datapackage_id
                exchange = self.get_technosphere_exchange(
                    input_id=row_id,
                    output_id=col_id,
                )

                edge_type = exchange.data[
                    "type"
                ]  # can be technosphere, substitution, production or other string

                # Extract temporal evolution data from exchange
                temporal_evolution = None
                exc_data = exchange.data
                has_amounts = exc_data.get("temporal_evolution_amounts") is not None
                has_factors = exc_data.get("temporal_evolution_factors") is not None

                if has_amounts and has_factors:
                    raise ValueError(
                        f"Exchange from {exc_data.get('input')} to "
                        f"{exc_data.get('output', node.key)} has both "
                        f"'temporal_evolution_amounts' and 'temporal_evolution_factors'. "
                        f"These are mutually exclusive — use one or the other."
                    )

                if has_amounts:
                    base_amount = exc_data["amount"]
                    if base_amount != 0:
                        temporal_evolution = {
                            k: v / abs(base_amount)
                            for k, v in exc_data["temporal_evolution_amounts"].items()
                        }
                elif has_factors:
                    temporal_evolution = exc_data["temporal_evolution_factors"]
                temporal_evolution_reference = exc_data.get(
                    "temporal_evolution_reference", "producer"
                )

                td_producer = (  # td_producer is the TemporalDistribution of the edge
                    self._exchange_value(
                        exchange=exchange,
                        row_id=row_id,
                        col_id=col_id,
                        matrix_label="technosphere_matrix",
                    )
                    / abs(node.reference_product_production_amount)
                )
                producer = self.nodes[edge.producer_unique_id]
                leaf = self.edge_ff(row_id)

                # If an edge does not have a TD, give it a td with timedelta=0 and the amount= 'edge value'
                if isinstance(td_producer, Number):
                    td_producer = TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([td_producer]),
                    )

                distribution = (
                    td * td_producer
                ).simplify()  # convolution-multiplication of TemporalDistribution of consuming node (td) and consumed edge (edge) gives TD of producing node

                timeline.append(
                    Edge(
                        edge_type=edge_type,
                        distribution=distribution,
                        leaf=leaf,
                        consumer=node.activity_datapackage_id,
                        producer=producer.activity_datapackage_id,
                        td_producer=td_producer,
                        td_consumer=td_parent,
                        abs_td_producer=self.join_datetime_and_timedelta_distributions(
                            td_producer, abs_td
                        ),
                        abs_td_consumer=abs_td,
                        temporal_evolution=temporal_evolution,
                        temporal_evolution_reference=temporal_evolution_reference,
                    )
                )
                if not leaf:
                    heappush(
                        heap,
                        (
                            1 / node.cumulative_score,
                            distribution,
                            td_producer,
                            self.join_datetime_and_timedelta_distributions(
                                td_producer, abs_td
                            ),
                            producer,
                        ),
                    )
        return timeline

    def join_datetime_and_timedelta_distributions(
        self, td_producer: TemporalDistribution, td_consumer: TemporalDistribution
    ) -> TemporalDistribution:
        """
        Joins a datetime TemporalDistribution (td_producer) with a timedelta
        TemporalDistribution (td_consumer) to create a new TemporalDistribution.

        If the producer does not have a TemporalDistribution, the consumer's
        TemporalDistribution is returned to continue the timeline.
        If both the producer and consumer have TemporalDistributions, they are joined together.

        Parameters
        ----------
        td_producer : TemporalDistribution
            TemporalDistribution of the producer. Expected to be a datetime TemporalDistribution.
        td_consumer : TemporalDistribution
            TemporalDistribution of the consumer. Expected to be a timedelta TemporalDistribution.

        Returns
        -------
        TemporalDistribution
            A new TemporalDistribution that is the result of joining the producer
            and consumer TemporalDistributions.

        Raises
        ------
        ValueError
            If the dtype of `td_consumer.date` is not `datetime64[s]` or the dtype
            of `td_producer.date` is not `timedelta64[s]`.

        """
        # if an edge does not have a TD, then return the consumer_td so that the timeline continues
        if isinstance(td_consumer, TemporalDistribution) and isinstance(
            td_producer, Number
        ):
            return td_consumer

        # Else, if both consumer and producer have a td (absolute and relative, respectively) join to TDs
        if isinstance(td_producer, TemporalDistribution) and isinstance(
            td_consumer, TemporalDistribution
        ):
            if not (td_consumer.date.dtype == datetime_type):
                raise ValueError(
                    f"`td_consumer.date` must have dtype `datetime64[s]`, but got `{td_consumer.date.dtype}`"
                )
            if not (td_producer.date.dtype == timedelta_type):
                raise ValueError(
                    f"`td_producer.date` must have dtype `timedelta64[s]`, but got `{td_producer.date.dtype}`"
                )
            date = (
                td_consumer.date.reshape((-1, 1)) + td_producer.date.reshape((1, -1))
            ).ravel()
            amount = np.array(len(td_consumer) * [td_producer.amount]).ravel()
            return TemporalDistribution(date, amount)
        else:
            raise ValueError(
                f"Can't join TemporalDistribution and something else: Trying with {type(td_consumer.date)} and {type(td_producer.date)}"
            )


class EdgeExtractorBFS:
    """
    Breadth-First-Search (BFS) graph traversal for extracting temporal edges from
    the supply chain.

    Unlike EdgeExtractor (which inherits from TemporalisLCA and uses priority-first
    traversal with per-subgraph LCA calculations), this class works directly with
    the technosphere matrix from a bw2calc LCA object and traverses using BFS.
    This avoids the overhead of computing individual subgraph LCAs for priority
    ordering.

    Returns the same list[Edge] format as EdgeExtractor, so all downstream code
    (TimelineBuilder, MatrixModifier, etc.) works unchanged.
    """

    def __init__(
        self,
        lca_object,
        starting_datetime: datetime | str = "now",
        edge_filter_function: Callable = None,
        cutoff: float = 1e-9,
        static_activity_indices: set[int] | None = None,
    ) -> None:
        self.lca_object = lca_object
        self.edge_ff = edge_filter_function if edge_filter_function else lambda x: False
        self.cutoff = cutoff
        self.static_activity_indices = static_activity_indices or set()

        if isinstance(starting_datetime, str):
            if starting_datetime == "now":
                starting_datetime = datetime.now()
            else:
                starting_datetime = datetime.fromisoformat(starting_datetime)

        self.t0 = TemporalDistribution(
            np.array([np.datetime64(starting_datetime)]),
            np.array([1]),
        )

        self.tech_matrix_csc = self.lca_object.technosphere_matrix.tocsc()
        self._ad_cache = {}

    def _get_activity_dataset(self, activity_id: int) -> AD:
        if activity_id not in self._ad_cache:
            self._ad_cache[activity_id] = AD.get(AD.id == activity_id)
        return self._ad_cache[activity_id]

    def _get_exchange(self, input_id: int, output_id: int):
        """Look up exchange between two activities. Returns ExchangeDataset or None."""
        inp = self._get_activity_dataset(input_id)
        outp = self._get_activity_dataset(output_id)
        exchanges = list(
            ED.select().where(
                ED.input_code == inp.code,
                ED.input_database == inp.database,
                ED.output_code == outp.code,
                ED.output_database == outp.database,
            )
        )
        if len(exchanges) == 1:
            return exchanges[0]
        elif len(exchanges) > 1:
            raise ValueError(
                f"Found {len(exchanges)} exchanges between {input_id} and {output_id}"
            )
        return None

    def _get_exchange_td_and_type(self, input_id: int, output_id: int):
        """
        Get temporal distribution and edge type for an exchange.

        Returns (td_or_amount, edge_type) where td_or_amount is either a
        TemporalDistribution or a float (the signed matrix value).
        """
        exchange = self._get_exchange(input_id, output_id)

        row_idx = self.lca_object.dicts.product[input_id]
        col_idx = self.lca_object.dicts.activity[output_id]
        matrix_value = self.tech_matrix_csc[row_idx, col_idx]

        if exchange is None:
            sign = 1 if input_id == output_id else -1
            return sign * matrix_value, "technosphere"

        edge_type = exchange.data["type"]
        sign = -1 if edge_type in ("generic consumption", "technosphere") else 1
        amount = sign * matrix_value

        td = exchange.data.get("temporal_distribution")
        if td is not None:
            if isinstance(td, str) and "__loader__" in td:
                data = json.loads(td)
                td = loader_registry[data["__loader__"]](data)
            if isinstance(td, TemporalDistribution):
                return td * amount, edge_type

        return amount, edge_type

    def _get_production_amount(self, activity_id: int) -> float:
        """Get the reference product production amount (diagonal of tech matrix)."""
        product_idx = self.lca_object.dicts.product[activity_id]
        col_idx = self.lca_object.dicts.activity[activity_id]
        return self.tech_matrix_csc[product_idx, col_idx]

    def _get_technosphere_inputs(self, activity_id: int) -> list[int]:
        """Get all technosphere input activity IDs for a given activity."""
        col_idx = self.lca_object.dicts.activity[activity_id]
        col = self.tech_matrix_csc[:, col_idx]
        product_idx = self.lca_object.dicts.product.get(activity_id)

        inputs = []
        for row_idx in col.nonzero()[0]:
            if row_idx == product_idx:
                continue
            input_id = self.lca_object.dicts.product.reversed[row_idx]
            if input_id in self.static_activity_indices:
                continue
            inputs.append(input_id)
        return inputs

    def build_edge_timeline(self) -> list:
        """
        Breadth-First-Search (BFS) traversal of the supply chain, extracting
        temporal edges.

        Returns a list of Edge instances compatible with the existing
        EdgeExtractor output format.
        """
        timeline = []
        queue = deque()

        demand_array = self.lca_object.demand_array
        fu_activity_ids = [
            self.lca_object.dicts.activity.reversed[idx]
            for idx in demand_array.nonzero()[0]
        ]

        total_demand = float(np.abs(demand_array).sum())

        for fu_id in fu_activity_ids:
            fu_amount = demand_array[self.lca_object.dicts.activity[fu_id]]
            td = self.t0 * fu_amount

            timeline.append(
                Edge(
                    edge_type="production",
                    distribution=td,
                    leaf=False,
                    consumer=-1,
                    producer=fu_id,
                    td_producer=fu_amount,
                    td_consumer=self.t0,
                    abs_td_producer=self.t0,
                )
            )

            queue.append((fu_id, td, self.t0, self.t0, abs(fu_amount)))

        while queue:
            node_id, td, td_parent, abs_td, supply = queue.popleft()

            production_amount = self._get_production_amount(node_id)
            input_ids = self._get_technosphere_inputs(node_id)

            for input_id in input_ids:
                leaf = self.edge_ff(input_id)

                td_producer_raw, edge_type = self._get_exchange_td_and_type(
                    input_id, node_id
                )

                td_producer = td_producer_raw / abs(production_amount)

                if isinstance(td_producer, Number):
                    td_producer = TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([td_producer]),
                    )

                distribution = (td * td_producer).simplify()

                abs_td_producer = _join_datetime_and_timedelta_distributions(
                    td_producer, abs_td
                )

                timeline.append(
                    Edge(
                        edge_type=edge_type,
                        distribution=distribution,
                        leaf=leaf,
                        consumer=node_id,
                        producer=input_id,
                        td_producer=td_producer,
                        td_consumer=td_parent,
                        abs_td_producer=abs_td_producer,
                        abs_td_consumer=abs_td,
                    )
                )

                if isinstance(td_producer_raw, TemporalDistribution):
                    edge_supply = abs(td_producer_raw.amount.sum())
                else:
                    edge_supply = abs(td_producer_raw)
                new_supply = supply * edge_supply / abs(production_amount)

                if not leaf and new_supply >= self.cutoff * total_demand:
                    queue.append(
                        (
                            input_id,
                            distribution,
                            td_producer,
                            abs_td_producer,
                            new_supply,
                        )
                    )

        return timeline


def _join_datetime_and_timedelta_distributions(
    td_producer: TemporalDistribution,
    td_consumer: TemporalDistribution,
) -> TemporalDistribution:
    """
    Join a timedelta TemporalDistribution (td_producer) with a datetime
    TemporalDistribution (td_consumer).

    If the producer does not have a TemporalDistribution, the consumer's
    TemporalDistribution is returned. If both have TDs, they are joined
    via broadcasting.
    """
    if isinstance(td_consumer, TemporalDistribution) and isinstance(
        td_producer, Number
    ):
        return td_consumer

    if isinstance(td_producer, TemporalDistribution) and isinstance(
        td_consumer, TemporalDistribution
    ):
        if not (td_consumer.date.dtype == datetime_type):
            raise ValueError(
                f"`td_consumer.date` must have dtype `datetime64[s]`, "
                f"but got `{td_consumer.date.dtype}`"
            )
        if not (td_producer.date.dtype == timedelta_type):
            raise ValueError(
                f"`td_producer.date` must have dtype `timedelta64[s]`, "
                f"but got `{td_producer.date.dtype}`"
            )
        date = (
            td_consumer.date.reshape((-1, 1)) + td_producer.date.reshape((1, -1))
        ).ravel()
        amount = np.array(len(td_consumer) * [td_producer.amount]).ravel()
        return TemporalDistribution(date, amount)
    else:
        raise ValueError(
            f"Can't join TemporalDistribution and something else: "
            f"Trying with {type(td_consumer)} and {type(td_producer)}"
        )
