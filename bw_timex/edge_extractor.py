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

from .utils import get_reference_product_production_amount

if not hasattr(np, "in1d"):
    np.in1d = np.isin

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

    def __init__(
        self,
        *args,
        edge_filter_function: Callable = None,
        demand_temporal_distributions: dict | None = None,
        **kwargs,
    ) -> None:
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

        self.demand_temporal_distributions = demand_temporal_distributions or {}

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
            row_id = self.lca_object.dicts.product.reversed[edge.product_index]
            col_id = node.activity_datapackage_id

            # A `demand` value of a `TemporalDistribution` replaces the implicit
            # single-point `t0` for this FU: the demand profile itself becomes the
            # absolute timing of consumption. Scale falls out of TD amounts; we
            # treat the FU amount as 1 here because the TD already carries it.
            demand_td = self.demand_temporal_distributions.get(row_id)
            if demand_td is not None:
                t0_eff = demand_td
                edge_amount_eff = 1.0
            else:
                t0_eff = self.t0
                edge_amount_eff = edge.amount

            td_producer = edge_amount_eff
            initial_distribution = t0_eff * edge_amount_eff
            abs_td_producer = t0_eff
            abs_td_consumer = None if demand_td is None else t0_eff

            exchange = self.get_technosphere_exchange(input_id=row_id, output_id=col_id)

            # In the explicit process/product paradigm, the demanded product is produced
            # by an off-diagonal production edge. A TD on that edge distributes the
            # process invocation itself before the process inputs are traversed.
            if (
                row_id != col_id
                and hasattr(exchange, "data")
                and exchange.data.get("type") == "production"
            ):
                production_amount = abs(
                    get_reference_product_production_amount(
                        col_id, reference_product=row_id, lca=self.lca_object
                    )
                )
                td_producer = (
                    self._exchange_value(
                        exchange=exchange,
                        row_id=row_id,
                        col_id=col_id,
                        matrix_label="technosphere_matrix",
                    )
                    / production_amount
                    * edge_amount_eff
                )
                if isinstance(td_producer, Number):
                    td_producer = TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([td_producer]),
                    )
                initial_distribution = (t0_eff * td_producer).simplify()
                abs_td_producer = self.join_datetime_and_timedelta_distributions(
                    td_producer, t0_eff
                )
                abs_td_consumer = t0_eff

            # `join_datetime_and_timedelta_distributions` only carries producer
            # amounts forward (consumer is normally a unit `t0`). When the demand
            # is a TD with non-unit cohort amounts, we need to broadcast the
            # consumer amounts onto the FU edge so downstream sees the right
            # fleet sizes per cohort.
            if demand_td is not None and isinstance(abs_td_producer, TemporalDistribution):
                if isinstance(td_producer, TemporalDistribution):
                    prod_amount = td_producer.amount
                else:
                    prod_amount = np.array([td_producer])
                scaled = (
                    t0_eff.amount.reshape(-1, 1) * prod_amount.reshape(1, -1)
                ).ravel()
                abs_td_producer = TemporalDistribution(abs_td_producer.date, scaled)

            heappush(
                heap,
                (
                    1 / node.cumulative_score,
                    initial_distribution,
                    t0_eff,
                    abs_td_producer,
                    node,
                ),
            )

            timeline.append(
                Edge(
                    edge_type="production",  # FU exchange always type production (?)
                    distribution=initial_distribution,
                    leaf=False,
                    consumer=self.unique_id,
                    producer=node.activity_datapackage_id,
                    td_producer=td_producer,
                    td_consumer=t0_eff,
                    abs_td_producer=abs_td_producer,
                    abs_td_consumer=abs_td_consumer,
                )
            )

        while heap:
            _, td, td_parent, abs_td, node = heappop(heap)

            for edge in self.edge_mapping[node.unique_id]:
                row_id = self.nodes[edge.producer_unique_id].activity_datapackage_id
                col_id = node.activity_datapackage_id
                product_id = self.lca_object.dicts.product.reversed[edge.product_index]
                exchange = self.get_technosphere_exchange(
                    input_id=product_id,
                    output_id=col_id,
                )
                if not hasattr(exchange, "data"):
                    exchange = self.get_technosphere_exchange(
                        input_id=row_id,
                        output_id=col_id,
                    )
                    product_id = row_id

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
                        row_id=product_id,
                        col_id=col_id,
                        matrix_label="technosphere_matrix",
                    )
                    / abs(
                        get_reference_product_production_amount(
                            node.activity_datapackage_id, lca=self.lca_object
                        )
                    )
                )
                producer = self.nodes[edge.producer_unique_id]
                leaf = self.edge_ff(row_id)

                # If an edge does not have a TD, give it a td with timedelta=0 and the amount= 'edge value'
                if isinstance(td_producer, Number):
                    td_producer = TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([td_producer]),
                    )

                if product_id != row_id:
                    production_exchange = self.get_technosphere_exchange(
                        input_id=product_id,
                        output_id=row_id,
                    )
                    if (
                        hasattr(production_exchange, "data")
                        and production_exchange.data.get("type") == "production"
                    ):
                        production_amount = abs(
                            get_reference_product_production_amount(
                                row_id,
                                reference_product=product_id,
                                lca=self.lca_object,
                            )
                        )
                        production_td = (
                            self._exchange_value(
                                exchange=production_exchange,
                                row_id=product_id,
                                col_id=row_id,
                                matrix_label="technosphere_matrix",
                            )
                            / production_amount
                        )
                        if isinstance(production_td, Number):
                            production_td = TemporalDistribution(
                                date=np.array([0], dtype="timedelta64[Y]"),
                                amount=np.array([production_td]),
                            )
                        td_producer = (td_producer * production_td).simplify()

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
        Joins a relative or absolute TemporalDistribution (td_producer) with an
        absolute TemporalDistribution (td_consumer) to create a new
        TemporalDistribution.

        If the producer does not have a TemporalDistribution, the consumer's
        TemporalDistribution is returned to continue the timeline.
        If both the producer and consumer have TemporalDistributions, they are joined together.

        Parameters
        ----------
        td_producer : TemporalDistribution
            TemporalDistribution of the producer. Expected to be a timedelta or
            datetime TemporalDistribution.
        td_consumer : TemporalDistribution
            TemporalDistribution of the consumer. Expected to be a datetime
            TemporalDistribution.

        Returns
        -------
        TemporalDistribution
            A new TemporalDistribution that is the result of joining the producer
            and consumer TemporalDistributions.

        Raises
        ------
        ValueError
            If the dtype of `td_consumer.date` is not `datetime64[s]` or the dtype
            of `td_producer.date` is neither `datetime64[s]` nor `timedelta64[s]`.

        """
        return _join_datetime_and_timedelta_distributions(td_producer, td_consumer)


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
        demand_temporal_distributions: dict | None = None,
    ) -> None:
        self.lca_object = lca_object
        self.edge_ff = edge_filter_function if edge_filter_function else lambda x: False
        self.cutoff = cutoff
        self.static_activity_indices = static_activity_indices or set()
        self.demand_temporal_distributions = demand_temporal_distributions or {}

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
        """Get the reference product production amount."""
        activity = self.lca_object.dicts.activity.reversed[
            self.lca_object.dicts.activity[activity_id]
        ]
        return get_reference_product_production_amount(activity)

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
            demand_td = self.demand_temporal_distributions.get(fu_id)
            if demand_td is not None:
                td = demand_td
                fu_amount = float(np.abs(demand_td.amount).sum())
                t0_eff = demand_td
            else:
                fu_amount = demand_array[self.lca_object.dicts.activity[fu_id]]
                td = self.t0 * fu_amount
                t0_eff = self.t0

            timeline.append(
                Edge(
                    edge_type="production",
                    distribution=td,
                    leaf=False,
                    consumer=-1,
                    producer=fu_id,
                    td_producer=fu_amount,
                    td_consumer=t0_eff,
                    abs_td_producer=t0_eff,
                )
            )

            queue.append((fu_id, td, t0_eff, t0_eff, abs(fu_amount)))

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
    Join a relative or absolute TemporalDistribution (td_producer) with an
    absolute TemporalDistribution (td_consumer).

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
        if td_producer.date.dtype == timedelta_type:
            date = (
                td_consumer.date.reshape((-1, 1))
                + td_producer.date.reshape((1, -1))
            ).ravel()
        elif td_producer.date.dtype == datetime_type:
            date = np.array(len(td_consumer) * [td_producer.date]).ravel()
        else:
            raise ValueError(
                f"`td_producer.date` must have dtype `datetime64[s]` "
                f"or `timedelta64[s]`, "
                f"but got `{td_producer.date.dtype}`"
            )
        amount = np.array(len(td_consumer) * [td_producer.amount]).ravel()
        return TemporalDistribution(date, amount)
    else:
        raise ValueError(
            f"Can't join TemporalDistribution and something else: "
            f"Trying with {type(td_consumer)} and {type(td_producer)}"
        )
