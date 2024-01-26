from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, List

from bw_temporalis import TemporalisLCA, TemporalDistribution


@dataclass
class Edge:
    """
    Class for storing a temporal edge with source and target.

    Leaf edges link to a source process which is a leaf in
    our graph traversal (either through cutoff or a filter
    function).

    Attributes
    ----------
    distribution : TemporalDistribution
    leaf : bool
    consumer : int
    producer : int

    """

    distribution: TemporalDistribution
    leaf: bool
    consumer: int
    producer: int
    value: TemporalDistribution


class EdgeExtracter(TemporalisLCA):
    """
    Child class of TemporalisLCA that traverses the supply chain just as the parent class but can create a timeline of edges, in addition timeline of flows or nodes.
    The edge timeline is then used to match the timestamp of edges to that of background databases and to replace these edges with edges from these background databases using Brightway Datapackages
    The current implementation is work in progress and can only handle temporal distributions in the foreground.

    """
    def __init__(self, *args, edge_filter_function: Callable = None, **kwargs):
        super().__init__(*args, **kwargs)
        if edge_filter_function:
            self.edge_ff = edge_filter_function
        else:
            self.edge_ff = lambda x: False

    def build_edge_timeline(self) -> List:
        """
        Creates a timeline of the edges from the graph traversal. Starting from the edges of the functional unit node, it goes through each node using a heap, 
        and propagates the TemporalDistributions of the edges from node to node through time using convolution-operators during multiplication. 
        It stops in case the current edge is known to have no temporal distribution (=leaf) (e.g. part of background database).

        :return: A list of Edge instances with timestamps and amounts, and ids of its producing and consuming node.
        -------------------
        """
        heap = []
        timeline = []

        for edge in self.edge_mapping[self.unique_id]: # starting at the edges of the functional unit
            node = self.nodes[edge.producer_unique_id]
            heappush(
                heap,
                (
                    1 / node.cumulative_score,
                    self.t0 * edge.amount,
                    node,
                ),
            )
            timeline.append(
                Edge(
                    distribution=self.t0 * edge.amount,
                    leaf=False,
                    consumer=self.unique_id,
                    producer=node.activity_datapackage_id,
                    value=edge.amount,
                )
            )

        while heap:
            _, td, node = heappop(heap)

            for edge in self.edge_mapping[node.unique_id]:
                row_id = self.nodes[edge.producer_unique_id].activity_datapackage_id
                col_id = node.activity_datapackage_id
                exchange = self.get_technosphere_exchange(
                    input_id=row_id,
                    output_id=col_id,
                )
                value = (                           # value is the TemporalDistribution of the edge
                    self._exchange_value(
                        exchange=exchange,
                        row_id=row_id,
                        col_id=col_id,
                        matrix_label="technosphere_matrix",
                    )
                    / node.reference_product_production_amount 
                )
                producer = self.nodes[edge.producer_unique_id]
                leaf = self.edge_ff(row_id)

                distribution = (td * value).simplify() # convolution-multiplication of TemporalDistribution of consuming node (td) and consumed edge (edge) gives TD of producing node
                timeline.append(
                    Edge(
                        distribution=distribution,
                        leaf=leaf,
                        consumer=node.activity_datapackage_id,
                        producer=producer.activity_datapackage_id,
                        value=value,
                    )
                )
                if not leaf:
                    heappush(
                        heap,
                        (
                            1,
                            distribution,
                            producer,
                        ),
                    )
        return timeline