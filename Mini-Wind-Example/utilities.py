from datetime import datetime, timedelta
from typing import Union, Tuple, Optional
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
import uuid
import logging
import numpy as np


def find_closest_date(target, dates):
    """
    Find the closest date to the target in the dates list.

    :param target: Target datetime.datetime object.
    :param dates: List of datetime.datetime objects.
    :return: Closest datetime.datetime object from the list.

    ---------------------
    # Example usage
    target = datetime.strptime("2023-01-15", "%Y-%m-%d")
    dates_list = [
        datetime.strptime("2020", "%Y"),
        datetime.strptime("2022", "%Y"),
        datetime.strptime("2025", "%Y"),
    ]

    print(closest_date(target, dates_list))
    """

    # If the list is empty, return None
    if not dates:
        return None

    # Sort the dates
    dates = sorted(dates)

    # Use min function with a key based on the absolute difference between the target and each date
    closest = min(dates, key=lambda date: abs(target - date))

    return closest

def safety_razor(
        consumer: Union[bd.Node, Tuple[str, str], int], 
        previous_producer: Union[bd.Node, Tuple[str, str], int], 
        new_producer: Union[bd.Node, Tuple[str, str], int], 
        datapackage: Optional[bwp.Datapackage] = None,
        amount: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bwp.Datapackage:
    """Replace an existing edge with another edge. Zeroes out the existing edge.

    Inputs:
    consumer: Union[bd.Node, Tuple[str, str], int]
        The consuming node 
    previous_producer: Union[bd.Node, Tuple[str, str], int]
        The producing node which should be replaced
    new_producer: Union[bd.Node, Tuple[str, str], int]
        The new producing node
    datapackage: Optional[bwp.Datapackage]
        Append to this datapackage, if available. Otherwise create a new datapackage.
    amount: Optional[float]
        Amount of the new edge. Will be the *sum of all (previous_producer, consumer) edge amounts if not provided.
    name: Optional[str]
        Name of this datapackage resource.
    
    Returns a `bw_processing.Datapackage` with the modified data."""

    def resolve_node(node: Union[bd.Node, Tuple[str, str], int]) -> bd.Node:
        """Return a Brightway node from many different input possibilities.
        
        This isn't super-efficient - you could look up the `id` values ahead of time.
        In production you don't need fancy logging messages."""
        if isinstance(node, tuple):
            assert len(node) == 2
            return bd.get_node(database=node[0], code=node[1])
        elif isinstance(node, int):
            return bd.get_node(id=int)
        elif isinstance(node, bd.Node):
            return node
        else:
            raise ValueError(f"Can't understand {node}")
                
    consumer = resolve_node(consumer)
    previous_producer = resolve_node(previous_producer)
    new_producer = resolve_node(new_producer)

    assert new_producer.get("type", "process") == "process", "Wrong type of edge source"
    # Remove if creating new edge instead of moving or replacing existing an edge
    assert any(exc.input == previous_producer for exc in consumer.technosphere())

    if not name:
        name = uuid.uuid4().hex
        # logger.info(f"Using random name {name}")

    if not amount:
        amount = sum(
            exc['amount'] 
            for exc in consumer.technosphere() 
            if exc.input == previous_producer
        )
        # logger.info(f"Using database net amount {amount}")

    # logger.info(f"Zeroing exchange from {previous_producer} to {consumer}")
    # logger.info(f"Adding exchange of {amount} {new_producer} to {consumer}")

    if datapackage is None:
        datapackage = bwp.create_datapackage()

    datapackage.add_persistent_vector(
        # This function would need to be adapted for biosphere edges
        matrix="technosphere_matrix",
        name=name,
        data_array=np.array([0, amount], dtype=float),
        indices_array=np.array([
                (previous_producer.id, consumer.id), 
                (new_producer.id, consumer.id)
            ], dtype=bwp.INDICES_DTYPE),
        flip_array=np.array([False, True], dtype=bool)
    )  
    return datapackage