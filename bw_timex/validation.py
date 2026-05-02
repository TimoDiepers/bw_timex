from datetime import datetime
from typing import Callable, Literal, Optional, Union

import bw2data as bd
from bw_temporalis import TemporalDistribution
from pydantic import BaseModel, Field, field_validator, model_validator


class TimexLCAInputs(BaseModel):
    """Validates inputs to TimexLCA.__init__"""

    model_config = {"arbitrary_types_allowed": True}

    demand: dict
    method: tuple
    database_dates: Optional[dict] = None

    @field_validator("demand")
    @classmethod
    def validate_demand(cls, v: dict) -> dict:
        if not v:
            raise ValueError("demand must be a non-empty dictionary.")
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"demand values must be numeric, got {type(value).__name__} for key {key}."
                )
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: tuple) -> tuple:
        if not v:
            raise ValueError("method must be a non-empty tuple.")
        if not all(isinstance(item, str) for item in v):
            raise ValueError(
                f"method must be a tuple of strings, got types: {[type(item).__name__ for item in v]}."
            )
        if v not in bd.methods:
            raise ValueError(
                f"Method {v} not found in the current Brightway project. "
                f"Available methods can be listed with `bw2data.methods`."
            )
        return v

    @field_validator("database_dates")
    @classmethod
    def validate_database_dates(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "database_dates must be a non-empty dictionary if provided."
            )
        for database, value in v.items():
            if not isinstance(database, str):
                raise ValueError(
                    f"database_dates keys must be strings (database names), got {type(database).__name__}."
                )
            if not (value == "dynamic" or isinstance(value, datetime)):
                raise ValueError(
                    f"The value {value} for database '{database}' is neither 'dynamic' nor a datetime object. Check the format of the database_dates."
                )
            if database not in bd.databases:
                raise ValueError(
                    f"Database '{database}' not available in the Brightway2 project."
                )
        return v

    @model_validator(mode="after")
    def validate_demand_in_dynamic_databases(self) -> "TimexLCAInputs":
        if self.database_dates is None:
            return self
        dynamic_database_names = [
            db for db, value in self.database_dates.items() if value == "dynamic"
        ]
        for act in self.demand:
            if not isinstance(act, bd.Node):
                act = bd.get_activity(act)
            if act["database"] not in dynamic_database_names:
                raise ValueError(
                    f"Demand activity {act} from database {act['database']}: "
                    f"This database is not marked as 'dynamic' in database_dates. "
                    f"Please check."
                )
        return self


class BuildTimelineInputs(BaseModel):
    """Validates inputs to TimexLCA.build_timeline"""

    model_config = {"arbitrary_types_allowed": True}

    starting_datetime: Union[datetime, str] = "now"
    temporal_grouping: Literal["year", "month", "day", "hour"] = "year"
    interpolation_type: Literal["linear", "closest", "nearest"] = "linear"
    edge_filter_function: Optional[Callable] = None
    cutoff: float = Field(default=1e-9, gt=0)
    max_calc: int = Field(default=2000, gt=0)
    graph_traversal: Literal["priority", "bfs"] = "priority"

    @field_validator("starting_datetime")
    @classmethod
    def validate_starting_datetime(
        cls, v: Union[datetime, str]
    ) -> Union[datetime, str]:
        if isinstance(v, str):
            if v == "now":
                return v
            # Try to parse the string as a date
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(
                    f"starting_datetime string must be 'now' or a valid ISO format date string, got '{v}'."
                )
        return v

    @field_validator("interpolation_type")
    @classmethod
    def normalize_interpolation_type(cls, v: str) -> str:
        if v == "closest":
            return "nearest"
        return v


class LCIInputs(BaseModel):
    """Validates inputs to TimexLCA.lci"""

    build_dynamic_biosphere: bool = True
    expand_technosphere: bool = True

    @model_validator(mode="after")
    def validate_combination(self) -> "LCIInputs":
        if not self.expand_technosphere and not self.build_dynamic_biosphere:
            raise ValueError(
                "Currently, it is not possible to skip the construction of the dynamic "
                "biosphere when building the inventories from the timeline. "
                "Please either set build_dynamic_biosphere=True or expand_technosphere=True."
            )
        return self


class DynamicLCIAInputs(BaseModel):
    """Validates inputs to TimexLCA.dynamic_lcia"""

    model_config = {"arbitrary_types_allowed": True}

    metric: Literal["radiative_forcing", "GWP"] = "radiative_forcing"
    time_horizon: int = Field(default=100, gt=0)
    fixed_time_horizon: bool = False
    time_horizon_start: Optional[datetime] = None
    characterization_functions: Optional[dict] = None
    characterization_function_co2: Optional[Callable] = None
    use_disaggregated_lci: bool = False


class TemporalDistributionExchangeInputs(BaseModel):
    """Validates inputs to add_temporal_distribution_to_exchange"""

    model_config = {"arbitrary_types_allowed": True}

    temporal_distribution: TemporalDistribution

    @field_validator("temporal_distribution", mode="before")
    @classmethod
    def validate_temporal_distribution(cls, v):
        if not isinstance(v, TemporalDistribution):
            raise ValueError(
                f"temporal_distribution must be a TemporalDistribution instance, got {type(v).__name__}."
            )
        return v


class TemporalEvolutionExchangeInputs(BaseModel):
    """Validates inputs to add_temporal_evolution_to_exchange"""

    temporal_evolution_factors: Optional[dict] = None
    temporal_evolution_amounts: Optional[dict] = None
    temporal_evolution_reference: Literal["producer", "consumer"] = "producer"

    @model_validator(mode="after")
    def validate_mutual_exclusivity(self) -> "TemporalEvolutionExchangeInputs":
        if (
            self.temporal_evolution_factors is not None
            and self.temporal_evolution_amounts is not None
        ):
            raise ValueError(
                "'temporal_evolution_factors' and 'temporal_evolution_amounts' are "
                "mutually exclusive — use one or the other."
            )
        if (
            self.temporal_evolution_factors is None
            and self.temporal_evolution_amounts is None
        ):
            raise ValueError(
                "Either 'temporal_evolution_factors' or 'temporal_evolution_amounts' must be provided."
            )
        return self

    @field_validator("temporal_evolution_factors", "temporal_evolution_amounts")
    @classmethod
    def validate_temporal_evolution_dict(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        for key, value in v.items():
            if not isinstance(key, datetime):
                raise ValueError(
                    f"Keys must be datetime objects, got {type(key).__name__}: {key}."
                )
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Values must be numeric, got {type(value).__name__} for key {key}."
                )
        return v


class PlotDynamicInventoryInputs(BaseModel):
    """Validates inputs to TimexLCA.plot_dynamic_inventory"""

    bio_flows: list
    cumulative: bool = False

    @field_validator("bio_flows")
    @classmethod
    def validate_bio_flows(cls, v: list) -> list:
        if not v:
            raise ValueError("bio_flows must be a non-empty list.")
        for item in v:
            if not isinstance(item, int):
                raise ValueError(
                    f"bio_flows must contain integer database IDs, got {type(item).__name__}: {item}."
                )
        return v
