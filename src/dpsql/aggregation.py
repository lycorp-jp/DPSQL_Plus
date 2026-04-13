import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from .errors import AggregationError
from .utils import safely_get_threshold

logger = logging.getLogger(__name__)


class Aggregation(Enum):
    COUNT = auto()
    COUNT_DISTINCT = auto()
    SUM = auto()
    AVG = auto()
    SQUARED_SUM = auto()
    PRODUCT_SUM = auto()
    STDDEV_SAMP = auto()
    STDDEV_POP = auto()
    VAR_SAMP = auto()
    VAR_POP = auto()
    COVAR_SAMP = auto()
    COVAR_POP = auto()
    NONE = auto()

    @classmethod
    def from_str(cls, name: str | None) -> "Aggregation":
        logger.debug("Parsing aggregation from string: %s", name)
        if name is None:
            logger.debug("Aggregation name is None -> returning NONE")
            return cls.NONE
        else:
            try:
                agg = cls[name]
                logger.debug("Resolved aggregation: %s", agg.name)
                return agg
            except KeyError as err:
                logger.error("Unsupported aggregation name: %s", name)
                raise AggregationError(
                    f"Unsupported aggregation name: {name}",
                    context={"input_name": name, "allowed": [m.name for m in cls]},
                    hint="Use one of the defined enum members (e.g. COUNT, SUM)",
                    cause=err,
                ) from err

    def get_sensitivity(
        self,
        contribution_bound: float,
        clipping_threshold: list[tuple[float, float]] | None = None,
    ) -> float:
        """
        Get the sensitivity of the aggregation function.

        Args:
            contribution_bound (float): The contribution bound.
            clipping_threshold (list[tuple[float, float]] | None):
                The clipping thresholds for each column.

        Returns:
            float: The l^2 sensitivity of the aggregation function.
        """
        logger.debug(
            "Computing sensitivity: agg=%s, contribution_bound=%s,"
            " clipping_threshold=%s",
            self.name,
            contribution_bound,
            clipping_threshold,
        )
        match self:
            case Aggregation.NONE:
                logger.error("Sensitivity requested for NONE aggregation")
                raise AggregationError(
                    "Invalid aggregation type `NONE` for sensitivity calculation",
                    context={"aggregation": "NONE"},
                    hint="Specify a concrete aggregation before computing sensitivity",
                )
            case Aggregation.COUNT | Aggregation.COUNT_DISTINCT:
                sensitivity = contribution_bound
                logger.debug("Sensitivity (COUNT/COUNT_DISTINCT) = %s", sensitivity)
                return sensitivity
            case Aggregation.SUM:
                lower_bound, upper_bound = safely_get_threshold(
                    clipping_threshold, 0, self.name
                )
                C = max(abs(lower_bound), abs(upper_bound))
                sensitivity = contribution_bound * C
                logger.debug(
                    "Sensitivity (SUM): lower=%s upper=%s C=%s -> %s",
                    lower_bound,
                    upper_bound,
                    C,
                    sensitivity,
                )
                return sensitivity
            case (
                Aggregation.AVG
                | Aggregation.STDDEV_POP
                | Aggregation.STDDEV_SAMP
                | Aggregation.VAR_POP
                | Aggregation.VAR_SAMP
                | Aggregation.COVAR_POP
                | Aggregation.COVAR_SAMP
            ):
                lower_bound, upper_bound = safely_get_threshold(
                    clipping_threshold, 0, self.name
                )
                # Our implementation of AVG, STDDEV, VAR, and COVAR is based on
                # Bezier mechanism (https://arxiv.org/abs/2509.04919).
                # The l^2 sensitivity of Bernstein representation is bounded by 1,
                # which is scaled to contribution_bound * (upper_bound - lower_bound)
                # in our setting.
                sensitivity = contribution_bound * (upper_bound - lower_bound)
                logger.debug(
                    "Sensitivity (%s): lower=%s upper=%s -> %s",
                    self.name,
                    lower_bound,
                    upper_bound,
                    sensitivity,
                )
                return sensitivity
            case _:
                logger.error(
                    "Unsupported aggregation type for sensitivity: %s", self.name
                )
                raise AggregationError(
                    "Unsupported aggregation type",
                    context={"aggregation": self.name},
                    hint="Check Aggregation enum definition",
                )


@dataclass
class AggregationColumn:
    aggregation_type: Aggregation  # Aggregation type
    columns: list[str]  # List of column names to which the aggregation is applied.
    alias: str | None = None  # Alias for the aggregation result.
    parameters: list[float] = field(
        default_factory=list
    )  # Additional parameters for the aggregation.
