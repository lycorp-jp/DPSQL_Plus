import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..aggregation import Aggregation
from ..errors import InvalidPrivacyParametersError

if TYPE_CHECKING:
    from ..dp_params import DPParams

logger = logging.getLogger(__name__)


class Accountant(ABC):
    """
    Abstract class for an accountant that checks and updates the privacy budget.

    Args:
        epsilon (float): Privacy budget epsilon.
        delta (float): Privacy budget delta. delta = 1 represents infinite budget.
    """

    MAX_REMAINING_QUERIES = 10**6

    def __init__(self, epsilon: float, delta: float):
        if epsilon <= 0 or delta <= 0 or delta > 1:
            raise InvalidPrivacyParametersError(
                "Invalid privacy parameters: epsilon must be > 0; "
                "delta must be in (0, 1]",
                context={"epsilon": epsilon, "delta": delta},
                hint="Ensure epsilon > 0 and 0 < delta <= 1",
            )
        self.epsilon = epsilon
        self.delta = delta

    def get_sensitivities(
        self, agg_funcs: list[Aggregation], params: "DPParams"
    ) -> list[float]:
        """
        Get the sensitivities of the aggregation functions.

        Args:
            agg_funcs (list[Aggregation]): The aggregation functions.
            params (DPParams): The differential privacy parameters.

        Returns:
            list[float]: The sensitivities of the aggregation functions.
        """
        logger.debug("Computing sensitivities for %s aggs", len(agg_funcs))

        sensitivities: list[float] = []
        for i, agg_func in enumerate(agg_funcs):
            clipping_threshold = params.clipping_thresholds[i]
            sensitivities.append(
                agg_func.get_sensitivity(params.contribution_bound, clipping_threshold)
            )
        logger.debug("Sensitivities computed: %s", sensitivities)
        return sensitivities

    def check_budget(self, agg_funcs: list[Aggregation], params: "DPParams") -> bool:
        """
        Check if the privacy budget is sufficient for the given parameters.

        Args:
            agg_funcs (list[Aggregation]): The aggregation functions.
            params (DPParams): The differential privacy parameters.

        Returns:
            bool: True if the privacy budget is sufficient, False otherwise.
        """
        logger.info("Checking privacy budget sufficiency")
        sensitivities = self.get_sensitivities(agg_funcs, params)
        result = self._check_budget(sensitivities, params)
        logger.debug("Budget check result: %s", result)
        return result

    @abstractmethod
    def _check_budget(self, sensitivities: list[float], params: "DPParams") -> bool:
        """
        Check if the privacy budget is sufficient
        for the given parameters and sensitivities.

        Args:
            sensitivities (list[float]): The sensitivities of the aggregation functions.
            params (DPParams): The differential privacy parameters.

        Returns:
            bool: True if the privacy budget is sufficient, False otherwise.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_budget(self, agg_funcs: list[Aggregation], params: "DPParams") -> None:
        """
        Update the privacy budget based on the given parameters.

        Args:
            agg_func (list[Aggregation]): The aggregation functions.
            params (DPParams): The differential privacy parameters.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def remaining_queries(self, query_epsilon: float, query_delta: float) -> int:
        """
        Calculate how many more queries can be executed
        with the given epsilon and delta.

        Args:
            query_epsilon (float): The epsilon cost per query.
            query_delta (float): The delta cost per query.

        Returns:
            int: The maximum number of queries that can still be executed.
            Returns 0 if no more queries can be executed.
        """
        raise NotImplementedError("Subclasses must implement this method")
