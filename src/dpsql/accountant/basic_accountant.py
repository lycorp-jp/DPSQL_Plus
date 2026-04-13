import logging
import math
import warnings
from typing import TYPE_CHECKING

from ..aggregation import Aggregation
from ..errors import InvalidPrivacyParametersError
from .accountant import Accountant

if TYPE_CHECKING:
    from ..dp_params import DPParams

logger = logging.getLogger(__name__)


class BasicAccountant(Accountant):
    """
    Accountant which calculates the privacy budget using basic composition.
    Renyi Accountant and PLDAccountant provide advanced composition techniques,
    and thus yield tighter bounds on privacy budget consumption.
    Thus, BasicAccountant is mainly for testing and debugging purposes.

    Args:
        epsilon (float): Privacy budget epsilon.
        delta (float): Privacy budget delta.
    """

    def __init__(self, epsilon: float, delta: float, warn_on_init: bool = True):
        if warn_on_init:
            warnings.warn(
                "BasicAccountant is mainly for testing and debugging purposes. "
                "For production use, consider using RenyiAccountant or PLDAccountant "
                "for tighter privacy guarantees.",
                category=UserWarning,
                stacklevel=2,
            )
        super().__init__(epsilon, delta)
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        logger.info("Initializing BasicAccountant")
        logger.debug("Budget: epsilon=%s delta=%s", epsilon, delta)

    def _check_budget(self, sensitivities: list[float], params: "DPParams") -> bool:
        logger.debug("BasicAccountant._check_budget called")
        if params.epsilon is None or params.delta is None:
            return False
        elif self.delta >= 1.0:  # infinite budget
            logger.debug("Infinite budget (delta>=1) -> True")
            return True
        else:
            updated_epsilon, updated_delta = (
                self.current_epsilon + params.epsilon,
                self.current_delta + params.delta,
            )
            updated_delta = min(updated_delta, 1.0)  # delta cannot exceed 1.0
            logger.debug(
                "Updated budget: epsilon=%s (<=%s) delta=%s (<=%s) -> %s",
                updated_epsilon,
                self.epsilon,
                updated_delta,
                self.delta,
                updated_epsilon <= self.epsilon and updated_delta <= self.delta,
            )
            return updated_epsilon <= self.epsilon and updated_delta <= self.delta

    def update_budget(self, agg_funcs: list[Aggregation], params: "DPParams") -> None:
        logger.info("Updating BasicAccountant budget")
        if params.epsilon is None or params.delta is None:
            self.current_epsilon = float("inf")
            self.current_delta = float("inf")
            logger.debug("Params missing (epsilon/delta None) -> set to inf")
        else:
            self.current_epsilon += params.epsilon
            self.current_delta += params.delta
            self.current_delta = min(self.current_delta, 1.0)  # delta cannot exceed 1.0
            logger.debug(
                "Updated to: current_epsilon=%s current_delta=%s",
                self.current_epsilon,
                self.current_delta,
            )

    def remaining_queries(self, query_epsilon: float, query_delta: float) -> int:
        logger.info("Computing remaining queries (BasicAccountant)")
        logger.debug("Per-query: epsilon=%s delta=%s", query_epsilon, query_delta)
        if self.delta >= 1.0:  # infinite budget
            return self.MAX_REMAINING_QUERIES
        if query_epsilon <= 0 or query_delta <= 0:
            raise InvalidPrivacyParametersError(
                "Per-query epsilon and delta must be strictly positive",
                context={"query_epsilon": query_epsilon, "query_delta": query_delta},
                hint="Provide positive values (e.g., epsilon=0.1, delta=1e-6)",
            )

        # Calculate remaining budget
        remaining_epsilon = self.epsilon - self.current_epsilon
        remaining_delta = self.delta - self.current_delta

        # If already exceeded budget, return 0
        if remaining_epsilon <= 0 or remaining_delta <= 0:
            return 0

        # Calculate maximum queries based on epsilon and delta constraints
        # Use floor to be conservative (safe side)
        max_queries_epsilon = math.floor(remaining_epsilon / query_epsilon)
        max_queries_delta = math.floor(remaining_delta / query_delta)

        logger.debug(
            "Remaining budget: epsilon=%s delta=%s -> max_epsilon=%s max_delta=%s",
            remaining_epsilon,
            remaining_delta,
            max_queries_epsilon,
            max_queries_delta,
        )
        # Return the minimum (most restrictive constraint)
        return min(max_queries_epsilon, max_queries_delta)
