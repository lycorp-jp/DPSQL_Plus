import logging
import math
from typing import TYPE_CHECKING

from ..aggregation import Aggregation
from ..errors import InvalidPrivacyParametersError
from .accountant import Accountant
from .basic_accountant import BasicAccountant
from .utils import calc_alpha_beta_for_tau_thresholding

if TYPE_CHECKING:
    from ..dp_params import DPParams

logger = logging.getLogger(__name__)


class RenyiAccountant(Accountant):
    """
    Accountant for zero-concentrated differential privacy (zCDP)
    based on Renyi differential privacy.

    Args:
        epsilon (float): Privacy budget epsilon.
        delta (float): Privacy budget delta.
    """

    def __init__(self, epsilon: float, delta: float):
        super().__init__(epsilon, delta)
        self.alpha = 0.0
        self.beta = 0.0
        self.basic_accountant = BasicAccountant(epsilon, delta, warn_on_init=False)
        logger.info("Initializing RenyiAccountant")
        logger.debug("Budget: epsilon=%s delta=%s", epsilon, delta)

    def _calculate_budget(
        self, sensitivities: list[float], params: "DPParams"
    ) -> tuple[float, float]:
        """
        Calculate the privacy budget for an aggregation
        function with the given sensitivity.

        Args:
            sensitivities (list[float]): The sensitivities of the aggregation functions.
            params (DPParams): The differential privacy parameters.

        Returns:
            (alpha, beta) (tuple[float, float]):
                parameters of beta-approximate alpha-zCDP.
        """
        logger.debug(
            "RenyiAccountant._calculate_budget: sensitivities=%s", sensitivities
        )

        sigmas, tau, sigma_for_thresholding = params.get_noise_parameters(sensitivities)

        alpha_agg = 0.0
        for s, sigma in zip(sensitivities, sigmas, strict=False):
            alpha_agg += (s**2) / (2 * sigma**2)
        alpha_thresh, beta = calc_alpha_beta_for_tau_thresholding(
            params.min_frequency, tau, sigma_for_thresholding, params.contribution_bound
        )
        logger.debug(
            "Computed zCDP params: alpha_agg=%s alpha_thresh=%s beta=%s",
            alpha_agg,
            alpha_thresh,
            beta,
        )
        return alpha_agg + alpha_thresh, beta

    def calculate_budget(
        self, agg_funcs: list[Aggregation], params: "DPParams"
    ) -> tuple[float, float]:
        """
        Calculate the privacy budget for agg_funcs
        using zero-Concentrated Differential Privacy (zCDP).

        Args:
            agg_funcs (list[Aggregation]): The aggregation functions to be executed.
            params (DPParams): The differential privacy parameters.

        Returns:
            (alpha, beta) (tuple[float, float]):
                parameters of beta-approximate alpha-zCDP.
        """
        logger.info("Calculating zCDP budget (RenyiAccountant)")
        logger.debug("Agg funcs: %s", [a.name for a in agg_funcs])
        sensitivities = self.get_sensitivities(agg_funcs, params)
        result = self._calculate_budget(sensitivities, params)
        logger.debug("Result zCDP: alpha=%s beta=%s", *result)
        return result

    def calculate_min_epsilon(self, alpha: float, beta: float) -> float:
        """
        Calculate the minimum epsilon value which
        satisfies (epsilon, delta)-differential privacy.

        Args:
            alpha (float): A parameter of beta-approximate alpha-zCDP.
            beta (float): A parameter of beta-approximate alpha-zCDP.

        Returns:
            float: The minimum epsilon value.
            Returns infinity if beta is greater than or equal to delta.
            Returns 0 if delta is greater than or equal to 1.
        """
        logger.debug(
            "Calculating min epsilon from alpha=%s beta=%s (delta=%s)",
            alpha,
            beta,
            self.delta,
        )
        if self.delta >= 1.0:  # infinite budget
            result = 0.0
        elif beta >= self.delta:
            result = float("inf")
        else:
            result = alpha + math.sqrt(4 * alpha * math.log(1 / (self.delta - beta)))
        logger.debug("Min epsilon result: %s", result)
        return result

    def _check_budget(self, sensitivities: list[float], params: "DPParams") -> bool:
        logger.debug("RenyiAccountant._check_budget called")
        # Check with basic accountant first
        if self.basic_accountant._check_budget(sensitivities, params):
            logger.debug("BasicAccountant satisfied -> True")
            return True

        # If basic accountant is not satisfied, check with Renyi accountant
        alpha, beta = self._calculate_budget(sensitivities, params)
        min_epsilon = self.calculate_min_epsilon(
            self.alpha + alpha, min(1.0, self.beta + beta)
        )
        logger.debug(
            "Renyi check: min_epsilon=%s allowed=%s", min_epsilon, self.epsilon
        )
        if self.epsilon < min_epsilon:
            return False
        return True

    def update_budget(self, agg_funcs: list[Aggregation], params: "DPParams") -> None:
        logger.info("Updating RenyiAccountant budget")
        # Update basic accountant
        self.basic_accountant.update_budget(agg_funcs, params)

        # Update Renyi accountant
        alpha, beta = self.calculate_budget(agg_funcs, params)
        self.alpha += alpha
        self.beta += beta
        self.beta = min(self.beta, 1.0)  # delta cannot exceed 1.0
        logger.debug(
            "Updated alpha=%s beta=%s (clamped beta<=1)", self.alpha, self.beta
        )

    def remaining_queries(self, query_epsilon: float, query_delta: float) -> int:
        logger.info("Computing remaining queries (RenyiAccountant)")
        logger.debug("Per-query: epsilon=%s delta=%s", query_epsilon, query_delta)
        if self.delta >= 1.0:  # infinite budget
            return self.MAX_REMAINING_QUERIES
        if query_epsilon <= 0 or query_delta <= 0:
            raise InvalidPrivacyParametersError(
                "Per-query epsilon and delta must be strictly positive",
                context={"query_epsilon": query_epsilon, "query_delta": query_delta},
                hint="Use positive values, e.g. epsilon=0.05, delta=1e-6",
            )

        # Check with RenyiAccountant
        query_alpha = query_epsilon**2 / 2
        query_beta = query_delta
        logger.debug("Per-query zCDP: alpha=%s beta=%s", query_alpha, query_beta)

        n_queries = 0
        current_alpha = self.alpha + query_alpha
        current_beta = self.beta + query_beta
        while self.calculate_min_epsilon(current_alpha, current_beta) < self.epsilon:
            n_queries += 1
            current_alpha += query_alpha
            current_beta += query_beta
            if n_queries >= self.MAX_REMAINING_QUERIES:
                break
        logger.debug("Max queries by Renyi: %s", n_queries)

        # Return the maximum number of queries allowed by either accountant
        n_queries = max(
            n_queries,
            self.basic_accountant.remaining_queries(query_epsilon, query_delta),
        )
        logger.debug("Final remaining queries (max of accountants): %s", n_queries)
        return n_queries
