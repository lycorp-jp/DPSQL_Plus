import logging
from typing import TYPE_CHECKING

from dp_accounting.pld.common import DifferentialPrivacyParameters
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution,
    from_gaussian_mechanism,
    from_privacy_parameters,
)

from ..aggregation import Aggregation
from ..errors import InvalidPrivacyParametersError
from .accountant import Accountant
from .basic_accountant import BasicAccountant
from .utils import calc_pld_for_tau_thresholding

if TYPE_CHECKING:
    from ..dp_params import DPParams

logger = logging.getLogger(__name__)


class PLDAccountant(Accountant):
    """
    Accountant based on numerical composition.

    Args:
        epsilon (float): Privacy budget epsilon.
        delta (float): Privacy budget delta.
        discretization_interval (float): The discretization interval for the
            privacy loss distribution.
    """

    def __init__(
        self, epsilon: float, delta: float, discretization_interval: float = 1e-4
    ):
        super().__init__(epsilon, delta)
        if discretization_interval <= 0:
            raise InvalidPrivacyParametersError(
                "discretization_interval must be > 0",
                context={"discretization_interval": discretization_interval},
                hint="Use a small positive float, e.g. 1e-4",
            )
        self.discretization_interval = discretization_interval
        self.pld: PrivacyLossDistribution | None = None
        self.basic_accountant = BasicAccountant(epsilon, delta, warn_on_init=False)
        logger.info("Initializing PLDAccountant")
        logger.debug(
            "Budget: epsilon=%s delta=%s discretization_interval=%s",
            epsilon,
            delta,
            discretization_interval,
        )

    def _compute_pld(
        self, sensitivities: list[float], params: "DPParams"
    ) -> PrivacyLossDistribution:
        """
        Calculate the privacy loss distribution for an
        aggregation function with the given sensitivity.

        Args:
            sensitivities (list[float]): The sensitivities of the aggregation functions.
            params (DPParams): The differential privacy parameters.

        Returns:
            PrivacyLossDistribution: The privacy loss distribution.
        """
        logger.debug("Computing PLD for sensitivities=%s", sensitivities)

        sigmas, tau, sigma_for_thresholding = params.get_noise_parameters(sensitivities)

        pld = None

        for s, sigma in zip(sensitivities, sigmas, strict=False):
            pld_gaussian = from_gaussian_mechanism(
                sigma,
                s,
                value_discretization_interval=self.discretization_interval,
            )
            if pld is None:
                pld = pld_gaussian
            else:
                pld = pld.compose(pld_gaussian)

        logger.debug("PLD composed for Gaussian mechanisms")

        pld_thresholding = calc_pld_for_tau_thresholding(
            params.min_frequency,
            tau,
            sigma_for_thresholding,
            params.contribution_bound,
            self.discretization_interval,
        )
        if pld is None:
            pld = pld_thresholding
        else:
            pld = pld.compose(pld_thresholding)

        logger.debug("PLD thresholding composed")
        return pld

    def compute_pld(
        self, agg_funcs: list[Aggregation], params: "DPParams"
    ) -> PrivacyLossDistribution:
        """
        Calculate the privacy loss distribution for aggregation functions.

        Args:
            agg_funcs (list[Aggregation]): The aggregation functions to be executed.
            params (DPParams): The differential privacy parameters.

        Returns:
            PrivacyLossDistribution: The privacy loss distribution.
        """
        logger.info("Computing PLD for agg funcs")
        logger.debug("Agg funcs: %s", [a.name for a in agg_funcs])

        sensitivities = self.get_sensitivities(agg_funcs, params)
        return self._compute_pld(sensitivities, params)

    def calculate_min_epsilon(self, pld: PrivacyLossDistribution) -> float:
        """
        Calculate the minimum epsilon value which satisfies
        (epsilon, delta)-differential privacy.

        Args:
            pld (PrivacyLossDistribution): The privacy loss distribution.

        Returns:
            float: The minimum epsilon value.
        """
        result = pld.get_epsilon_for_delta(self.delta)
        logger.debug("PLD min epsilon for delta=%s -> %s", self.delta, result)
        return result

    def _check_budget(self, sensitivities: list[float], params: "DPParams") -> bool:
        logger.debug("PLDAccountant._check_budget called")
        # Check with basic accountant first
        if self.basic_accountant._check_budget(sensitivities, params):
            logger.debug("BasicAccountant satisfied -> True")
            return True

        # If basic accountant is not satisfied, check with PLD
        pld = self._compute_pld(sensitivities, params)
        if self.pld is None:
            composed = pld
        else:
            composed = self.pld.compose(pld)
        min_epsilon = self.calculate_min_epsilon(composed)
        logger.debug("PLD check: min_epsilon=%s allowed=%s", min_epsilon, self.epsilon)
        if self.epsilon < min_epsilon:
            return False
        return True

    def update_budget(self, agg_funcs: list[Aggregation], params: "DPParams") -> None:
        logger.info("Updating PLDAccountant budget")
        # Update basic accountant
        self.basic_accountant.update_budget(agg_funcs, params)

        # Update PLD
        pld = self.compute_pld(agg_funcs, params)
        if self.pld is None:
            self.pld = pld
        else:
            self.pld = self.pld.compose(pld)
        logger.debug("PLD updated (composed). Is None? %s", self.pld is None)

    def remaining_queries(self, query_epsilon: float, query_delta: float) -> int:
        logger.info("Computing remaining queries (PLDAccountant)")
        logger.debug("Per-query: epsilon=%s delta=%s", query_epsilon, query_delta)
        if self.delta >= 1.0:  # infinite budget
            return self.MAX_REMAINING_QUERIES
        if query_epsilon <= 0 or query_delta <= 0:
            raise InvalidPrivacyParametersError(
                "Per-query epsilon and delta must be strictly positive",
                context={"query_epsilon": query_epsilon, "query_delta": query_delta},
                hint="Use positive values, e.g. epsilon=0.1, delta=1e-6",
            )

        # Check with PLDAccountant
        query_pld = from_privacy_parameters(
            DifferentialPrivacyParameters(query_epsilon, query_delta),
            value_discretization_interval=self.discretization_interval,
        )
        n_queries = 0
        current_pld = self.pld.compose(query_pld) if self.pld is not None else query_pld
        logger.debug("Composed current PLD for query")
        while current_pld.get_epsilon_for_delta(self.delta) < self.epsilon:
            n_queries += 1
            current_pld = current_pld.compose(query_pld)
            if n_queries >= self.MAX_REMAINING_QUERIES:
                break

        logger.debug("Max queries by PLD: %s", n_queries)
        # Return the maximum number of queries allowed by either accountant
        n_queries = max(
            n_queries,
            self.basic_accountant.remaining_queries(query_epsilon, query_delta),
        )
        logger.debug("Final remaining queries (max of accountants): %s", n_queries)
        return n_queries
