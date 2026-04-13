import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import stats

from .aggregation import Aggregation, AggregationColumn
from .errors import (
    AggregationError,
    InvalidPrivacyParametersError,
    UnsupportedQueryError,
)
from .utils import calibrate_analytic_gaussian_mechanism
from .validator.utils import get_privacy_definition

if TYPE_CHECKING:
    from .accountant import Accountant

logger = logging.getLogger(__name__)


@dataclass
class DPParams:
    """
    Parameters for differential privacy.
    Either of (epsilon, delta) or (sigmas, tau, sigma_for_thresholding)
    must be provided.

    Args:
        contribution_bound (int): The maximum number of records that
            can be contributed by a single privacy unit.
        clipping_thresholds (list[list[tuple[float, float]] | None]):
            The list of clipping parameters for input data,
            used to clip the input within the range [L, U] where each tuple is (L, U).
            Each element is either None (for COUNT/COUNT_DISTINCT)
            or a list of clipping thresholds
            for each column in a multi-column aggregation.
        min_frequency (int): The threshold for first thresholding before adding noise.
            It satisfies minimum frequency rule. (Default: 1)
        epsilon (float): The privacy budget for the query.
        delta (float): The probability of failure for the privacy guarantee.
        sigmas (list[float]): The standard deviations for the Gaussian mechanism.
            Deprecated: Use epsilon and delta instead.
        tau (float): The threshold for second thresholding after adding noise.
            It is expected to be greater than or equal to min_frequency.
            Deprecated: Use epsilon and delta instead.
        sigma_for_thresholding (float): The standard deviation
            for the Gaussian mechanism before the second thresholding.
            Deprecated: Use epsilon and delta instead.
        accountant_class (type[Accountant]): The privacy accountant class
            to use for calculating tight privacy budget allocation.
    """

    contribution_bound: int
    clipping_thresholds: list[list[tuple[float, float]] | None]
    min_frequency: int = 1
    epsilon: float | None = None
    delta: float | None = None
    sigmas: list[float] | None = None
    tau: float | None = None
    sigma_for_thresholding: float | None = None
    accountant_class: type["Accountant"] | None = None

    def __post_init__(self):
        # Validate the parameters
        logger.info("Validating DPParams")
        logger.debug(
            "Init values: contribution_bound=%s min_frequency=%s epsilon=%s delta=%s "
            "sigmas=%s tau=%s sigma_for_thresholding=%s",
            self.contribution_bound,
            self.min_frequency,
            self.epsilon,
            self.delta,
            self.sigmas,
            self.tau,
            self.sigma_for_thresholding,
        )

        # Show deprecation warning if sigmas is used
        if self.sigmas is not None:
            warnings.warn(
                "The 'sigmas' parameter is deprecated."
                "Use 'epsilon' and 'delta' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        # contribution_bound must be at least 1
        if self.contribution_bound < 1:
            raise InvalidPrivacyParametersError(
                "Invalid `contribution_bound` (must be >= 1)",
                context={"contribution_bound": self.contribution_bound},
                hint="Set contribution_bound to an integer >= 1",
            )
        # min_frequency must be at least 1
        if self.min_frequency < 1:
            raise InvalidPrivacyParametersError(
                "Invalid `min_frequency` (must be >= 1)",
                context={"min_frequency": self.min_frequency},
                hint="Set min_frequency to an integer >= 1",
            )

        if self.epsilon is None or self.delta is None:
            # If epsilon and delta are not set, sigmas, tau,
            # and sigma_for_thresholding must be set
            if (
                self.sigmas is None
                or self.tau is None
                or self.sigma_for_thresholding is None
            ):
                raise InvalidPrivacyParametersError(
                    "Missing required privacy parameters "
                    "(provide either (epsilon, delta) "
                    "or (sigmas, tau, sigma_for_thresholding))",
                    context={
                        "epsilon": self.epsilon,
                        "delta": self.delta,
                        "sigmas": self.sigmas,
                        "tau": self.tau,
                        "sigma_for_thresholding": self.sigma_for_thresholding,
                    },
                    hint="Supply epsilon & delta OR sigmas, tau, "
                    "sigma_for_thresholding",
                )
            logger.debug("Using (sigmas, tau, sigma_for_thresholding) mode")
            # tau must be greater than or equal to min_frequency
            if self.tau < self.min_frequency:
                raise InvalidPrivacyParametersError(
                    "Invalid `tau` (must be >= min_frequency)",
                    context={"tau": self.tau, "min_frequency": self.min_frequency},
                    hint="Increase tau or decrease min_frequency",
                )
            # sigmas and sigma_for_thresholding must be non-negative
            if any(s < 0 for s in self.sigmas) or self.sigma_for_thresholding < 0:
                raise InvalidPrivacyParametersError(
                    "Invalid noise scales "
                    "(sigmas and sigma_for_thresholding must be >= 0)",
                    context={
                        "sigmas": self.sigmas,
                        "sigma_for_thresholding": self.sigma_for_thresholding,
                    },
                    hint="Ensure all noise scale parameters are >= 0",
                )
        else:
            # epsilon and delta must be positive and delta must be <= 1
            if self.epsilon <= 0 or self.delta <= 0 or self.delta > 1:
                raise InvalidPrivacyParametersError(
                    "Invalid (epsilon, delta) "
                    "(epsilon > 0 and 0 < delta <= 1 required)",
                    context={"epsilon": self.epsilon, "delta": self.delta},
                    hint="Use epsilon > 0 and 0 < delta <= 1",
                )

        logger.info("DPParams validation completed")

    def get_noise_parameters(
        self, sensitivities: list[float]
    ) -> tuple[list[float], float, float]:
        """
        Get the noise parameters (sigmas, tau, sigma_for_thresholding).

        Args:
            sensitivities (list[float]): The sensitivities of the aggregation functions.
        Returns:
            tuple[list[float], float, float]: The noise parameters
            (sigmas, tau, sigma_for_thresholding).
        """
        logger.info("Computing noise parameters")
        logger.debug("Sensitivities: %s", sensitivities)
        # If all noise parameters are already set, return them
        if (
            self.tau is not None
            and self.sigma_for_thresholding is not None
            and self.sigmas is not None
        ):
            return (self.sigmas, self.tau, self.sigma_for_thresholding)

        # Error handling
        if self.epsilon is None or self.delta is None:
            raise InvalidPrivacyParametersError(
                "Missing (epsilon, delta) to compute tau and sigma_for_thresholding",
                context={"epsilon": self.epsilon, "delta": self.delta},
                hint="Initialize DPParams with epsilon "
                "and delta or preset tau & sigma_for_thresholding",
            )
        if len(sensitivities) != len(self.clipping_thresholds):
            raise InvalidPrivacyParametersError(
                "Mismatch between sensitivities and clipping thresholds count",
                context={
                    "num_sensitivities": len(sensitivities),
                    "num_clipping_thresholds": len(self.clipping_thresholds),
                },
                hint="Provide one clipping threshold per aggregation column",
            )

        if self.accountant_class is None:
            # Directly use epsilon and delta without binary search
            logger.debug("No accountant_class; computing directly with epsilon/delta")
            sigmas, tau, sigma_for_thresholding = self._compute_noise_parameters(
                self.epsilon, self.delta, sensitivities
            )
        else:
            # For other accountants, use binary search
            # to find optimal epsilon allocation
            logger.debug(
                "Using accountant_class=%s for epsilon allocation",
                self.accountant_class,
            )
            accountant = self.accountant_class(self.epsilon, self.delta)
            epsilon_ok = self._binary_search_epsilon(
                self.epsilon, self.delta, sensitivities, accountant
            )
            sigmas, tau, sigma_for_thresholding = self._compute_noise_parameters(
                epsilon_ok, self.delta, sensitivities
            )

        logger.debug(
            "Noise parameters computed: sigmas=%s tau=%s sigma_for_thresholding=%s",
            sigmas,
            tau,
            sigma_for_thresholding,
        )
        return sigmas, tau, sigma_for_thresholding

    def _binary_search_epsilon(
        self,
        epsilon: float,
        delta: float,
        sensitivities: list[float],
        accountant: "Accountant",
        relative_max_epsilon: float = 2**5,
        relative_tol: float = 1 / 2**5,
    ) -> float:
        """Binary search to find the maximum epsilon
        that yields noisy parameters satisfying the budget.

        Args:
            epsilon (float): The total privacy budget epsilon.
            delta (float): The total privacy budget delta.
            sensitivities (list[float]): The sensitivities of the aggregation functions.
            accountant (Accountant): The privacy accountant to use.
            relative_max_epsilon (float): The relative maximum epsilon for the search.
            relative_tol (float): The relative tolerance for convergence.

        Returns:
            float: The maximum epsilon satisfying the budget.
        """
        logger.debug("Starting binary search for epsilon")
        logger.debug(
            "Initial: epsilon=%s delta=%s rel_max=%s rel_tol=%s",
            epsilon,
            delta,
            relative_max_epsilon,
            relative_tol,
        )
        epsilon_ok = epsilon
        epsilon_ng = epsilon * relative_max_epsilon  # initial upper bound
        while epsilon_ng - epsilon_ok > epsilon * relative_tol:
            epsilon_mid = (epsilon_ng + epsilon_ok) / 2
            sigmas, tau, sigma_for_thresholding = self._compute_noise_parameters(
                epsilon_mid, delta, sensitivities
            )
            ok = accountant._check_budget(
                sensitivities=sensitivities,
                params=DPParams(
                    contribution_bound=self.contribution_bound,
                    clipping_thresholds=self.clipping_thresholds,
                    min_frequency=self.min_frequency,
                    sigmas=sigmas,
                    tau=tau,
                    sigma_for_thresholding=sigma_for_thresholding,
                ),
            )
            if ok:
                epsilon_ok = epsilon_mid
            else:
                epsilon_ng = epsilon_mid
            logger.debug(
                "Mid epsilon=%s -> ok=%s (epsilon_ok=%s epsilon_ng=%s)",
                epsilon_mid,
                ok,
                epsilon_ok,
                epsilon_ng,
            )
        # epsilon_ok is guaranteed to be greater than or equal to epsilon
        # because the initial lower bound is set to epsilon
        logger.debug("Binary search completed: epsilon_ok=%s", epsilon_ok)
        return epsilon_ok

    def _compute_noise_parameters(
        self, epsilon: float, delta: float, sensitivities: list[float]
    ):
        logger.debug(
            "Compute noise params: epsilon=%s delta=%s sensitivities=%s",
            epsilon,
            delta,
            sensitivities,
        )
        epsilon_per_agg = epsilon / (len(sensitivities) + 1)
        delta_per_agg = delta / (len(sensitivities) + 1)
        # calculate variance for each aggregation function
        sigmas = [
            calibrate_analytic_gaussian_mechanism(
                epsilon_per_agg, delta_per_agg, sensitivity
            )
            for sensitivity in sensitivities
        ]
        # Compute tau and sigma_for_thresholding
        # Split delta_for_thresholding into two parts
        delta_gaussian = delta_per_agg / 2
        delta_infinite = delta_per_agg / 2

        # Calculate sigma_for_thresholding using the analytic Gaussian mechanism
        sigma_for_thresholding = calibrate_analytic_gaussian_mechanism(
            epsilon_per_agg,
            delta_gaussian,
            self.contribution_bound,
        )

        # Calculate tau by inverting the formula in Theorem 4.1
        # of [Wilkins et al., TPDP'22]
        tau = self.min_frequency + sigma_for_thresholding * stats.norm.ppf(
            (1 - delta_infinite) ** (1 / self.contribution_bound)
        )
        logger.debug(
            "Computed: sigmas=%s tau=%s sigma_for_thresholding=%s",
            sigmas,
            tau,
            sigma_for_thresholding,
        )
        return sigmas, tau, sigma_for_thresholding


def generate_dpparams(
    privacy_params: dict[str, float | int],
    agg_columns: list[AggregationColumn],
) -> DPParams:
    """
    Generate dpparams from privacy_params, agg_func and parameters_list.

    Args:
        privacy_params (dict): The privacy parameters. {"EPSILON": float,
          "DELTA": float, "CONTRIBUTION_BOUND": int, "MIN_FREQUENCY": int}
        agg_columns (list[AggregationColumn]): The list of aggregation columns.

    Returns:
        DPParams: The dpparams.
    """

    clipping_thresholds: list[list[tuple[float, float]] | None] = []
    if get_privacy_definition(privacy_params) not in ("DP", "DP_MIN_FREQUENCY"):
        raise UnsupportedQueryError(
            "Unsupported privacy definition (expected DP or DP_MIN_FREQUENCY)",
            context={"privacy_definition": get_privacy_definition(privacy_params)},
            hint="Use DP or DP_MIN_FREQUENCY",
        )

    epsilon = float(privacy_params["EPSILON"])
    delta = float(privacy_params["DELTA"])
    contribution_bound = int(privacy_params["CONTRIBUTION_BOUND"])
    min_frequency = int(privacy_params.get("MIN_FREQUENCY", 1))

    logger.info("Generating DPParams from privacy_params")
    logger.debug("privacy_params=%s", privacy_params)
    logger.debug("agg_columns count=%s", len(agg_columns))

    for agg_column in agg_columns:
        agg = agg_column.aggregation_type
        parameters = agg_column.parameters
        match agg_column.aggregation_type:
            case Aggregation.COUNT | Aggregation.COUNT_DISTINCT:
                if len(parameters) != 0:
                    raise AggregationError(
                        f"Invalid parameter count for {agg.name} (expected 0)",
                        context={"aggregation": agg.name, "parameters": parameters},
                        hint="Remove parameters for COUNT or COUNT_DISTINCT",
                    )
                clipping_thresholds.append(None)
                logger.debug("Aggregation %s: no clipping params", agg.name)
            case (
                Aggregation.SUM
                | Aggregation.AVG
                | Aggregation.STDDEV_SAMP
                | Aggregation.STDDEV_POP
                | Aggregation.VAR_SAMP
                | Aggregation.VAR_POP
            ):
                if len(parameters) != 2:
                    raise AggregationError(
                        f"Invalid parameter count for {agg.name} (expected 2)",
                        context={"aggregation": agg.name, "parameters": parameters},
                        hint="Provide (lower_bound, upper_bound)",
                    )
                clipping_thresholds.append([(parameters[0], parameters[1])])
                logger.debug(
                    "Aggregation %s: clipping=(%s, %s)",
                    agg.name,
                    parameters[0],
                    parameters[1],
                )
            case Aggregation.COVAR_SAMP | Aggregation.COVAR_POP:
                if len(parameters) != 4:
                    raise AggregationError(
                        f"Invalid parameter count for {agg.name} (expected 4)",
                        context={"aggregation": agg.name, "parameters": parameters},
                        hint="Provide (lower_bound_1, upper_bound_1, "
                        "lower_bound_2, upper_bound_2)",
                    )
                clipping_thresholds.append(
                    [
                        (parameters[0], parameters[1]),
                        (parameters[2], parameters[3]),
                    ]
                )
                logger.debug(
                    "Aggregation %s: clipping1=(%s, %s) clipping2=(%s, %s)",
                    agg.name,
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    parameters[3],
                )
            case _:
                raise AggregationError(
                    "Unsupported aggregation type in parameter generation",
                    context={"aggregation": getattr(agg, "name", str(agg))},
                    hint="Extend generate_dpparams to handle this aggregation",
                )

    logger.debug(
        "Constructing DPParams with epsilon=%s delta=%s "
        "min_frequency=%s contribution_bound=%s",
        epsilon,
        delta,
        min_frequency,
        contribution_bound,
    )
    return DPParams(
        contribution_bound=contribution_bound,
        clipping_thresholds=clipping_thresholds,
        min_frequency=min_frequency,
        epsilon=epsilon,
        delta=delta,
    )
