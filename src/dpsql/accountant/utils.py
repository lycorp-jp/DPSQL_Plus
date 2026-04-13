import logging

from dp_accounting.pld.common import DifferentialPrivacyParameters
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution,
    from_gaussian_mechanism,
    from_privacy_parameters,
)
from scipy import stats

logger = logging.getLogger(__name__)


# For RenyiAccountant
def calc_alpha_beta_for_tau_thresholding(
    min_frequency: int, tau: float, sigma: float, contribution_bound: int
) -> tuple[float, float]:
    """
    Calculate (alpha, beta) for double tau-thresholding.

    Args:
        min_frequency (int): The threshold for first thresholding before adding noise.
            It satisfies to minimum frequency rule.
        tau (float): The threshold for second thresholding after adding noise.
        sigma (float): The standard deviation for the Gaussian mechanism
            before the second thresholding.
        contribution_bound (int): The contribution bound.

    Returns:
        (alpha, beta) (tuple[float, float]):
            The parameters of beta-approximate alpha-zCDP.
    """
    logger.debug(
        "calc_alpha_beta_for_tau_thresholding: min_frequency=%s"
        " tau=%s sigma=%s contribution_bound=%s",
        min_frequency,
        tau,
        sigma,
        contribution_bound,
    )
    # Calculate based on Theorem 4.1 of [Wilkins et al., TPDP'22]
    alpha = (contribution_bound**2) / (2 * sigma**2)
    beta = 1 - (stats.norm.cdf((tau - min_frequency) / sigma)) ** contribution_bound
    logger.debug("alpha/beta computed: alpha=%s beta=%s", alpha, beta)
    return alpha, beta


# For PLDAccountant
def calc_pld_for_tau_thresholding(
    min_frequency: int,
    tau: float,
    sigma: float,
    contribution_bound: int,
    discretization_interval: float = 1e-4,
) -> PrivacyLossDistribution:
    """
    Calculate the privacy loss distribution for double tau-thresholding.

    Args:
        min_frequency (int): The threshold for first thresholding before adding noise.
            It satisfies minimum frequency rule.
        tau (float): The threshold for second thresholding after adding noise.
        sigma (float): The standard deviation for the Gaussian mechanism
            before the second thresholding.
        contribution_bound (int): The contribution bound.
        discretization_interval (float): The discretization interval for the
            privacy loss distribution.

    Returns:
        pld (PrivacyLossDistribution): The privacy loss distribution.
    """
    logger.debug(
        "calc_pld_for_tau_thresholding: min_frequency=%s tau=%s sigma=%s"
        " contribution_bound=%s discretization_interval=%s",
        min_frequency,
        tau,
        sigma,
        contribution_bound,
        discretization_interval,
    )
    # Calculate based on Theorem 4.1 of [Wilkins et al., TPDP'22]
    pld_gaussian = from_gaussian_mechanism(
        sigma,
        contribution_bound,
        value_discretization_interval=discretization_interval,
    )
    logger.debug("PLD (gaussian) created")
    delta_infinite = (
        1 - (stats.norm.cdf((tau - min_frequency) / sigma)) ** contribution_bound
    )
    logger.debug("delta_infinite=%s", delta_infinite)
    pld_infinite = from_privacy_parameters(
        DifferentialPrivacyParameters(0, delta_infinite),
        value_discretization_interval=discretization_interval,
    )
    logger.debug("PLD (infinite) created")
    pld = pld_gaussian.compose(pld_infinite)
    logger.debug("PLD composed for tau-thresholding")
    return pld
