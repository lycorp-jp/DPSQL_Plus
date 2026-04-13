import logging
from collections.abc import Callable
from math import erf, exp, sqrt

from dpsql.errors import AggregationError

logger = logging.getLogger(__name__)


def phi(t: float) -> float:
    logger.debug("phi: t=%s", t)
    return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))


def compute_case_a(epsilon: float, s: float) -> float:
    logger.debug("compute_case_a: epsilon=%s s=%s", epsilon, s)
    return phi(sqrt(epsilon * s)) - exp(epsilon) * phi(-sqrt(epsilon * (s + 2.0)))


def compute_case_b(epsilon: float, s: float) -> float:
    logger.debug("compute_case_b: epsilon=%s s=%s", epsilon, s)
    return phi(-sqrt(epsilon * s)) - exp(epsilon) * phi(-sqrt(epsilon * (s + 2.0)))


def doubling_trick(
    predicate_stop: Callable[[float], bool], s_inf: float, s_sup: float
) -> tuple[float, float]:
    logger.debug("doubling_trick: start s_inf=%s s_sup=%s", s_inf, s_sup)
    iter_count = 0
    while not predicate_stop(s_sup):
        s_inf = s_sup
        s_sup = 2.0 * s_inf
        iter_count += 1
    logger.debug(
        "doubling_trick: end s_inf=%s s_sup=%s iterations=%s", s_inf, s_sup, iter_count
    )
    return s_inf, s_sup


def binary_search(
    predicate_stop: Callable[[float], bool],
    predicate_left: Callable[[float], bool],
    s_inf: float,
    s_sup: float,
) -> float:
    logger.debug("binary_search: start s_inf=%s s_sup=%s", s_inf, s_sup)
    s_mid = s_inf + (s_sup - s_inf) / 2.0
    iter_count = 0
    while not predicate_stop(s_mid):
        if predicate_left(s_mid):
            s_sup = s_mid
        else:
            s_inf = s_mid
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        iter_count += 1
    logger.debug("binary_search: end s_mid=%s iterations=%s", s_mid, iter_count)
    return s_mid


def calibrate_analytic_gaussian_mechanism(
    epsilon: float, delta: float, global_sensitivity: float, tol: float = 1.0e-12
) -> float:
    """
    Calibrate a Gaussian perturbation for differential privacy
    using the analytic Gaussian mechanism of [Balle and Wang, ICML'18].

    Args:
        epsilon (float): target epsilon (epsilon > 0)
        delta (float): target delta (0 < delta < 1)
        global_sensitivity (float): upper bound on L2 global sensitivity (>= 0)
        tol (float): error tolerance for binary search (tol > 0)

    Returns:
        float: standard deviation of Gaussian noise needed
        to achieve (epsilon, delta)-DP under global_sensitivity
    """
    logger.debug(
        "Calibrate analytic Gaussian: eps=%s delta=%s gs=%s tol=%s",
        epsilon,
        delta,
        global_sensitivity,
        tol,
    )

    # Threshold delta when s=0
    delta_thr = compute_case_a(epsilon, 0.0)
    logger.debug("delta_thr (s=0)=%s", delta_thr)

    # Directly compute alpha if delta == delta_thr
    if abs(delta - delta_thr) < tol:
        logger.debug("delta close to delta_thr -> direct alpha=1.0")
        alpha = 1.0
    else:
        if delta > delta_thr:
            logger.debug("case A (delta > delta_thr)")
            predicate_stop_dt: Callable[[float], bool] = (
                lambda s: compute_case_a(epsilon, s) >= delta
            )
            compute_delta: Callable[[float], float] = lambda s: compute_case_a(
                epsilon, s
            )
            predicate_left_bs: Callable[[float], bool] = (
                lambda s: compute_delta(s) > delta
            )
            compute_alpha: Callable[[float], float] = lambda s: sqrt(
                1.0 + s / 2.0
            ) - sqrt(s / 2.0)
        else:
            logger.debug("case B (delta <= delta_thr)")
            predicate_stop_dt = lambda s: compute_case_b(epsilon, s) <= delta
            compute_delta = lambda s: compute_case_b(epsilon, s)
            predicate_left_bs = lambda s: compute_delta(s) < delta
            compute_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        # Binary search within the expanded range found by doubling
        s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
        logger.debug("search range: s_inf=%s s_sup=%s", s_inf, s_sup)
        predicate_stop_bs: Callable[[float], bool] = (
            lambda s: abs(compute_delta(s) - delta) <= tol
        )
        s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
        alpha = compute_alpha(s_final)
        logger.debug("alpha computed: s_final=%s alpha=%s", s_final, alpha)

    sigma = alpha * global_sensitivity / sqrt(2.0 * epsilon)
    logger.debug("Calibrated sigma=%s", sigma)
    return sigma


def safely_get_threshold(
    clipping_threshold: list[tuple[float, float]] | None,
    index: int,
    agg_name: str,
) -> tuple[float, float]:
    """Safely get clipping thresholds for aggregation columns.

    Args:
        clipping_threshold (list[tuple[float, float]] | None):
            The provided clipping thresholds.
        index (int): The index of the aggregation column.
        agg_name (str): The name of the aggregation type.

    Returns:
        tuple[float, float]: The validated clipping thresholds.
    """
    logger.debug("safely_get_threshold: agg=%s index=%s", agg_name, index)
    if clipping_threshold is None or len(clipping_threshold) <= index:
        logger.error("Missing clipping thresholds: agg=%s index=%s", agg_name, index)
        raise AggregationError(
            f"Clipping thresholds must be provided for {agg_name} aggregation",
            context={"aggregation": agg_name},
            hint="Provide clipping thresholds for each aggregation column",
        )
    result = clipping_threshold[index]
    logger.debug("safely_get_threshold -> %s", result)
    return result
