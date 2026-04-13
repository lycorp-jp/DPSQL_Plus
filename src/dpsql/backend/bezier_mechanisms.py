import logging

import numpy as np

from ..errors import AggregationError
from .secure_sampling import secure_gauss

logger = logging.getLogger(__name__)


def private_avg(
    raw_count: float,
    raw_sum: float,
    sigma: float,
    clipping_threshold: tuple[float, float],
) -> float:
    """
    Compute noisy average by transformed noise addition
    (Algorithm3 in https://openreview.net/pdf?id=cwIhvoTzuK)

    Args:
        raw_count (float): The raw count.
        raw_sum (float): The raw sum.
        sigma (float): The noise scale.
        clipping_threshold (tuple[float, float]):
            The clipping threshold as (lower_bound, upper_bound).

    Returns:
        float: The noisy average.
    """
    lower_bound, upper_bound = clipping_threshold
    logger.debug(
        "private_avg: raw_count=%s raw_sum=%s sigma=%s clipping=%s",
        raw_count,
        raw_sum,
        sigma,
        clipping_threshold,
    )
    if lower_bound >= upper_bound:
        logger.error(
            "private_avg invalid clipping: lower=%s upper=%s",
            lower_bound,
            upper_bound,
        )
        raise AggregationError(
            "Invalid clipping threshold (lower bound exceeds upper bound)",
            context={"lower_bound": lower_bound, "upper_bound": upper_bound},
            hint="Ensure lower_bound < upper_bound",
        )
    # Apply linear transformation to (sum, count) to get (s1, s2)
    # input x_i \in [lower_bound, upper_bound] is normalized to
    # y_i = (x_i - lower_bound) / (upper_bound - lower_bound) in [0, 1]
    # s1 = sum(y_i), s2 = sum(1 - y_i)
    s1 = (raw_sum - lower_bound * raw_count) / (
        upper_bound - lower_bound
    )  # normalized sum
    s2 = raw_count - s1
    # Add noise to (s1, s2)
    s1 = secure_gauss(s1, sigma / (upper_bound - lower_bound))
    s2 = secure_gauss(s2, sigma / (upper_bound - lower_bound))
    # Get noisy count and normalized avg
    noisy_count = s1 + s2
    noisy_normalized_avg = np.clip(s1 / noisy_count, 0, 1)
    # Apply inverse transformation to get noisy average
    noisy_avg = (upper_bound - lower_bound) * noisy_normalized_avg + lower_bound
    logger.debug(
        "private_avg: noisy_count=%s noisy_normalized_avg=%s noisy_avg=%s",
        noisy_count,
        noisy_normalized_avg,
        noisy_avg,
    )
    return noisy_avg


def private_var(
    raw_count: float,
    raw_sum: float,
    raw_squared_sum: float,
    sigma: float,
    clipping_threshold: tuple[float, float],
    pop: bool = False,
) -> float:
    """
    Compute noisy variance based on Bezier mechanism (https://arxiv.org/abs/2509.04919).

    Args:
        raw_count (float): The raw count.
        raw_sum (float): The raw sum.
        raw_squared_sum (float): The raw squared sum.
        sigma (float): The noise scale.
        clipping_threshold (tuple[float, float]):
            The clipping threshold as (lower_bound, upper_bound).
        pop (bool): Whether to compute population variance or sample variance.

    Returns:
        float: The noisy variance.
    """
    lower_bound, upper_bound = clipping_threshold
    logger.debug(
        "private_var: raw_count=%s raw_sum=%s"
        " raw_squared_sum=%s sigma=%s clipping=%s pop=%s",
        raw_count,
        raw_sum,
        raw_squared_sum,
        sigma,
        clipping_threshold,
        pop,
    )
    if lower_bound >= upper_bound:
        logger.error(
            "private_var invalid clipping: lower=%s upper=%s",
            lower_bound,
            upper_bound,
        )
        raise AggregationError(
            "Invalid clipping threshold (lower bound exceeds upper bound)",
            context={"lower_bound": lower_bound, "upper_bound": upper_bound},
            hint="Ensure lower_bound < upper_bound",
        )
    dx = upper_bound - lower_bound

    normalized_raw_sum = (raw_sum - lower_bound * raw_count) / dx  # normalized sum
    normalized_raw_squared_sum = (
        raw_squared_sum - 2 * lower_bound * raw_sum + raw_count * lower_bound**2
    ) / dx**2  # normalized squared sum

    b0 = raw_count - 2 * normalized_raw_sum + normalized_raw_squared_sum
    b1 = 2 * (normalized_raw_sum - normalized_raw_squared_sum)
    b2 = normalized_raw_squared_sum
    # Add noise to (b0, b1, b2)
    b0 = secure_gauss(b0, sigma / dx)
    b1 = secure_gauss(b1, sigma / dx)
    b2 = secure_gauss(b2, sigma / dx)
    # Get noisy count and normalized avg
    noisy_count = b0 + b1 + b2
    noisy_normalized_sum = b1 / 2 + b2
    noisy_normalized_squared_sum = b2

    # Maximum theoretical variance for normalized data in [0,1] is 1/4
    noisy_normalized_var = np.clip(
        (noisy_normalized_squared_sum / noisy_count)
        - (noisy_normalized_sum / noisy_count) ** 2,
        0,
        1 / 4,
    )
    if not pop:
        if noisy_count <= 1:
            noisy_normalized_var = 0.0
        else:
            noisy_normalized_var *= noisy_count / (noisy_count - 1)
    noisy_var = noisy_normalized_var * (upper_bound - lower_bound) ** 2
    logger.debug(
        "private_var: noisy_count=%s noisy_norm_var=%s noisy_var=%s",
        noisy_count,
        noisy_normalized_var,
        noisy_var,
    )
    return noisy_var


def private_stddev(
    raw_count: float,
    raw_sum: float,
    raw_squared_sum: float,
    sigma: float,
    clipping_threshold: tuple[float, float],
    pop: bool = False,
) -> float:
    """
    Compute noisy standard deviation from noisy variance.

    Args:
        raw_count (float): The raw count.
        raw_sum (float): The raw sum.
        raw_squared_sum (float): The raw squared sum.
        sigma (float): The noise scale.
        clipping_threshold (tuple[float, float]):
            The clipping threshold as (lower_bound, upper_bound).
        pop (bool): Whether to compute population stddev or sample stddev.
    Returns:
        float: The noisy standard deviation.
    """
    noisy_var = private_var(
        raw_count, raw_sum, raw_squared_sum, sigma, clipping_threshold, pop
    )
    logger.debug("private_stddev: noisy_var=%s", noisy_var)
    return np.sqrt(noisy_var)


def private_covar(
    raw_count: float,
    raw_sum_x: float,
    raw_sum_y: float,
    raw_sum_xy: float,
    sigma: float,
    clipping_threshold_x: tuple[float, float],
    clipping_threshold_y: tuple[float, float],
    pop: bool = False,
) -> float:
    """
    Compute noisy covariance based on Bezier mechanism (https://arxiv.org/abs/2509.04919).

    Args:
        raw_count (float): The raw count.
        raw_sum_x (float): The raw sum of x.
        raw_sum_y (float): The raw sum of y.
        raw_sum_xy (float): The raw sum of x*y.
        sigma (float): The noise scale.
        clipping_threshold_x (tuple[float, float]):
            The clipping threshold for x as (lower_bound, upper_bound).
        clipping_threshold_y (tuple[float, float]):
            The clipping threshold for y as (lower_bound, upper_bound).
        pop (bool): Whether to compute population covariance or sample covariance.
    Returns:
        float: The noisy covariance.
    """
    lower_bound_x, upper_bound_x = clipping_threshold_x
    lower_bound_y, upper_bound_y = clipping_threshold_y
    logger.debug(
        "private_covar: raw_count=%s raw_sum_x=%s raw_sum_y=%s"
        " raw_sum_xy=%s sigma=%s clip_x=%s clip_y=%s pop=%s",
        raw_count,
        raw_sum_x,
        raw_sum_y,
        raw_sum_xy,
        sigma,
        clipping_threshold_x,
        clipping_threshold_y,
        pop,
    )
    if lower_bound_x >= upper_bound_x:
        logger.error(
            "private_covar invalid clipping x: lower=%s upper=%s",
            lower_bound_x,
            upper_bound_x,
        )
        raise AggregationError(
            "Invalid clipping threshold for x (lower bound exceeds upper bound)",
            context={"lower_bound_x": lower_bound_x, "upper_bound_x": upper_bound_x},
            hint="Ensure lower_bound_x < upper_bound_x",
        )
    if lower_bound_y >= upper_bound_y:
        logger.error(
            "private_covar invalid clipping y: lower=%s upper=%s",
            lower_bound_y,
            upper_bound_y,
        )
        raise AggregationError(
            "Invalid clipping threshold for y (lower bound exceeds upper bound)",
            context={"lower_bound_y": lower_bound_y, "upper_bound_y": upper_bound_y},
            hint="Ensure lower_bound_y < upper_bound_y",
        )

    dx = upper_bound_x - lower_bound_x
    dy = upper_bound_y - lower_bound_y

    normalized_raw_sum_x = (
        raw_sum_x - lower_bound_x * raw_count
    ) / dx  # normalized sum x
    normalized_raw_sum_y = (
        raw_sum_y - lower_bound_y * raw_count
    ) / dy  # normalized sum y
    normalized_raw_sum_xy = (
        raw_sum_xy
        - lower_bound_y * raw_sum_x
        - lower_bound_x * raw_sum_y
        + raw_count * lower_bound_x * lower_bound_y
    ) / (dx * dy)  # normalized sum xy

    b0 = raw_count - normalized_raw_sum_x - normalized_raw_sum_y + normalized_raw_sum_xy
    b1 = normalized_raw_sum_y - normalized_raw_sum_xy
    b2 = normalized_raw_sum_x - normalized_raw_sum_xy
    b3 = normalized_raw_sum_xy
    # Add noise to (b0, b1, b2, b3)
    b0 = secure_gauss(b0, sigma / dx)
    b1 = secure_gauss(b1, sigma / dx)
    b2 = secure_gauss(b2, sigma / dx)
    b3 = secure_gauss(b3, sigma / dx)
    # Get noisy count and normalized avg
    noisy_count = b0 + b1 + b2 + b3
    noisy_normalized_sum_x = b2 + b3
    noisy_normalized_sum_y = b1 + b3
    noisy_normalized_sum_xy = b3

    noisy_normalized_covar = np.clip(
        (noisy_normalized_sum_xy / noisy_count)
        - (noisy_normalized_sum_x / noisy_count)
        * (noisy_normalized_sum_y / noisy_count),
        -0.25,
        0.25,
    )
    if not pop:
        if noisy_count <= 1:
            noisy_normalized_covar = 0.0
        else:
            noisy_normalized_covar *= noisy_count / (noisy_count - 1)
    noisy_covar = noisy_normalized_covar * dx * dy
    logger.debug(
        "private_covar: noisy_count=%s noisy_norm_covar=%s noisy_covar=%s",
        noisy_count,
        noisy_normalized_covar,
        noisy_covar,
    )
    return noisy_covar
