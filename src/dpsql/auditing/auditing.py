import logging
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Mechanism(metaclass=ABCMeta):
    """
    The abstract class for a randomized mechanism.
    """

    @abstractmethod
    def __init__(self):
        logger.debug("Initializing abstract Mechanism")
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def randomize(self, value: Any) -> np.ndarray:
        """
        Randomize the input value.

        Args:
            value (Any): The input value.

        Returns:
            np.ndarray: The generated sample.
        """
        logger.debug("Mechanism.randomize called with value=%s", value)
        raise NotImplementedError("Subclasses must implement this method")


def calculate_hockey_stick_divergence(
    dist1: np.ndarray, dist2: np.ndarray, epsilon: float
) -> float:
    """
    Calculate hockey stick divergence between two distributions.

    Args:
        dist1 (np.ndarray): The first distribution.
        dist2 (np.ndarray): The second distribution.
        epsilon (float): The privacy parameter.

    Returns:
        float: The hockey stick divergence between the two distributions.
    """
    logger.debug(
        "Calculating hockey stick divergence: epsilon=%s,"
        " dist1.shape=%s, dist2.shape=%s",
        epsilon,
        getattr(dist1, "shape", None),
        getattr(dist2, "shape", None),
    )
    d1 = (dist1 - dist2 * np.exp(epsilon)).clip(min=0).sum()
    d2 = (dist2 - dist1 * np.exp(epsilon)).clip(min=0).sum()
    result = max(d1, d2)
    logger.debug("Hockey stick divergence result: %s (d1=%s, d2=%s)", result, d1, d2)
    return result


def estimate_hockey_stick_divergence(
    mechanism: Mechanism,
    input_values: tuple[Any, Any],
    epsilon: float,
    num_bin: int,
    num_samples: int,
) -> float:
    """ "
    Estimate the hockey stick divergence between two mechanisms
    using Algorithm 3 in https://arxiv.org/abs/2307.05608

    Args:
        mechanism (Mechanism): The mechanism to be audited.
        input_values (tuple[Any, Any]): The input values to be compared.
        epsilon (float): The privacy parameter.
        num_bin (int): The number of bins for each dimension.
        num_samples (int): The number of samples.

    Returns:
        float: The estimated hockey stick divergence between the two mechanisms.
    """
    logger.info("Estimating hockey stick divergence")
    logger.debug(
        "Params: epsilon=%s, num_bin=%s, num_samples=%s, input_values_types=%s",
        epsilon,
        num_bin,
        num_samples,
        tuple(type(v).__name__ for v in input_values),
    )
    value1, value2 = input_values
    samples1 = np.array([mechanism.randomize(value1) for _ in range(num_samples)])
    samples2 = np.array([mechanism.randomize(value2) for _ in range(num_samples)])
    d = samples1.shape[1]
    logger.debug(
        "Generated samples: shape1=%s shape2=%s dims=%s",
        samples1.shape,
        samples2.shape,
        d,
    )

    ranges = []
    for i in range(d):
        lower = min(samples1[:, i].min(), samples2[:, i].min())
        upper = max(samples1[:, i].max(), samples2[:, i].max())
        ranges.append((lower, upper))
    logger.debug("Histogram ranges: %s", ranges)

    hist1, _ = np.histogramdd(samples1, range=ranges, bins=num_bin)
    hist2, _ = np.histogramdd(samples2, range=ranges, bins=num_bin)
    dist1 = hist1 / num_samples
    dist2 = hist2 / num_samples
    logger.debug("Histograms computed. Normalized distributions ready.")

    result = calculate_hockey_stick_divergence(dist1, dist2, epsilon)
    logger.info("Estimated hockey stick divergence: %s", result)
    return result
