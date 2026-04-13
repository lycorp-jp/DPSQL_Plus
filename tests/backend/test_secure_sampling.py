import random
from math import sqrt

from scipy.stats import kstest

import dpsql.backend.secure_sampling as secure_sampling


def test_secure_gauss():
    input = 0.0
    sigma = 2.0
    num_samples = 10000
    data = [secure_sampling.secure_gauss(input, sigma) for _ in range(num_samples)]

    # Kolmogorov-Smirnov test for normal distribution
    stat, p_value = kstest(data, "norm", args=(input, sigma))
    assert p_value > 0.05


def test_secure_gauss_against_precision_based_attack():
    """
    Test against Precision-based attack (https://arxiv.org/abs/2207.13793).
    """
    num_samples = 1000

    def gauss(sigma: float, n: int = 1) -> float:
        """
        Gaussian sampling from multiple random numbers (https://arxiv.org/pdf/2107.10138).
        """
        return sum(random.gauss(0, sigma) for _ in range(2 * n)) / sqrt(2 * n)

    # When adding noise to 1 without any rounding,
    # all possible outputs are multiples of 2^-53
    data_a = [1.0 + gauss(1.0) for _ in range(num_samples)]
    data_b = [0.0 + gauss(1.0) for _ in range(num_samples)]
    assert sum([(o * (2.0**53)).is_integer() for o in data_a]) == num_samples
    assert sum([(o * (2.0**53)).is_integer() for o in data_b]) < num_samples

    # When adding noise to 1 with secure sampling,
    # not all possible outputs are multiples of 2^-53
    data_a = [secure_sampling.secure_gauss(1.0, 1.0) for _ in range(num_samples)]
    data_b = [secure_sampling.secure_gauss(0.0, 1.0) for _ in range(num_samples)]
    assert sum([(o * (2.0**53)).is_integer() for o in data_a]) < num_samples
    assert sum([(o * (2.0**53)).is_integer() for o in data_b]) < num_samples
