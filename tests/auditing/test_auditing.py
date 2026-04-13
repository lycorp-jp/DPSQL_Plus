from math import log

import numpy as np
import pytest

from dpsql.auditing import (
    Mechanism,
    calculate_hockey_stick_divergence,
    estimate_hockey_stick_divergence,
)
from dpsql.utils import calibrate_analytic_gaussian_mechanism


def test_calculate_hockey_stick_divergence():
    dist1 = np.array([0.1, 0.2, 0.3, 0.4])
    dist2 = np.array([0.2, 0.3, 0.4, 0.1])
    epsilon = log(2)
    div = calculate_hockey_stick_divergence(dist1, dist2, epsilon)
    assert div == pytest.approx(0.2)


def test_estimate_hockey_stick_divergence():
    class GaussianMechanism(Mechanism):
        def __init__(self, scale):
            self.scale = scale

        def randomize(self, value):
            return np.random.normal(value, self.scale, 1)

    epsilon = 0.1
    delta = 1e-2
    num_samples = int(1e6)
    tol = 1e-3

    np.random.seed(0)
    # Correctly calibrated mechanisms
    sigma = calibrate_analytic_gaussian_mechanism(epsilon, delta, 1)
    mechanism = GaussianMechanism(sigma)
    div = estimate_hockey_stick_divergence(mechanism, (0, 1), epsilon, 10, num_samples)
    assert div == pytest.approx(delta, abs=tol)

    # Incorrectly calibrated mechanisms
    sigma = 0.9 * calibrate_analytic_gaussian_mechanism(epsilon, delta, 1)
    mechanism = GaussianMechanism(sigma)
    div = estimate_hockey_stick_divergence(mechanism, (0, 1), epsilon, 10, num_samples)
    assert div > delta + tol
