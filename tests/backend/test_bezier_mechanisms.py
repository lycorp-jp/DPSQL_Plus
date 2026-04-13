from __future__ import annotations

import numpy as np
import pytest

from dpsql.auditing import Mechanism, estimate_hockey_stick_divergence
from dpsql.backend import secure_sampling
from dpsql.backend.bezier_mechanisms import (
    private_avg,
    private_covar,
    private_stddev,
    private_var,
)
from dpsql.utils import calibrate_analytic_gaussian_mechanism


def test_private_avg():
    raw_count = 10
    raw_sum = 50
    # No noise
    noisy_avg = private_avg(raw_count, raw_sum, 0, (-10, 10))
    true_avg = raw_sum / raw_count
    assert noisy_avg == pytest.approx(true_avg)


def test_private_var():
    raw_count = 10
    raw_sum = 50
    raw_sum_sq = 300
    true_var_pop = raw_sum_sq / raw_count - (raw_sum / raw_count) ** 2

    # No noise
    assert private_var(
        raw_count, raw_sum, raw_sum_sq, 0, (-10, 10), pop=True
    ) == pytest.approx(true_var_pop)
    assert private_var(
        raw_count, raw_sum, raw_sum_sq, 0, (-10, 10), pop=False
    ) == pytest.approx(true_var_pop * raw_count / (raw_count - 1))


def test_private_stddev():
    raw_count = 10
    raw_sum = 50
    raw_sum_sq = 300
    # No noise
    true_stddev_pop = np.sqrt(raw_sum_sq / raw_count - (raw_sum / raw_count) ** 2)
    true_stddev_samp = np.sqrt(raw_count / (raw_count - 1) * true_stddev_pop**2)
    assert private_stddev(
        raw_count, raw_sum, raw_sum_sq, 0, (-10, 10), pop=True
    ) == pytest.approx(true_stddev_pop)

    assert private_stddev(
        raw_count, raw_sum, raw_sum_sq, 0, (-10, 10), pop=False
    ) == pytest.approx(true_stddev_samp)


def test_private_covar():
    raw_count = 10
    raw_sum_x = 50
    raw_sum_y = 80
    raw_sum_xy = 450
    # No noise
    true_covar_pop = raw_sum_xy / raw_count - (raw_sum_x / raw_count) * (
        raw_sum_y / raw_count
    )
    true_covar_samp = (raw_count / (raw_count - 1)) * true_covar_pop
    assert private_covar(
        raw_count,
        raw_sum_x,
        raw_sum_y,
        raw_sum_xy,
        sigma=0,
        clipping_threshold_x=(-10, 10),
        clipping_threshold_y=(-10, 10),
        pop=True,
    ) == pytest.approx(true_covar_pop)
    assert private_covar(
        raw_count,
        raw_sum_x,
        raw_sum_y,
        raw_sum_xy,
        sigma=0,
        clipping_threshold_x=(-10, 10),
        clipping_threshold_y=(-10, 10),
        pop=False,
    ) == pytest.approx(true_covar_samp)


class AvgMechanism(Mechanism):
    def __init__(self, sigma: float, C: float):
        self.sigma = sigma
        self.C = C

    def randomize(self, value):
        raw_count, raw_sum = value
        return np.array(
            [private_avg(raw_count, raw_sum, self.sigma, (-self.C, self.C))]
        )


def audit_private_avg():
    # Set sampling mode to AUDIT for faster auditing
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.AUDIT

    try:
        # Set parameters
        epsilon = 0.1
        delta = 1e-2
        C = 4
        contribution_bound = 2
        sensitivity = 2 * contribution_bound * C
        sigma = calibrate_analytic_gaussian_mechanism(epsilon, delta, sensitivity)

        # Prepare neighboring data
        raw_count1 = 10000
        raw_sum1 = 50000
        raw_count2 = raw_count1 - contribution_bound
        raw_sum2 = raw_sum1 - contribution_bound * C
        value1 = (raw_count1, raw_sum1)
        value2 = (raw_count2, raw_sum2)

        # Create mechanisms
        mechanism = AvgMechanism(sigma, C)

        # Estimate hockey stick divergence
        num_samples = int(1e6)
        num_bin = 5
        tol = 1e-3

        np.random.seed(0)
        div = estimate_hockey_stick_divergence(
            mechanism, (value1, value2), epsilon, num_bin, num_samples
        )
        assert div <= delta + tol
    finally:
        # Reset sampling mode to SECURE
        secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.SECURE


class StddevMechanism(Mechanism):
    def __init__(self, sigma: float, C: float):
        self.sigma = sigma
        self.C = C

    def randomize(self, value):
        raw_count, raw_sum, raw_squared_sum = value
        return np.array(
            [
                private_stddev(
                    raw_count, raw_sum, raw_squared_sum, self.sigma, (-self.C, self.C)
                )
            ]
        )


def audit_private_stddev():
    # Set sampling mode to AUDIT for faster auditing
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.AUDIT

    try:
        # Set parameters
        epsilon = 0.1
        delta = 1e-2
        C = 7
        contribution_bound = 2
        sensitivity = 2 * contribution_bound * C
        sigma = calibrate_analytic_gaussian_mechanism(epsilon, delta, sensitivity)

        # Prepare neighboring data
        raw_count1 = 10000
        raw_sum1 = 50000
        raw_squared_sum1 = 300000
        raw_count2 = raw_count1 - contribution_bound
        raw_sum2 = raw_sum1 - contribution_bound * C
        raw_squared_sum2 = raw_squared_sum1 - contribution_bound * C * C
        value1 = (raw_count1, raw_sum1, raw_squared_sum1)
        value2 = (raw_count2, raw_sum2, raw_squared_sum2)

        # Create mechanisms
        mechanism = StddevMechanism(sigma, C)

        # Estimate hockey stick divergence
        num_samples = int(1e6)
        num_bin = 5
        tol = 1e-3

        np.random.seed(0)
        div = estimate_hockey_stick_divergence(
            mechanism, (value1, value2), epsilon, num_bin, num_samples
        )
        assert div <= delta + tol
    finally:
        # Reset sampling mode to SECURE
        secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.SECURE


class CovarMechanism(Mechanism):
    def __init__(self, sigma: float, C: float):
        self.sigma = sigma
        self.C = C

    def randomize(self, value):
        raw_count, raw_sum_x, raw_sum_y, raw_sum_xy = value
        return np.array(
            [
                private_covar(
                    raw_count,
                    raw_sum_x,
                    raw_sum_y,
                    raw_sum_xy,
                    self.sigma,
                    (-self.C, self.C),
                    (-self.C, self.C),
                )
            ]
        )


def audit_private_covar():
    # Set sampling mode to AUDIT for faster auditing
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.AUDIT

    try:
        # Set parameters
        epsilon = 0.1
        delta = 1e-2
        C = 5
        contribution_bound = 2
        sensitivity = 2 * contribution_bound * C
        sigma = calibrate_analytic_gaussian_mechanism(epsilon, delta, sensitivity)

        # Prepare neighboring data
        raw_count1 = 10000
        raw_sum_x1 = 50000
        raw_sum_y1 = 80000
        raw_sum_xy1 = 450000
        raw_count2 = raw_count1 - contribution_bound
        raw_sum_x2 = raw_sum_x1 - contribution_bound * C
        raw_sum_y2 = raw_sum_y1 - contribution_bound * C
        raw_sum_xy2 = raw_sum_xy1 - contribution_bound * C * C
        value1 = (raw_count1, raw_sum_x1, raw_sum_y1, raw_sum_xy1)
        value2 = (raw_count2, raw_sum_x2, raw_sum_y2, raw_sum_xy2)

        # Create mechanisms
        mechanism = CovarMechanism(sigma, C)

        # Estimate hockey stick divergence
        num_samples = int(1e6)
        num_bin = 5
        tol = 1e-3

        np.random.seed(0)
        div = estimate_hockey_stick_divergence(
            mechanism, (value1, value2), epsilon, num_bin, num_samples
        )
        assert div <= delta + tol
    finally:
        # Reset sampling mode to SECURE
        secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.SECURE
