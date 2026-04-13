import math

import pytest
from dp_accounting.pld.common import DifferentialPrivacyParameters
from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters

from dpsql.accountant import PLDAccountant
from dpsql.accountant.utils import calc_pld_for_tau_thresholding
from dpsql.aggregation import Aggregation
from dpsql.dp_params import DPParams
from dpsql.errors import InvalidPrivacyParametersError


def assert_pld_equal(pld1, pld2):
    for prob, prob_expected in zip(
        pld1._pmf_add._probs, pld2._pmf_add._probs, strict=False
    ):
        assert prob == pytest.approx(prob_expected)


def test_compute_pld():
    accountant = PLDAccountant(1.0, 0.1)
    sensitivities = [3.0]
    params = DPParams(
        contribution_bound=2.0,
        tau=3.0,
        sigma_for_thresholding=1.0,
        sigmas=[3.0],
        clipping_thresholds=[None],
    )
    delta_0 = 0.5

    pld = accountant._compute_pld(sensitivities, params)
    epsilon = pld.get_epsilon_for_delta(delta_0)

    # Basic composition
    pld_expected = calc_pld_for_tau_thresholding(
        params.min_frequency,
        params.tau,
        params.sigma_for_thresholding,
        params.contribution_bound,
    )
    epsilon_expected = pld_expected.get_epsilon_for_delta(delta_0 / 2)
    epsilon_expected += sum(
        s / sigma * math.sqrt(2 * math.log(1.25 / (delta_0 / 2)))
        for s, sigma in zip(sensitivities, params.sigmas, strict=False)
    )

    # Numerical composition yields a better epsilon than basic composition
    assert epsilon <= epsilon_expected


@pytest.mark.parametrize(
    "aggregation,clipping_thresholds,sensitivities",
    [
        (Aggregation.COUNT, [None], [1.0]),
        (Aggregation.COUNT_DISTINCT, [None], [1.0]),
        (Aggregation.SUM, [[(-10.0, 10.0)]], [10.0]),
        (Aggregation.AVG, [[(-10.0, 10.0)]], [20.0]),
        (Aggregation.STDDEV_SAMP, [[(-10.0, 10.0)]], [20.0]),
        (Aggregation.STDDEV_POP, [[(-10.0, 10.0)]], [20.0]),
        (Aggregation.VAR_SAMP, [[(-10.0, 10.0)]], [20.0]),
        (Aggregation.VAR_POP, [[(-10.0, 10.0)]], [20.0]),
        (Aggregation.COVAR_SAMP, [[(-10.0, 10.0), (-5.0, 5.0)]], [20.0]),
        (Aggregation.COVAR_POP, [[(-10.0, 10.0), (-5.0, 5.0)]], [20.0]),
    ],
)
def test_compute_pld_aggregations(aggregation, clipping_thresholds, sensitivities):
    accountant = PLDAccountant(1.0, 0.1)
    base_kwargs = {
        "contribution_bound": 1.0,
        "sigma_for_thresholding": 1.0,
        "tau": 1.0,
        "sigmas": [1.0],
        "clipping_thresholds": clipping_thresholds,
    }
    params = DPParams(**base_kwargs)

    pld = accountant.compute_pld([aggregation], params)
    pld_expected = accountant._compute_pld(sensitivities, params)
    assert_pld_equal(pld, pld_expected)


def test_calculate_min_epsilon():
    epsilon = 0.1
    delta = 0.01
    accountant = PLDAccountant(1.0, delta)
    pld = from_privacy_parameters(
        DifferentialPrivacyParameters(epsilon, delta),
    )

    min_epsilon = accountant.calculate_min_epsilon(pld)

    assert min_epsilon == pytest.approx(min_epsilon, epsilon)


def test_check_budget():
    accountant = PLDAccountant(1.0, 0.1)
    params = DPParams(
        contribution_bound=2.0,
        tau=3.0,
        sigma_for_thresholding=1.0,
        sigmas=[3.0],
        clipping_thresholds=[None],
    )
    delta = 0.5
    pld = accountant.compute_pld([Aggregation.COUNT], params)
    epsilon = pld.get_epsilon_for_delta(delta)

    accountant = PLDAccountant(epsilon, delta)
    # Budget is sufficient for the first query
    assert accountant.check_budget([Aggregation.COUNT], params)

    accountant.update_budget([Aggregation.COUNT], params)
    # Budget is not sufficient for the second query
    assert not accountant.check_budget([Aggregation.COUNT], params)


def test_update_budget():
    accountant = PLDAccountant(1.0, 0.5)
    params = DPParams(
        contribution_bound=1.0,
        sigma_for_thresholding=1.0,
        tau=1.0,
        sigmas=[1.0],
        clipping_thresholds=[None],
    )
    pld = accountant.compute_pld([Aggregation.COUNT], params)

    accountant.update_budget([Aggregation.COUNT], params)
    assert_pld_equal(accountant.pld, pld)

    accountant.update_budget([Aggregation.COUNT], params)
    assert_pld_equal(accountant.pld, pld.compose(pld))

    # Test updating with infinite budget
    accountant = PLDAccountant(1.0, 1.0)
    for _ in range(10):
        accountant.update_budget([Aggregation.COUNT], params)
        assert accountant.check_budget([Aggregation.COUNT], params)


def test_remaining_queries(mocker):
    total_epsilon = 1.0 + 1e-5  # To avoid floating point precision issues
    total_delta = 0.1
    accountant = PLDAccountant(total_epsilon, total_delta)

    mocker.patch.object(
        accountant.basic_accountant, "remaining_queries", return_value=0
    )

    # Test with epsilon being the limiting factor
    # PLDAccountant may have more remaining queries than BasicAccountant
    # if epsilon is the limiting factor
    remaining = accountant.remaining_queries(0.2, 0.01)
    assert remaining >= 5  # min(1.0/0.2, 0.1/0.01) = min(5, 10) = 5
    assert remaining < 10  # Not too many

    # Test with delta being the limiting factor
    remaining = accountant.remaining_queries(0.1, 0.05)
    assert remaining == 2  # min(1.0/0.1, 0.1/0.05) = min(10, 2) = 2

    # Test after some queries have been executed
    params = DPParams(
        epsilon=total_epsilon / 2,
        delta=total_delta / 2,
        clipping_thresholds=[None],
        contribution_bound=1.0,
    )
    accountant.update_budget([Aggregation.COUNT], params)
    remaining_after_update = accountant.remaining_queries(0.1, 0.05)
    assert remaining_after_update < remaining

    # Test with invalid parameters
    with pytest.raises(InvalidPrivacyParametersError):
        accountant.remaining_queries(-0.1, 0.01)

    with pytest.raises(InvalidPrivacyParametersError):
        accountant.remaining_queries(0.1, -0.01)

    # Test with infinite budget
    accountant = PLDAccountant(1.0, 1.0)
    remaining = accountant.remaining_queries(0.1, 1.0)
    assert remaining == accountant.MAX_REMAINING_QUERIES
