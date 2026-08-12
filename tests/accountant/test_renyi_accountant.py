import math

import pytest

from dpsql.accountant import PLDAccountant, RenyiAccountant
from dpsql.aggregation import Aggregation
from dpsql.dp_params import DPParams
from dpsql.errors import InvalidPrivacyParametersError


def test_calculate_budget():
    accountant = RenyiAccountant(1.0, 0.1)
    sensitivities = [3.0]
    params = DPParams(
        contribution_bound=2,
        tau=1.0,
        sigma_for_thresholding=1.0,
        sigmas=[3.0],
        clipping_thresholds=[None],
    )

    alpha, beta = accountant._calculate_budget(sensitivities, params)
    alpha_expected, beta_expected = 1.5, 0.75

    assert alpha == pytest.approx(alpha_expected)
    assert beta == pytest.approx(beta_expected)


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
def test_calculate_budget_aggregations(aggregation, clipping_thresholds, sensitivities):
    accountant = RenyiAccountant(1.0, 0.1)
    base_kwargs = {
        "contribution_bound": 1,
        "sigma_for_thresholding": 1.0,
        "tau": 1.0,
        "sigmas": [1.0],
        "clipping_thresholds": clipping_thresholds,
    }
    params = DPParams(**base_kwargs)

    alpha, beta = accountant.calculate_budget([aggregation], params)
    alpha_expected, beta_expected = accountant._calculate_budget(sensitivities, params)

    assert alpha == pytest.approx(alpha_expected)
    assert beta == pytest.approx(beta_expected)


def test_calculate_min_epsilon():
    accountant = RenyiAccountant(1.0, 0.1)
    alpha = 0.5
    beta = 0.05

    min_epsilon = accountant.calculate_min_epsilon(alpha, beta)
    min_epsilon_expected = 0.5 + math.sqrt(2 * math.log(20))

    assert min_epsilon == pytest.approx(min_epsilon_expected)


def test_calculate_min_epsilon_infinity():
    accountant = RenyiAccountant(1.0, 0.1)
    alpha = 0.5
    beta = 0.2

    min_epsilon = accountant.calculate_min_epsilon(alpha, beta)

    assert min_epsilon == float("inf")


def test_check_budget():
    accountant = RenyiAccountant(5, 0.5)
    params = DPParams(
        contribution_bound=2,
        tau=3.0,
        sigma_for_thresholding=1.0,
        sigmas=[3.0],
        clipping_thresholds=[None],
    )

    # Budget is sufficient for the first query
    assert accountant.check_budget([Aggregation.COUNT], params)

    accountant.update_budget([Aggregation.COUNT], params)

    # Budget is not sufficient for the second query
    assert not accountant.check_budget([Aggregation.COUNT], params)


def test_update_budget():
    accountant = RenyiAccountant(7, 0.5)
    params = DPParams(
        contribution_bound=2,
        tau=3.0,
        sigma_for_thresholding=1.0,
        sigmas=[3.0],
        clipping_thresholds=[None],
    )
    alpha, beta = accountant._calculate_budget([params.contribution_bound], params)

    accountant.update_budget([Aggregation.COUNT], params)
    accountant.update_budget([Aggregation.COUNT], params)

    assert accountant.alpha == pytest.approx(
        alpha * 2
    ) and accountant.beta == pytest.approx(beta * 2)

    # Test updating with infinite budget
    accountant = RenyiAccountant(1.0, 1.0)
    for _ in range(10):
        accountant.update_budget([Aggregation.COUNT], params)
        assert accountant.check_budget([Aggregation.COUNT], params)


def test_remaining_queries(mocker):
    total_epsilon = 1.0 + 1e-5  # To avoid floating point precision issues
    total_delta = 0.1
    accountant = RenyiAccountant(total_epsilon, total_delta)
    pld_accountant = PLDAccountant(total_epsilon, total_delta)

    # Mock BasicAccountant to test RenyiAccountant
    mocker.patch.object(
        accountant.basic_accountant, "remaining_queries", return_value=0
    )

    # RenyiAccountant is expected to have fewer remaining queries than PLDAccountant

    # Test with epsilon being the limiting factor
    remaining = accountant.remaining_queries(0.2, 0.01)
    assert remaining <= pld_accountant.remaining_queries(0.2, 0.01)

    # Test with delta being the limiting factor
    remaining = accountant.remaining_queries(0.1, 0.03)
    assert remaining <= pld_accountant.remaining_queries(0.1, 0.03)

    # Test after some queries have been executed
    params = DPParams(
        epsilon=total_epsilon / 2,
        delta=total_delta / 2,
        clipping_thresholds=[None],
        contribution_bound=1,
    )
    accountant.update_budget([Aggregation.COUNT], params)
    remaining_after_update = accountant.remaining_queries(0.1, 0.03)
    assert remaining_after_update < remaining

    # Test with invalid parameters
    with pytest.raises(InvalidPrivacyParametersError):
        accountant.remaining_queries(-0.1, 0.01)

    with pytest.raises(InvalidPrivacyParametersError):
        accountant.remaining_queries(0.1, -0.01)

    # Test with infinite budget
    accountant = RenyiAccountant(1.0, 1.0)
    remaining = accountant.remaining_queries(0.1, 1.0)
    assert remaining == accountant.MAX_REMAINING_QUERIES
