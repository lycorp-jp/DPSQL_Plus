import pytest

from dpsql.accountant import BasicAccountant
from dpsql.aggregation import Aggregation
from dpsql.dp_params import DPParams
from dpsql.errors import InvalidPrivacyParametersError


def test_basic_accountant_warning():
    """Test that initializing BasicAccountant raises a warning."""
    with pytest.warns(
        UserWarning,
        match="BasicAccountant is mainly for testing and debugging purposes",
    ):
        BasicAccountant(1.0, 0.1)


def test_check_budget():
    total_epsilon = 1.0
    total_delta = 0.1
    accountant = BasicAccountant(total_epsilon, total_delta)

    # Success case
    params = DPParams(
        contribution_bound=1, clipping_thresholds=[None], epsilon=1, delta=0.1
    )
    assert accountant.check_budget([Aggregation.COUNT], params)

    # Failure case: exceeds epsilon
    params = DPParams(
        contribution_bound=1, clipping_thresholds=[None], epsilon=2, delta=0.1
    )
    assert not accountant.check_budget([Aggregation.COUNT], params)

    # Failure case: exceeds delta
    params = DPParams(
        contribution_bound=1, clipping_thresholds=[None], epsilon=1, delta=0.2
    )
    assert not accountant.check_budget([Aggregation.COUNT], params)

    # Failure case: epsilon and delta are not set
    params = DPParams(
        contribution_bound=1,
        clipping_thresholds=[None],
        sigmas=[1.0],
        tau=1.0,
        sigma_for_thresholding=1.0,
    )
    assert not accountant.check_budget([Aggregation.COUNT], params)


def test_update_budget():
    total_epsilon = 1.0
    total_delta = 0.1
    accountant = BasicAccountant(total_epsilon, total_delta)

    # Update budget with valid parameters
    params = DPParams(
        contribution_bound=1, clipping_thresholds=[None], epsilon=0.5, delta=0.05
    )
    accountant.update_budget([Aggregation.COUNT], params)
    assert accountant.current_epsilon == 0.5
    assert accountant.current_delta == 0.05

    # Update budget with invalid parameters (no epsilon or delta)
    params = DPParams(
        contribution_bound=1,
        clipping_thresholds=[None],
        sigmas=[1.0],
        tau=1.0,
        sigma_for_thresholding=1.0,
    )
    accountant.update_budget([Aggregation.COUNT], params)
    assert accountant.current_epsilon == float("inf")
    assert accountant.current_delta == float("inf")

    # Test updating budget with infinite budget
    params = DPParams(
        contribution_bound=1, clipping_thresholds=[None], epsilon=0.5, delta=0.05
    )
    accountant = BasicAccountant(1.0, 1.0)
    for _ in range(10):
        accountant.update_budget([Aggregation.COUNT], params)
        assert accountant.check_budget([Aggregation.COUNT], params)


def test_remaining_queries():
    total_epsilon = 1.0 + 1e-5  # To avoid floating point precision issues
    total_delta = 0.1
    accountant = BasicAccountant(total_epsilon, total_delta)

    # Test with fresh accountant (no queries executed yet)
    remaining = accountant.remaining_queries(0.1, 0.01)
    assert remaining == 10  # min(1.0/0.1, 0.1/0.01) = min(10, 10) = 10

    # Test with epsilon being the limiting factor
    remaining = accountant.remaining_queries(0.2, 0.01)
    assert remaining == 5  # min(1.0/0.2, 0.1/0.01) = min(5, 10) = 5

    # Test with delta being the limiting factor
    remaining = accountant.remaining_queries(0.1, 0.05)
    assert remaining == 2  # min(1.0/0.1, 0.1/0.05) = min(10, 2) = 2

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
    accountant = BasicAccountant(1.0, 1.0)
    remaining = accountant.remaining_queries(0.1, 1.0)
    assert remaining == accountant.MAX_REMAINING_QUERIES
