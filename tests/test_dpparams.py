import pytest

from dpsql.accountant import PLDAccountant, RenyiAccountant
from dpsql.accountant.utils import calc_pld_for_tau_thresholding
from dpsql.aggregation import Aggregation
from dpsql.backend.sql_backend import AggregationColumn
from dpsql.dp_params import DPParams, generate_dpparams
from dpsql.errors import InvalidPrivacyParametersError
from dpsql.utils import calibrate_analytic_gaussian_mechanism


def test_get_noise_parameters():
    epsilon = 1.0
    delta = 1e-5

    dp_params = DPParams(
        contribution_bound=1,
        clipping_thresholds=[None],
        epsilon=epsilon,
        delta=delta,
    )
    sensitivities = [1.0]
    sigmas, tau, sigma_for_thresholding = dp_params.get_noise_parameters(sensitivities)
    pld = calc_pld_for_tau_thresholding(
        dp_params.min_frequency,
        tau,
        sigma_for_thresholding,
        dp_params.contribution_bound,
    )

    # epsilon and delta are equally distributed
    # among the aggregation function and tau-thresholding
    estimated_epsilon = pld.get_epsilon_for_delta(delta / 2)
    assert estimated_epsilon == pytest.approx(epsilon / 2)
    assert sigmas[0] == pytest.approx(
        calibrate_analytic_gaussian_mechanism(epsilon / 2, delta / 2, 1.0)
    )


def test_get_noise_parameters_with_small_epsilon():
    epsilon = 0.01
    delta = 1e-1

    dp_params = DPParams(
        contribution_bound=2,
        clipping_thresholds=[None],
        epsilon=epsilon,
        delta=delta,
    )
    sensitivities = [1.0]
    _, tau, sigma_for_thresholding = dp_params.get_noise_parameters(sensitivities)
    pld = calc_pld_for_tau_thresholding(
        dp_params.min_frequency,
        tau,
        sigma_for_thresholding,
        dp_params.contribution_bound,
    )

    # epsilon and delta are equally distributed
    # among the aggregation function and tau-thresholding
    # Since calculation of tau and sigma_for_thresholding is suboptimal,
    # calculation result may be different from the given epsilon.
    estimated_epsilon = pld.get_epsilon_for_delta(delta / 2)
    assert estimated_epsilon <= epsilon / 2


def test_get_noise_parameters_uses_l2_sensitivity_for_thresholding():
    epsilon = 1.0
    delta = 1e-5
    contribution_bound = 4
    dp_params = DPParams(
        contribution_bound=contribution_bound,
        clipping_thresholds=[None],
        epsilon=epsilon,
        delta=delta,
    )

    _, _, sigma_for_thresholding = dp_params.get_noise_parameters([1.0])

    assert sigma_for_thresholding == pytest.approx(
        calibrate_analytic_gaussian_mechanism(
            epsilon / 2,
            delta / 4,
            contribution_bound**0.5,
        )
    )


@pytest.mark.parametrize("accountant_class", [RenyiAccountant, PLDAccountant])
def test_get_noise_parameters_with_accountant(accountant_class):
    """Test get_noise_parameters with RenyiAccountant."""
    epsilon = 1.0
    delta = 1e-5
    sensitivities = [1.0] * 10
    clipping_thresholds = [None] * 10

    # Compute parameters using BasicAccountant
    dp_params_basic = DPParams(
        contribution_bound=1,
        clipping_thresholds=clipping_thresholds,
        epsilon=epsilon,
        delta=delta,
    )
    sigmas_basic, tau_basic, sigma_for_thresholding_basic = (
        dp_params_basic.get_noise_parameters(sensitivities)
    )

    # Compute parameters using RenyiAccountant
    dp_params = DPParams(
        contribution_bound=1,
        clipping_thresholds=clipping_thresholds,
        epsilon=epsilon,
        delta=delta,
        accountant_class=accountant_class,
    )
    sigmas, tau, sigma_for_thresholding = dp_params.get_noise_parameters(sensitivities)

    # We expect that RenyiAccountant provides smaller noise parameters
    assert tau <= tau_basic / 2
    assert sigma_for_thresholding <= sigma_for_thresholding_basic / 2
    assert sigmas[0] <= sigmas_basic[0] / 2

    # Check that the noise parameters satisfy the budget
    accountant = accountant_class(epsilon, delta)
    is_valid = accountant.check_budget(
        agg_funcs=[],
        params=DPParams(
            contribution_bound=1,
            clipping_thresholds=[None],
            min_frequency=1,
            sigmas=sigmas,
            tau=tau,
            sigma_for_thresholding=sigma_for_thresholding,
        ),
    )
    assert is_valid


def test_sigmas_deprecation_warning():
    """Test that using sigmas parameter raises a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="The 'sigmas' parameter is deprecated"):
        DPParams(
            contribution_bound=1,
            clipping_thresholds=[None],
            sigmas=[1.0],
            tau=2.0,
            sigma_for_thresholding=1.0,
        )


def test_mixed_privacy_parameter_modes_are_rejected():
    with pytest.raises(
        InvalidPrivacyParametersError,
        match="Conflicting privacy parameter modes",
    ):
        DPParams(
            contribution_bound=1,
            clipping_thresholds=[None],
            epsilon=1.0,
            delta=1e-5,
            sigmas=[1.0],
            tau=2.0,
            sigma_for_thresholding=1.0,
        )


@pytest.mark.parametrize("name", ["contribution_bound", "min_frequency"])
@pytest.mark.parametrize("value", [1.5, 2.0])
def test_integer_parameters_reject_float_values(name, value):
    parameters = {
        "contribution_bound": 1,
        "min_frequency": 1,
        "clipping_thresholds": [None],
        "epsilon": 1.0,
        "delta": 1e-5,
    }
    parameters[name] = value

    with pytest.raises(
        InvalidPrivacyParametersError,
        match=rf"Invalid `{name}` \(must be an integer >= 1\)",
    ):
        DPParams(**parameters)


def test_generate_dpparams():
    epsilon = 1.0
    delta = 1e-5
    contribution_bound = 5
    min_frequency = 10

    agg_columns = [
        AggregationColumn(Aggregation.COUNT, ["uid"], None, []),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["uid"], "cnt_uniq_uid", []),
        AggregationColumn(Aggregation.SUM, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.AVG, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.STDDEV_POP, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.STDDEV_SAMP, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.VAR_POP, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.VAR_SAMP, ["value"], None, [1.0, 4.0]),
        AggregationColumn(Aggregation.COVAR_POP, ["value"], None, [1.0, 4.0, 1.0, 4.0]),
        AggregationColumn(
            Aggregation.COVAR_SAMP, ["value"], None, [1.0, 4.0, 1.0, 4.0]
        ),
    ]

    dp_params = generate_dpparams(
        {
            "EPSILON": epsilon,
            "DELTA": delta,
            "CONTRIBUTION_BOUND": contribution_bound,
            "MIN_FREQUENCY": min_frequency,
        },
        agg_columns,
    )
    assert dp_params.epsilon == epsilon
    assert dp_params.delta == delta
    assert dp_params.contribution_bound == contribution_bound
    assert dp_params.min_frequency == min_frequency
    assert dp_params.clipping_thresholds == [
        None,
        None,
        [(1.0, 4.0)],
        [(1.0, 4.0)],
        [(1.0, 4.0)],
        [(1.0, 4.0)],
        [(1.0, 4.0)],
        [(1.0, 4.0)],
        [(1.0, 4.0), (1.0, 4.0)],
        [(1.0, 4.0), (1.0, 4.0)],
    ]
