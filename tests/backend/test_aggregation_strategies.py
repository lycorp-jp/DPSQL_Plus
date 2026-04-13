from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend.aggregation_strategies import (
    AggregationFactory,
    AvgAggregation,
    CountAggregation,
    CountDistinctAggregation,
    StddevAggregation,
    SumAggregation,
    VarAggregation,
)

agg_types = [
    (Aggregation.COUNT, CountAggregation),
    (Aggregation.COUNT_DISTINCT, CountDistinctAggregation),
    (Aggregation.SUM, SumAggregation),
    (Aggregation.AVG, AvgAggregation),
    (Aggregation.STDDEV_SAMP, StddevAggregation),
    (Aggregation.STDDEV_POP, StddevAggregation),
    (Aggregation.VAR_SAMP, VarAggregation),
    (Aggregation.VAR_POP, VarAggregation),
]


@pytest.mark.parametrize(
    "agg_type,expected_strategy",
    agg_types,
)
def test_aggregation_factory(agg_type, expected_strategy):
    # Test that the factory creates the correct strategy instance
    strategy = AggregationFactory.create_strategy(agg_type, backend=None)
    assert isinstance(strategy, expected_strategy)


@pytest.mark.parametrize(
    "agg_type",
    [agg_type for agg_type, _ in agg_types],
)
def test_get_column_name(agg_type):
    # Test without Alias
    strategy = AggregationFactory.create_strategy(agg_type, backend=None)
    agg_column = AggregationColumn(agg_type, ["test_column"])
    assert (
        strategy.get_column_name(agg_column) == f"{agg_type.name.lower()}(test_column)"
    )

    # Test with Alias
    agg_column_with_alias = AggregationColumn(
        agg_type, ["test_column"], alias="custom_alias"
    )
    assert strategy.get_column_name(agg_column_with_alias) == "custom_alias"


@pytest.mark.parametrize(
    "agg_type",
    [agg_type for agg_type, _ in agg_types],
)
def test_compute(mocker, agg_type):
    # Create a mock backend
    aggregation_results = {
        Aggregation.COUNT: pd.Series([10.0]),
        Aggregation.COUNT_DISTINCT: pd.Series([8.0]),
        Aggregation.SUM: pd.Series([150.0]),
        Aggregation.SQUARED_SUM: pd.Series([2500.0]),
        Aggregation.AVG: pd.Series([15.0]),
        Aggregation.STDDEV_SAMP: pd.Series([np.sqrt(250 / 9)]),
        Aggregation.STDDEV_POP: pd.Series([5.0]),
        Aggregation.VAR_SAMP: pd.Series([250 / 9]),
        Aggregation.VAR_POP: pd.Series([25.0]),
    }
    mock_backend = mocker.MagicMock()
    mock_backend.apply_aggregation.side_effect = (
        lambda agg_type, *args, **kwargs: aggregation_results[agg_type]
    )

    # Test the compute method of each strategy
    strategy = AggregationFactory.create_strategy(agg_type, backend=mock_backend)
    agg_column = AggregationColumn(agg_type, ["test_column"])
    clipping_threshold = [(0.0, 100.0)]
    result = strategy.compute(
        filtered_df=None,
        agg_column=agg_column,
        group_by=None,
        sigma=0.0,
        clipping_threshold=clipping_threshold,
    )
    expected_result = aggregation_results[agg_type]
    pd.testing.assert_series_equal(result, expected_result)


@pytest.mark.parametrize(
    "agg_type",
    [agg_type for agg_type, _ in agg_types],
)
def test_compute_group_by(mocker, agg_type):
    # Create a mock backend
    attributes = ["A", "B"]
    aggregation_results = {
        Aggregation.COUNT: pd.Series([10.0, 5.0], index=attributes),
        Aggregation.COUNT_DISTINCT: pd.Series([8.0, 4.0], index=attributes),
        Aggregation.SUM: pd.Series([150.0, 80.0], index=attributes),
        Aggregation.SQUARED_SUM: pd.Series([2500.0, 1600.0], index=attributes),
        Aggregation.AVG: pd.Series([15.0, 16.0], index=attributes),
        Aggregation.STDDEV_SAMP: pd.Series(
            [np.sqrt(250 / 9), np.sqrt(80.0)], index=attributes
        ),
        Aggregation.STDDEV_POP: pd.Series([5.0, 8.0], index=attributes),
        Aggregation.VAR_SAMP: pd.Series([250 / 9, 80.0], index=attributes),
        Aggregation.VAR_POP: pd.Series([25.0, 64.0], index=attributes),
        Aggregation.COVAR_SAMP: pd.Series([12.5, 20.0], index=attributes),
        Aggregation.COVAR_POP: pd.Series([10.0, 16.0], index=attributes),
    }
    mock_backend = mocker.MagicMock()
    mock_backend.apply_aggregation.side_effect = (
        lambda agg_type, *args, **kwargs: aggregation_results[agg_type]
    )

    # Test the compute method of each strategy
    strategy = AggregationFactory.create_strategy(agg_type, backend=mock_backend)
    agg_column = AggregationColumn(agg_type, ["test_column"])
    clipping_threshold = [(0.0, 100.0)]
    result = strategy.compute(
        filtered_df=None,
        agg_column=agg_column,
        group_by="attribute",
        sigma=0.0,
        clipping_threshold=clipping_threshold,
    )
    expected_result = aggregation_results[agg_type]
    pd.testing.assert_series_equal(result, expected_result)
