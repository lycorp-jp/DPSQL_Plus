import pytest

from dpsql.aggregation import (
    Aggregation,
    AggregationColumn,
)
from dpsql.errors import AggregationError


def test_aggregation_column():
    """Test AggregationColumn creation and properties."""
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

    aggs = [agg_column.aggregation_type for agg_column in agg_columns]
    columns = [agg_column.columns for agg_column in agg_columns]
    aliases = [agg_column.alias for agg_column in agg_columns]
    parameters = [agg_column.parameters for agg_column in agg_columns]

    assert aggs == [
        Aggregation.COUNT,
        Aggregation.COUNT_DISTINCT,
        Aggregation.SUM,
        Aggregation.AVG,
        Aggregation.STDDEV_POP,
        Aggregation.STDDEV_SAMP,
        Aggregation.VAR_POP,
        Aggregation.VAR_SAMP,
        Aggregation.COVAR_POP,
        Aggregation.COVAR_SAMP,
    ]
    assert columns == [
        ["uid"],
        ["uid"],
        ["value"],
        ["value"],
        ["value"],
        ["value"],
        ["value"],
        ["value"],
        ["value"],
        ["value"],
    ]
    assert aliases == [
        None,
        "cnt_uniq_uid",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert parameters == [
        [],
        [],
        [1.0, 4.0],
        [1.0, 4.0],
        [1.0, 4.0],
        [1.0, 4.0],
        [1.0, 4.0],
        [1.0, 4.0],
        [1.0, 4.0, 1.0, 4.0],
        [1.0, 4.0, 1.0, 4.0],
    ]


def test_aggregation_from_str():
    """Test Aggregation.from_str method with legacy and new names."""
    # Test exact matches
    assert Aggregation.from_str("COUNT") == Aggregation.COUNT
    assert Aggregation.from_str("COUNT_DISTINCT") == Aggregation.COUNT_DISTINCT
    assert Aggregation.from_str("SUM") == Aggregation.SUM
    assert Aggregation.from_str("AVG") == Aggregation.AVG
    assert Aggregation.from_str("STDDEV_SAMP") == Aggregation.STDDEV_SAMP
    assert Aggregation.from_str("STDDEV_POP") == Aggregation.STDDEV_POP
    assert Aggregation.from_str("VAR_SAMP") == Aggregation.VAR_SAMP
    assert Aggregation.from_str("VAR_POP") == Aggregation.VAR_POP
    assert Aggregation.from_str("COVAR_SAMP") == Aggregation.COVAR_SAMP
    assert Aggregation.from_str("COVAR_POP") == Aggregation.COVAR_POP
    assert Aggregation.from_str("SQUARED_SUM") == Aggregation.SQUARED_SUM

    # Test None case
    assert Aggregation.from_str(None) == Aggregation.NONE

    # Test invalid aggregation name
    with pytest.raises(AggregationError):
        Aggregation.from_str("INVALID_AGG")


def test_aggregation_get_sensitivity():
    """Test get_sensitivity calculation for different aggregation types."""
    contribution_bound = 2.0
    clipping_threshold = [(-10.0, 10.0)]

    # Test COUNT and COUNT_DISTINCT
    assert Aggregation.COUNT.get_sensitivity(contribution_bound) == contribution_bound
    assert (
        Aggregation.COUNT_DISTINCT.get_sensitivity(contribution_bound)
        == contribution_bound
    )

    # Test SUM
    expected_sum_sensitivity = contribution_bound * max(
        abs(-10), abs(10)
    )  # 2.0 * 10 = 20.0
    assert (
        Aggregation.SUM.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_sum_sensitivity
    )

    # Test aggregations with Bezier mechanism
    # (sensitivity = contribution_bound * (upper - lower))
    expected_bezier_sensitivity = contribution_bound * (
        10.0 - (-10.0)
    )  # 2.0 * 20 = 40.0

    assert (
        Aggregation.AVG.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.STDDEV_SAMP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.STDDEV_POP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.VAR_SAMP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.VAR_POP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.COVAR_SAMP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
    assert (
        Aggregation.COVAR_POP.get_sensitivity(contribution_bound, clipping_threshold)
        == expected_bezier_sensitivity
    )
