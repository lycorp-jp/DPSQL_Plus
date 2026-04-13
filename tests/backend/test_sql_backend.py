import pandas as pd

from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend import SQLBackend
from dpsql.backend.aggregation_strategies import (
    AvgAggregation,
    CountAggregation,
    CountDistinctAggregation,
    StddevAggregation,
    SumAggregation,
    VarAggregation,
)
from dpsql.dp_params import DPParams

agg_types = [
    (Aggregation.COUNT, CountAggregation),
    (Aggregation.COUNT_DISTINCT, CountDistinctAggregation),
    (Aggregation.SUM, SumAggregation),
    (Aggregation.AVG, AvgAggregation),
    (Aggregation.STDDEV_POP, StddevAggregation),
    (Aggregation.STDDEV_SAMP, StddevAggregation),
    (Aggregation.VAR_POP, VarAggregation),
    (Aggregation.VAR_SAMP, VarAggregation),
]


class MockSQLBackend(SQLBackend):
    def create_inner_df(self, inner_sql):
        pass

    def contribution_bound(
        self,
        inner_df,
        privacy_unit,
        params,
    ):
        pass

    def apply_aggregation(
        self,
        agg_type,
        column_name,
        df,
        group_by,
        clipping_threshold,
    ):
        pass

    def filter_by_selected_keys(
        self,
        df,
        group_by,
        selected_keys,
    ):
        pass

    def get_table_name(self):
        pass

    def use_database(self, database_name):
        pass

    def get_column_name(self, table_name):
        pass

    def create_temporary_table(self, df, table_name, index):
        pass


def test_execute_sql(mocker):
    backend = MockSQLBackend()
    agg_columns = [AggregationColumn(agg_type, ["value"]) for agg_type, _ in agg_types]
    expected_result_df = pd.DataFrame(
        [
            (3, 60, 400, 20.0, 10.0, 12.5, 125.0, 25.0),
            (1, 20, 400, 20.0, 0.0, 0.0, 0.0, 0.0),
        ],
        columns=[
            "count(value)",
            "count_distinct(value)",
            "sum(value)",
            "avg(value)",
            "stddev_pop(value)",
            "stddev_samp(value)",
            "var_pop(value)",
            "var_samp(value)",
        ],
    )

    # Mock the methods called within execute_sql
    mocker.patch.object(backend, "create_inner_df", return_value=None)
    mocker.patch.object(backend, "contribution_bound", return_value=None)
    mocker.patch.object(backend, "key_selection", return_value=None)
    mocker.patch.object(backend, "filter_by_selected_keys", return_value=None)
    mocker.patch.object(backend, "create_final_df", return_value=expected_result_df)

    result_df = backend.execute_sql(
        privacy_unit="privacy_unit",
        params=DPParams(
            epsilon=0.1,
            delta=1e-5,
            contribution_bound=1,
            clipping_thresholds=[[(0, 100)]] * len(agg_columns),
        ),
        inner_sql="",
        agg_columns=agg_columns,
        group_by_columns=["attribute"],
        ordering_terms=[],
    )

    backend.create_inner_df.assert_called_once()
    assert backend.contribution_bound.call_count == 2  # contribution bound called twice
    backend.key_selection.assert_called_once()
    backend.filter_by_selected_keys.assert_called_once()
    backend.create_final_df.assert_called_once()

    pd.testing.assert_frame_equal(result_df, expected_result_df)


def test_create_final_df(mocker):
    backend = MockSQLBackend()
    agg_columns = [AggregationColumn(agg_type, ["value"]) for agg_type, _ in agg_types]

    # Mock the compute methods of all aggregation strategies
    for _, agg_strategy_cls in agg_types:
        mocker.patch.object(
            agg_strategy_cls,
            "compute",
            return_value=pd.Series([10.0, 10.0], index=["a", "b"]),
        )

    final_df = backend.create_final_df(
        None,
        agg_columns,
        group_by=["attribute"],
        sigmas=[1.0] * len(agg_columns),
        clipping_thresholds=[[(0, 100)]] * len(agg_columns),
    )
    expected_df = pd.DataFrame(
        [
            (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        ],
        columns=[
            "count(value)",
            "count_distinct(value)",
            "sum(value)",
            "avg(value)",
            "stddev_pop(value)",
            "stddev_samp(value)",
            "var_pop(value)",
            "var_samp(value)",
        ],
        index=["a", "b"],
    )
    pd.testing.assert_frame_equal(final_df, expected_df)


def test_key_selection(mocker):
    backend = MockSQLBackend()

    # Mock the apply_aggregation method to return fixed counts
    mocker.patch.object(
        backend,
        "apply_aggregation",
        return_value=pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"]),
    )

    # Set min_frequency=2 for minimum frequency rule
    selected_keys = backend.key_selection(
        None, ["group"], "uid", min_frequency=2, sigma=0, tau=1
    )
    assert ("a",) in selected_keys
    assert ("b",) in selected_keys
    assert ("c",) not in selected_keys

    # Set min_frequency=3 for minimum frequency rule
    selected_keys = backend.key_selection(
        None, ["group"], "uid", min_frequency=3, sigma=0, tau=1
    )
    assert ("a",) in selected_keys
    assert ("b",) not in selected_keys
    assert ("c",) not in selected_keys
