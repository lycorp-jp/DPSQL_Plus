import os

import duckdb
import pandas as pd
import polars as pl
import pytest
from utils import create_test_cases

from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend import DuckDBBackend
from dpsql.dp_params import DPParams
from dpsql.errors import ExecutionBackendError


@pytest.fixture(scope="function")
def conn():
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


# Get test cases and IDs from shared utility
TEST_CASES, TEST_IDS = create_test_cases("duckdb")


@pytest.mark.parametrize(
    "agg_type,column_name,df,group_by,clipping_threshold,expected_result",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_apply_aggregation(
    conn, agg_type, column_name, df, group_by, clipping_threshold, expected_result
):
    """Test apply_aggregation method for various aggregation types."""
    backend = DuckDBBackend(conn)

    result = backend.apply_aggregation(
        agg_type=agg_type,
        column_name=column_name,
        df=df,
        group_by=group_by,
        clipping_threshold=clipping_threshold,
    )

    # Compare the result with expected result
    pd.testing.assert_series_equal(
        result.sort_index(),
        expected_result.sort_index(),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data,expected_count",
    [
        (
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "uid": ["uid1", "uid2", "uid3", "uid1"],
                    "value": [10, 20, 30, 20],
                }
            ),
            3,
        ),
        (
            pl.DataFrame(
                {
                    "id": [],
                    "uid": [],
                    "value": [],
                }
            ),
            0,
        ),
    ],
    ids=["normal case", "empty case"],
)
def test_contribution_bound(conn, data, expected_count):
    backend = DuckDBBackend(conn)
    params = DPParams(
        contribution_bound=1,
        sigma_for_thresholding=0,
        tau=1.0,
        sigmas=[0],
        clipping_thresholds=[None],
    )

    priority_column = "_priority"
    prioritized_data = backend.add_random_priority(data, priority_column)
    filtered_df = backend.contribution_bound(
        prioritized_data, "uid", params, priority_column
    )
    assert len(filtered_df) == expected_count


@pytest.mark.parametrize(
    "data,group_by,selected_keys,expected_count",
    [
        (
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "uid": ["uid1", "uid2", "uid3", "uid1"],
                    "attribute": ["a", "b", "a", "a"],
                    "value": [10, 20, 30, 20],
                }
            ),
            ["attribute"],
            [("a",)],
            3,
        ),
        (
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "uid": ["uid1", "uid2", "uid3", "uid1"],
                    "attribute1": ["a", "a", "a", "b"],
                    "attribute2": ["x", "y", "x", "y"],
                }
            ),
            ["attribute1", "attribute2"],
            [("a", "x")],
            2,
        ),
        (
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "uid": ["uid1", "uid2", "uid3", "uid1"],
                    "attribute": ["a", "b", "a", "a"],
                    "value": [10, 20, 30, 20],
                }
            ),
            ["attribute"],
            [],
            0,
        ),
        (
            pl.DataFrame({"id": [1, 2], "uid": ["uid1", "uid2"], "value": [10, 20]}),
            [],
            [],
            0,
        ),
    ],
    ids=[
        "single column filtering",
        "multi column filtering",
        "empty selected_keys case",
        "no group by and empty selected_keys case",
    ],
)
def test_filter_by_selected_keys(conn, data, group_by, selected_keys, expected_count):
    backend = DuckDBBackend(conn)
    filtered_df = backend.filter_by_selected_keys(data, group_by, selected_keys)
    assert len(filtered_df) == expected_count


def test_execute_sql_preserves_first_bounded_records(conn, mocker):
    backend = DuckDBBackend(conn)
    inner_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "uid": ["uid1", "uid1", "uid1", "uid2"],
            "group": ["A", "X", "A", "A"],
            "value": [10, 20, 30, 40],
            "priority": [0, 1, 2, 0],
        }
    )
    mocker.patch.object(backend, "create_inner_df", return_value=inner_df)
    mocker.patch.object(
        backend,
        "add_random_priority",
        side_effect=lambda df, column: df.with_columns(
            pl.col("priority").alias(column)
        ),
    )
    create_final_df = mocker.patch.object(
        backend,
        "create_final_df",
        return_value=pd.DataFrame({"count(value)": [3.0]}, index=["A"]),
    )

    backend.execute_sql(
        privacy_unit="uid",
        params=DPParams(
            contribution_bound=2,
            min_frequency=2,
            sigma_for_thresholding=0,
            tau=2,
            sigmas=[0],
            clipping_thresholds=[None],
        ),
        inner_sql="",
        agg_columns=[AggregationColumn(Aggregation.COUNT, ["value"])],
        group_by_columns=["group"],
        ordering_terms=[],
    )

    final_filtered_df = create_final_df.call_args.args[0]
    assert sorted(final_filtered_df["id"].to_list()) == [1, 3, 4]
    assert "_dpsql_contribution_priority_" not in final_filtered_df.columns


def test_execute_sql_suppresses_global_aggregation_below_min_frequency(conn, mocker):
    backend = DuckDBBackend(conn)
    inner_df = pl.DataFrame(
        {
            "id": [1, 2],
            "uid": ["uid1", "uid2"],
            "value": [10, 20],
        }
    )
    mocker.patch.object(backend, "create_inner_df", return_value=inner_df)

    result = backend.execute_sql(
        privacy_unit="uid",
        params=DPParams(
            contribution_bound=1,
            min_frequency=3,
            sigma_for_thresholding=0,
            tau=3,
            sigmas=[0],
            clipping_thresholds=[None],
        ),
        inner_sql="",
        agg_columns=[AggregationColumn(Aggregation.COUNT, ["value"])],
        group_by_columns=[],
        ordering_terms=[],
    )

    assert result.empty


def test_get_column_name(conn):
    conn.execute("CREATE TABLE table1 (column1 INT, column2 INT)")
    conn.execute("CREATE TABLE table2 (columnA INT, columnB INT)")

    backend = DuckDBBackend(conn)

    table_name = backend.get_table_name()
    assert table_name == ["table1", "table2"]

    for table in table_name:
        column_name = backend.get_column_name(table)
        if table == "table1":
            assert column_name == ["column1", "column2"]
        elif table == "table2":
            assert column_name == ["columnA", "columnB"]


def test_get_column_name_quotes_identifier(conn):
    conn.execute("CREATE TABLE table1 (column1 INT)")
    conn.execute('CREATE TABLE "table1; CREATE TABLE hacked(z INT)" (safe_column INT)')
    backend = DuckDBBackend(conn)

    assert backend.get_column_name("table1; CREATE TABLE hacked(z INT)") == [
        "safe_column"
    ]
    assert "hacked" not in backend.get_table_name()


def test_create_temporary_table_with_inmemory(conn):
    backend = DuckDBBackend(conn)
    df = pd.DataFrame(
        [
            (1, "uid1", "a", 10),
            (2, "uid2", "b", 20),
            (3, "uid3", "a", 30),
            (4, "uid1", "a", 20),
        ],
        columns=["id", "uid", "attribute", "value"],
    )

    backend.create_temporary_table(df, "test_table", False)

    assert "test_table" in backend.get_table_name()

    assert backend.get_column_name("test_table") == [
        "id",
        "uid",
        "attribute",
        "value",
    ]

    backend.create_temporary_table(df, "test_table2", True)

    assert "test_table2" in backend.get_table_name()

    assert backend.get_column_name("test_table2") == [
        "index",
        "id",
        "uid",
        "attribute",
        "value",
    ]


def test_create_temporary_table_rejects_invalid_name(conn):
    backend = DuckDBBackend(conn)
    df = pd.DataFrame([(1,)], columns=["id"])

    with pytest.raises(ExecutionBackendError):
        backend.create_temporary_table(df, "x; create table hacked(z int)", False)

    assert "hacked" not in backend.get_table_name()


def test_create_temporary_table_rejects_name_collision(conn):
    conn.execute("CREATE TABLE existing_table (id INT)")
    backend = DuckDBBackend(conn)
    df = pd.DataFrame([(1,)], columns=["id"])

    with pytest.raises(ExecutionBackendError):
        backend.create_temporary_table(df, "existing_table", False)


def test_create_temporary_table_with_file():
    conn = duckdb.connect("test.db")
    backend = DuckDBBackend(conn)

    df = pd.DataFrame(
        [
            (1, "uid1", "a", 10),
            (2, "uid2", "b", 20),
            (3, "uid3", "a", 30),
            (4, "uid1", "a", 20),
        ],
        columns=["id", "uid", "attribute", "value"],
    )

    backend.create_temporary_table(df, "test_table", False)

    assert "test_table" in backend.get_table_name()

    assert backend.get_column_name("test_table") == [
        "id",
        "uid",
        "attribute",
        "value",
    ]

    backend.create_temporary_table(df, "test_table2", True)

    assert "test_table2" in backend.get_table_name()

    assert backend.get_column_name("test_table2") == [
        "index",
        "id",
        "uid",
        "attribute",
        "value",
    ]

    conn.close()

    conn = duckdb.connect("test.db")
    backend = DuckDBBackend(conn)

    assert "test_table" not in backend.get_table_name()
    assert "test_table2" not in backend.get_table_name()

    conn.close()
    os.remove("test.db")
