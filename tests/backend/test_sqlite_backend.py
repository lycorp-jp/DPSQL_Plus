import os
import sqlite3

import pandas as pd
import pytest
from utils import create_test_cases

from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend import SQLiteBackend
from dpsql.dp_params import DPParams


@pytest.fixture(scope="function")
def conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


# Get test cases and IDs from shared utility
TEST_CASES, TEST_IDS = create_test_cases("sqlite")


@pytest.mark.parametrize(
    "agg_type,column_name,df,group_by,clipping_threshold,expected_result",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_apply_aggregation(
    conn, agg_type, column_name, df, group_by, clipping_threshold, expected_result
):
    """Test apply_aggregation method for various aggregation types."""
    backend = SQLiteBackend(conn)

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
            pd.DataFrame(
                [(1, "uid1", 10), (2, "uid2", 20), (3, "uid3", 30), (4, "uid1", 20)],
                columns=["id", "uid", "value"],
            ),
            3,
        ),
        (
            pd.DataFrame({"id": [], "uid": [], "value": []}),
            0,
        ),
    ],
    ids=["normal case", "empty case"],
)
def test_contribution_bound(conn, data, expected_count):
    backend = SQLiteBackend(conn)
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
            pd.DataFrame(
                [
                    (1, "uid1", "a"),
                    (2, "uid2", "b"),
                    (3, "uid3", "a"),
                    (4, "uid1", "a"),
                ],
                columns=["id", "uid", "attribute"],
            ),
            ["attribute"],
            [("a",)],
            3,
        ),
        (
            pd.DataFrame(
                [
                    (1, "uid1", "a", "x"),
                    (2, "uid2", "a", "y"),
                    (3, "uid3", "a", "x"),
                    (4, "uid1", "b", "y"),
                ],
                columns=["id", "uid", "attribute1", "attribute2"],
            ),
            ["attribute1", "attribute2"],
            [("a", "x")],
            2,
        ),
        (
            pd.DataFrame(
                [
                    (1, "uid1", "a"),
                    (2, "uid2", "b"),
                    (3, "uid3", "a"),
                    (4, "uid1", "a"),
                ],
                columns=["id", "uid", "attribute"],
            ),
            ["attribute"],
            [],
            0,
        ),
        (
            pd.DataFrame(
                [
                    (1, "uid1", 10),
                    (2, "uid2", 20),
                ],
                columns=["id", "uid", "value"],
            ),
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
    backend = SQLiteBackend(conn)
    filtered_df = backend.filter_by_selected_keys(data, group_by, selected_keys)
    assert len(filtered_df) == expected_count


def test_execute_sql_preserves_first_bounded_records(conn, mocker):
    backend = SQLiteBackend(conn)
    inner_df = pd.DataFrame(
        [
            (1, "uid1", "A", 10, 0),
            (2, "uid1", "X", 20, 1),
            (3, "uid1", "A", 30, 2),
            (4, "uid2", "A", 40, 0),
        ],
        columns=["id", "uid", "group", "value", "priority"],
    )
    mocker.patch.object(backend, "create_inner_df", return_value=inner_df)
    mocker.patch.object(
        backend,
        "add_random_priority",
        side_effect=lambda df, column: df.assign(**{column: df["priority"]}),
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
    assert final_filtered_df["id"].tolist() == [1, 3, 4]
    assert "_dpsql_contribution_priority_" not in final_filtered_df


def test_execute_sql_suppresses_global_aggregation_below_min_frequency(conn, mocker):
    backend = SQLiteBackend(conn)
    mocker.patch.object(
        backend,
        "create_inner_df",
        return_value=pd.DataFrame(
            [(1, "uid1", 10), (2, "uid2", 20)],
            columns=["id", "uid", "value"],
        ),
    )

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
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE table1 (column1 INT, column2 INT)")
    cursor.execute("CREATE TABLE table2 (columnA INT, columnB INT)")

    backend = SQLiteBackend(conn)

    table_name = backend.get_table_name()
    assert table_name == ["table1", "table2"]

    for table in table_name:
        column_name = backend.get_column_name(table)
        if table == "table1":
            assert column_name == ["column1", "column2"]
        elif table == "table2":
            assert column_name == ["columnA", "columnB"]


def test_get_column_name_escapes_quotes(conn):
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE "odd\'name" (safe_column INT)')

    backend = SQLiteBackend(conn)

    assert backend.get_column_name("odd'name") == ["safe_column"]


def test_get_column_name_rejects_injected_identifier(conn):
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE table1 (safe_column INT)")

    backend = SQLiteBackend(conn)

    assert backend.get_column_name("table1'; CREATE TABLE hacked(z INT); --") == []
    assert backend.get_table_name() == ["table1"]


def test_is_inmemory_db():
    conn = sqlite3.connect(":memory:")
    backend = SQLiteBackend(conn)

    assert backend.is_inmemory_db()

    conn.close()

    conn = sqlite3.connect("test.db")
    backend = SQLiteBackend(conn)

    assert not backend.is_inmemory_db()

    conn.close()
    os.remove("test.db")


def test_create_temporary_table():
    conn = sqlite3.connect(":memory:")
    backend = SQLiteBackend(conn)

    df = pd.DataFrame(
        [
            (1, "uid1", "a", 10),
            (2, "uid2", "b", 20),
            (3, "uid3", "a", 30),
            (4, "uid1", "a", 20),
        ],
        columns=["id", "uid", "attribute", "value"],
    )

    backend.create_temporary_table(df, "temp.test_table", False)

    assert "temp.test_table" in backend.get_table_name()

    assert backend.get_column_name("temp.test_table") == [
        "id",
        "uid",
        "attribute",
        "value",
    ]

    backend.create_temporary_table(df, "temp.test_table2", True)

    assert "temp.test_table2" in backend.get_table_name()

    assert backend.get_column_name("temp.test_table2") == [
        "index",
        "id",
        "uid",
        "attribute",
        "value",
    ]

    conn.close()

    conn = sqlite3.connect(":memory:")
    backend = SQLiteBackend(conn)

    assert "temp.test_table" not in backend.get_table_name()
    assert "temp.test_table2" not in backend.get_table_name()

    conn.close()
