import os
import sqlite3

import pandas as pd
import pytest
from utils import (
    create_nonfinite_aggregation_cases,
    create_null_aggregation_cases,
    create_test_cases,
)

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
NULL_AGGREGATION_CASES, NULL_AGGREGATION_IDS = create_null_aggregation_cases()
NONFINITE_AGGREGATION_CASES, NONFINITE_AGGREGATION_IDS = (
    create_nonfinite_aggregation_cases("sqlite")
)


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


def test_prepare_inner_df_drops_null_privacy_units(conn):
    conn.execute("CREATE TABLE source (uid INTEGER, category TEXT, value REAL)")
    conn.executemany(
        "INSERT INTO source VALUES (?, ?, ?)",
        [(None, "a", 100.0), (1, None, None), (2, "a", 2.0)],
    )
    backend = SQLiteBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")

    result = backend.prepare_inner_df(inner_df, "uid", ["category"])

    assert result["uid"].tolist() == [1, 2]
    assert result["category"].isna().tolist() == [True, False]
    assert result["value"].isna().tolist() == [True, False]
    assert len(inner_df) == 3


@pytest.mark.parametrize(
    "agg_type,columns,rows,expected", NULL_AGGREGATION_CASES, ids=NULL_AGGREGATION_IDS
)
def test_apply_aggregation_with_null_values(conn, agg_type, columns, rows, expected):
    conn.execute("CREATE TABLE source (category TEXT, x REAL, y REAL)")
    conn.executemany("INSERT INTO source VALUES (?, ?, ?)", rows)
    backend = SQLiteBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")

    result = backend.apply_aggregation(
        agg_type, columns, inner_df, ["category"], [(1.0, 5.0)] * len(columns)
    )

    assert result.to_dict() == expected


@pytest.mark.parametrize(
    "agg_type,columns,rows,expected",
    NONFINITE_AGGREGATION_CASES,
    ids=NONFINITE_AGGREGATION_IDS,
)
def test_apply_aggregation_with_nonfinite_values(
    conn, agg_type, columns, rows, expected
):
    conn.execute("CREATE TABLE source (x REAL, y REAL)")
    conn.executemany("INSERT INTO source VALUES (?, ?)", rows)
    # SQLite converts NaN to SQL NULL before DPSQL reads the values.
    assert (
        conn.execute("SELECT COUNT(*) FROM source WHERE x IS NULL").fetchone()[0] == 2
    )
    backend = SQLiteBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")

    result = backend.apply_aggregation(
        agg_type, columns, inner_df, [], [(1.0, 5.0)] * len(columns)
    )

    assert result.tolist() == [expected]


@pytest.mark.parametrize(
    "agg_type,expected", [(Aggregation.COVAR_POP, 1.0), (Aggregation.COVAR_SAMP, 2.0)]
)
def test_covar_uses_complete_pairs(conn, agg_type, expected):
    conn.execute("CREATE TABLE source (category TEXT, x REAL, y REAL)")
    conn.executemany(
        "INSERT INTO source VALUES (?, ?, ?)",
        [
            ("mixed", 1.0, 2.0),
            ("mixed", 3.0, 4.0),
            ("mixed", None, 100.0),
            ("mixed", 100.0, None),
            ("all_null", None, None),
        ],
    )
    backend = SQLiteBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")
    paired_df, columns = backend.prepare_covar_df(inner_df, ["x", "y"])

    assert paired_df[columns[0]].isna().tolist() == [False, False, True, True, True]
    assert paired_df[columns[1]].isna().tolist() == [False, False, True, True, True]
    pd.testing.assert_frame_equal(paired_df[inner_df.columns], inner_df)
    counts = backend.apply_aggregation(
        Aggregation.COUNT, columns, paired_df, ["category"]
    )
    assert counts.to_dict() == {"all_null": 0, "mixed": 2}

    # Zero-count statistics are outside this pair-masking regression.
    result = backend.create_final_df(
        inner_df[inner_df["category"] == "mixed"].copy(),
        [AggregationColumn(agg_type, ["x", "y"])],
        ["category"],
        [0],
        [[(0, 10), (0, 10)]],
    )

    assert result.iloc[0, 0] == pytest.approx(expected)


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
    assert sorted(final_filtered_df["id"].tolist()) == [1, 3, 4]
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


@pytest.fixture(params=[["g"], ["g", "h"]], ids=["single_group", "multi_group"])
def null_group_result(conn, request):
    conn.execute("CREATE TABLE source (uid INTEGER, g INTEGER, h TEXT)")
    conn.executemany(
        "INSERT INTO source VALUES (?, ?, ?)",
        [
            (1, None, "a"),
            (2, None, "a"),
            (3, 8, None),
            (4, 8, None),
            (5, 9, None),
            (6, 9, None),
            (7, None, None),
            (8, None, None),
            (9, 7, "suppressed"),
            (None, 8, None),
        ],
    )
    backend = SQLiteBackend(conn)
    result = backend.execute_sql(
        privacy_unit="uid",
        params=DPParams(
            contribution_bound=1,
            min_frequency=2,
            sigma_for_thresholding=0,
            tau=2,
            sigmas=[0],
            clipping_thresholds=[None],
        ),
        inner_sql="SELECT * FROM source",
        agg_columns=[AggregationColumn(Aggregation.COUNT, ["*"], alias="n")],
        group_by_columns=request.param,
        ordering_terms=[],
    )
    expected = (
        {(None,): 4, (8,): 2, (9,): 2}
        if request.param == ["g"]
        else {(None, "a"): 2, (None, None): 2, (8, None): 2, (9, None): 2}
    )
    return backend, result, expected


def test_execute_sql_preserves_null_group_keys(null_group_result):
    _, result, expected = null_group_result
    actual = {
        tuple(None if pd.isna(value) else value for value in key): count
        for key, count in zip(
            result.index.to_frame(index=False).itertuples(index=False, name=None),
            result["n"],
            strict=True,
        )
    }

    assert actual == expected


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


def test_create_temporary_table_preserves_null_group_keys(conn, null_group_result):
    backend, result, expected = null_group_result

    backend.create_temporary_table(result, "temp.result")

    keys = result.index.names
    selected_columns = ", ".join([*keys, "n"])
    saved = conn.execute(f'SELECT {selected_columns} FROM "temp.result"').fetchall()
    assert {tuple(row[:-1]): row[-1] for row in saved} == expected
    assert conn.execute(
        'SELECT COUNT(*) FROM "temp.result" WHERE g IS NULL'
    ).fetchone()[0] == sum(key[0] is None for key in expected)
