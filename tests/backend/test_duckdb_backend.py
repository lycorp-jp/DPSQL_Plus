import os

import duckdb
import pandas as pd
import polars as pl
import pytest
from utils import (
    create_nonfinite_aggregation_cases,
    create_null_aggregation_cases,
    create_test_cases,
)

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
NULL_AGGREGATION_CASES, NULL_AGGREGATION_IDS = create_null_aggregation_cases()
NONFINITE_AGGREGATION_CASES, NONFINITE_AGGREGATION_IDS = (
    create_nonfinite_aggregation_cases("duckdb")
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


def test_prepare_inner_df_drops_null_privacy_units(conn):
    backend = DuckDBBackend(conn)
    inner_df = backend.create_inner_df(
        "SELECT * FROM (VALUES (1, NULL), (NULL, 'discarded'), (2, 'A')) "
        "AS records(uid, category)"
    )

    result = backend.prepare_inner_df(inner_df, "uid", ["category"])

    assert result.to_dicts() == [
        {"uid": 1, "category": None},
        {"uid": 2, "category": "A"},
    ]


@pytest.mark.parametrize("key_column", ["uid", "category"])
@pytest.mark.parametrize(
    "values",
    [[1.0], [float("nan")], [float("inf")], [-float("inf")], [None], []],
    ids=["finite", "nan", "positive_inf", "negative_inf", "all_null", "empty"],
)
def test_prepare_inner_df_rejects_floating_point_keys(conn, key_column, values):
    uid_type = "DOUBLE" if key_column == "uid" else "BIGINT"
    category_type = "DOUBLE" if key_column == "category" else "VARCHAR"
    conn.execute(f"CREATE TABLE source (uid {uid_type}, category {category_type})")
    rows = [(value, "a") if key_column == "uid" else (1, value) for value in values]
    if rows:
        conn.executemany("INSERT INTO source VALUES (?, ?)", rows)
    backend = DuckDBBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")

    with pytest.raises(ExecutionBackendError, match="NaN or infinity") as error:
        backend.prepare_inner_df(inner_df, "uid", ["category"])

    assert error.value.context["column"] == key_column


@pytest.mark.parametrize(
    "agg_type,columns,rows,expected", NULL_AGGREGATION_CASES, ids=NULL_AGGREGATION_IDS
)
def test_apply_aggregation_with_null_values(conn, agg_type, columns, rows, expected):
    conn.execute("CREATE TABLE source (category VARCHAR, x DOUBLE, y DOUBLE)")
    conn.executemany("INSERT INTO source VALUES (?, ?, ?)", rows)
    backend = DuckDBBackend(conn)
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
    conn.execute("CREATE TABLE source (x DOUBLE, y DOUBLE)")
    conn.executemany("INSERT INTO source VALUES (?, ?)", rows)
    backend = DuckDBBackend(conn)
    inner_df = backend.create_inner_df("SELECT * FROM source")

    result = backend.apply_aggregation(
        agg_type, columns, inner_df, [], [(1.0, 5.0)] * len(columns)
    )

    assert result.tolist() == [expected]


@pytest.mark.parametrize(
    "agg_type,expected", [(Aggregation.COVAR_POP, 1.0), (Aggregation.COVAR_SAMP, 2.0)]
)
def test_covar_uses_complete_pairs(conn, agg_type, expected):
    backend = DuckDBBackend(conn)
    conn.execute("CREATE TABLE source (category VARCHAR, x DOUBLE, y DOUBLE)")
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
    inner_df = backend.create_inner_df("SELECT * FROM source")
    masked, paired_columns = backend.prepare_covar_df(inner_df, ["x", "y"])

    assert masked.select(inner_df.columns).equals(inner_df)
    assert masked.select(paired_columns).to_dict(as_series=False) == {
        paired_columns[0]: [1.0, 3.0, None, None, None],
        paired_columns[1]: [2.0, 4.0, None, None, None],
    }
    counts = backend.apply_aggregation(
        Aggregation.COUNT, paired_columns, masked, ["category"]
    )
    assert counts.to_dict() == {"all_null": 0, "mixed": 2}

    # Zero-count statistics are outside this pair-masking regression.
    result = backend.create_final_df(
        inner_df.filter(pl.col("category") == "mixed"),
        [AggregationColumn(agg_type, ["x", "y"], alias="covar")],
        ["category"],
        [0],
        [[(0, 10), (0, 10)]],
    )

    assert result.loc["mixed", "covar"] == pytest.approx(expected)


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


@pytest.mark.parametrize("group_by", [["category"], ["category", "segment"]])
def test_execute_sql_preserves_null_group_keys(conn, group_by):
    backend = DuckDBBackend(conn)
    large_key = 2**60 + 1
    inner_sql = (
        f"SELECT * FROM (VALUES (1, {large_key}, 'X', 2.0), "
        f"(2, {large_key}, 'X', 3.0), (3, NULL, NULL, 4.0), "
        f"(4, NULL, NULL, 5.0), (5, {large_key + 1}, 'rare', 6.0), "
        "(NULL, NULL, NULL, 100.0)) AS records(uid, category, segment, x)"
    )

    result = backend.execute_sql(
        privacy_unit="uid",
        params=DPParams(
            contribution_bound=1,
            min_frequency=2,
            sigma_for_thresholding=0,
            tau=2,
            sigmas=[0, 0],
            clipping_thresholds=[None, [(1, 10)]],
        ),
        inner_sql=inner_sql,
        agg_columns=[
            AggregationColumn(Aggregation.COUNT, ["*"], alias="count"),
            AggregationColumn(Aggregation.SUM, ["x"], alias="total"),
        ],
        group_by_columns=group_by,
        ordering_terms=[],
    )

    rows = result.reset_index()
    assert len(rows) == 2
    assert rows["count"].tolist() == [2, 2]
    assert rows.loc[rows["category"].notna(), "category"].tolist() == [large_key]
    assert rows.loc[rows["category"].notna(), "total"].tolist() == [5]
    assert rows.loc[rows["category"].isna(), "total"].tolist() == [9]
    if len(group_by) == 2:
        assert rows.loc[rows["category"].isna(), "segment"].isna().all()


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


@pytest.mark.parametrize("group_by", [["category"], ["category", "segment"]])
def test_create_temporary_table_preserves_null_group_keys(conn, group_by):
    backend = DuckDBBackend(conn)
    large_key = 2**60 + 1
    inner_df = backend.create_inner_df(
        f"SELECT * FROM (VALUES ({large_key}, 'X', 2.0), "
        f"({large_key}, 'X', 3.0), (NULL, NULL, 4.0)) "
        "AS records(category, segment, x)"
    )
    result = backend.apply_aggregation(
        Aggregation.SUM, ["x"], inner_df, group_by, [(1, 10)]
    ).to_frame("total")

    backend.create_temporary_table(result, "private_result")

    assert conn.execute(
        "SELECT category, total FROM private_result ORDER BY category"
    ).fetchall() == [(large_key, 5.0), (None, 4.0)]
    if len(group_by) == 2:
        assert conn.execute(
            "SELECT segment FROM private_result WHERE category IS NULL"
        ).fetchall() == [(None,)]


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
