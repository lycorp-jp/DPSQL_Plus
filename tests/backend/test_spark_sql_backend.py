import tempfile

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from utils import (
    create_nonfinite_aggregation_cases,
    create_null_aggregation_cases,
    create_test_cases,
)

from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend import SparkSQLBackend
from dpsql.dp_params import DPParams
from dpsql.errors import ExecutionBackendError


@pytest.fixture(scope="function")
def spark():
    with tempfile.TemporaryDirectory() as warehouse_location:
        spark = (
            SparkSession.builder.appName("Example")
            .config("spark.sql.warehouse.dir", warehouse_location)
            .config("spark.sql.legacy.createHiveTableByDefault", "false")
            .getOrCreate()
        )
        yield spark
        spark.stop()


# Get test cases and IDs from shared utility
TEST_CASES, TEST_IDS = create_test_cases("spark")
NULL_AGGREGATION_CASES, NULL_AGGREGATION_IDS = create_null_aggregation_cases()
NONFINITE_AGGREGATION_CASES, NONFINITE_AGGREGATION_IDS = (
    create_nonfinite_aggregation_cases("spark")
)


@pytest.mark.parametrize(
    "agg_type,column_name,data,schema,group_by,clipping_threshold,expected_result",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_apply_aggregation(
    spark,
    agg_type,
    column_name,
    data,
    schema,
    group_by,
    clipping_threshold,
    expected_result,
):
    """Test apply_aggregation method for various aggregation types."""
    backend = SparkSQLBackend(spark)
    df = spark.createDataFrame(data, schema)

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


def test_prepare_inner_df_drops_null_privacy_units(spark):
    backend = SparkSQLBackend(spark)
    df = spark.createDataFrame(
        [(None, "A", 100), ("u1", None, 2), ("u2", "A", 3)],
        "uid string, g string, x int",
    )

    result = backend.prepare_inner_df(df, "uid", ["g"])

    assert {(row.uid, row.g, row.x) for row in result.collect()} == {
        ("u1", None, 2),
        ("u2", "A", 3),
    }


@pytest.mark.parametrize("key_column", ["uid", "g"])
@pytest.mark.parametrize("key_type", [FloatType(), DoubleType()])
@pytest.mark.parametrize(
    "values",
    [[1.0], [float("nan")], [float("inf")], [-float("inf")], [None], []],
    ids=["finite", "nan", "positive_inf", "negative_inf", "all_null", "empty"],
)
def test_prepare_inner_df_rejects_floating_point_keys(
    spark, key_column, key_type, values
):
    backend = SparkSQLBackend(spark)
    schema = StructType(
        [
            StructField("uid", key_type if key_column == "uid" else StringType()),
            StructField("g", key_type if key_column == "g" else StringType()),
        ]
    )
    rows = [(value, "A") if key_column == "uid" else ("u1", value) for value in values]
    df = spark.createDataFrame(rows, schema)

    with pytest.raises(ExecutionBackendError, match="Floating-point"):
        backend.prepare_inner_df(df, "uid", ["g"])


@pytest.mark.parametrize(
    "agg_type,columns,rows,expected", NULL_AGGREGATION_CASES, ids=NULL_AGGREGATION_IDS
)
def test_apply_aggregation_with_null_values(spark, agg_type, columns, rows, expected):
    backend = SparkSQLBackend(spark)
    inner_df = spark.createDataFrame(rows, "category string, x double, y double")

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
    spark, agg_type, columns, rows, expected
):
    backend = SparkSQLBackend(spark)
    inner_df = spark.createDataFrame(rows, "x double, y double")

    result = backend.apply_aggregation(
        agg_type, columns, inner_df, [], [(1.0, 5.0)] * len(columns)
    )

    assert result.tolist() == [expected]


@pytest.mark.parametrize(
    "agg_type,expected", [(Aggregation.COVAR_POP, 1.0), (Aggregation.COVAR_SAMP, 2.0)]
)
def test_covar_uses_complete_pairs(spark, agg_type, expected):
    backend = SparkSQLBackend(spark)
    df = spark.createDataFrame(
        [
            ("mixed", 1.0, 2.0),
            ("mixed", 3.0, 4.0),
            ("mixed", None, 100.0),
            ("mixed", 100.0, None),
            ("all_null", None, None),
        ],
        "g string, x double, y double",
    )
    masked, columns = backend.prepare_covar_df(df, ["x", "y"])
    all_null = masked.filter("g = 'all_null'").select(*columns).collect()
    assert len(all_null) == 1
    assert tuple(all_null[0]) == (None, None)
    assert df.count() == masked.count()

    result = backend.create_final_df(
        df.filter("g = 'mixed'"),
        [AggregationColumn(agg_type, ["x", "y"], alias="covar")],
        ["g"],
        [0],
        [[(0.0, 10.0), (0.0, 10.0)]],
    )

    assert result.loc["mixed", "covar"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "data,schema,expected_count",
    [
        (
            [
                (1, "uid1", 10),
                (2, "uid2", 20),
                (3, "uid3", 30),
                (4, "uid1", 20),
            ],
            ["id", "uid", "value"],
            3,
        ),
        (
            [],
            StructType(
                [
                    StructField("id", IntegerType(), True),
                    StructField("uid", StringType(), True),
                    StructField("value", IntegerType(), True),
                ]
            ),
            0,
        ),
    ],
    ids=["normal case", "empty case"],
)
def test_contribution_bound(spark, data, schema, expected_count):
    backend = SparkSQLBackend(spark)
    df = backend.spark.createDataFrame(data, schema)

    params = DPParams(
        contribution_bound=1,
        sigma_for_thresholding=0,
        tau=1.0,
        sigmas=[0],
        clipping_thresholds=[None],
    )

    priority_column = "_priority"
    prioritized_df = backend.add_random_priority(df, priority_column)
    filtered_df = backend.contribution_bound(
        prioritized_df, "uid", params, priority_column
    )
    assert filtered_df.count() == expected_count


@pytest.mark.parametrize(
    "data,schema,group_by,selected_keys,expected_count",
    [
        (
            [
                (1, "uid1", "a"),
                (2, "uid2", "a"),
                (3, "uid3", "a"),
                (4, "uid1", "b"),
            ],
            ["id", "uid", "attribute"],
            ["attribute"],
            [("a",)],
            3,
        ),
        (
            [
                (1, "uid1", "a", "x"),
                (2, "uid2", "a", "y"),
                (3, "uid3", "a", "x"),
                (4, "uid1", "b", "y"),
            ],
            ["id", "uid", "attribute1", "attribute2"],
            ["attribute1", "attribute2"],
            [("a", "x")],
            2,
        ),
        (
            [
                (1, "uid1", "a"),
                (2, "uid2", "a"),
                (3, "uid3", "a"),
                (4, "uid1", "b"),
            ],
            ["id", "uid", "attribute"],
            ["attribute"],
            [],
            0,
        ),
        (
            [(1, "uid1", 10), (2, "uid2", 20)],
            ["id", "uid", "value"],
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
def test_filter_by_selected_keys(
    spark, data, schema, group_by, selected_keys, expected_count
):
    backend = SparkSQLBackend(spark)
    df = backend.spark.createDataFrame(data, schema)
    filtered_df = backend.filter_by_selected_keys(df, group_by, selected_keys)
    assert filtered_df.count() == expected_count


def test_execute_sql_preserves_first_bounded_records(spark, mocker):
    backend = SparkSQLBackend(spark)
    inner_df = spark.createDataFrame(
        [
            (1, "uid1", "A", 10, 0),
            (2, "uid1", "X", 20, 1),
            (3, "uid1", "A", 30, 2),
            (4, "uid2", "A", 40, 0),
        ],
        ["id", "uid", "group", "value", "priority"],
    )
    mocker.patch.object(backend, "create_inner_df", return_value=inner_df)
    mocker.patch.object(
        backend,
        "add_random_priority",
        side_effect=lambda df, column: df.withColumn(column, df["priority"]),
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
    assert sorted(row.id for row in final_filtered_df.select("id").collect()) == [
        1,
        3,
        4,
    ]
    assert "_dpsql_contribution_priority_" not in final_filtered_df.columns


def test_execute_sql_suppresses_global_aggregation_below_min_frequency(spark, mocker):
    backend = SparkSQLBackend(spark)
    inner_df = spark.createDataFrame(
        [(1, "uid1", 10), (2, "uid2", 20)],
        ["id", "uid", "value"],
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


def _execute_null_group_query(spark, group_kind, all_null=False):
    backend = SparkSQLBackend(spark)
    large_key = 2**60 + 1
    if group_kind == "integer":
        schema = "uid string, g long"
        group_by = ["g"]
        keys = [(None,), (None,), (large_key,), (large_key,)]
    elif group_kind == "string":
        schema = "uid string, g string"
        group_by = ["g"]
        keys = [(None,), (None,), ("A",), ("A",)]
    else:
        schema = "uid string, g long, h string"
        group_by = ["g", "h"]
        keys = [(None, "A"), (None, "A"), (large_key, None), (large_key, None)]
    if all_null:
        keys = [tuple(None for _ in group_by)] * 4
    df = spark.createDataFrame([(f"u{i}", *key) for i, key in enumerate(keys)], schema)
    df.createOrReplaceTempView("null_group_source")
    result = backend.execute_sql(
        privacy_unit="uid",
        params=DPParams(
            contribution_bound=1,
            min_frequency=1,
            sigma_for_thresholding=0,
            tau=1,
            sigmas=[0],
            clipping_thresholds=[None],
        ),
        inner_sql="SELECT * FROM null_group_source",
        agg_columns=[AggregationColumn(Aggregation.COUNT, ["*"], alias="n")],
        group_by_columns=group_by,
        ordering_terms=[],
    )
    return backend, df, group_by, keys, result


@pytest.mark.parametrize("group_kind", ["integer", "string", "multi"])
def test_execute_sql_preserves_null_group_keys(spark, group_kind):
    backend, df, group_by, keys, result = _execute_null_group_query(spark, group_kind)
    result_keys = [key if isinstance(key, tuple) else (key,) for key in result.index]
    normalized = [
        tuple(None if pd.isna(value) else value for value in key) for key in result_keys
    ]

    assert set(normalized) == set(keys)
    assert result["n"].tolist() == [2.0, 2.0]
    assert backend.filter_by_selected_keys(df, group_by, result_keys).count() == 4  # type: ignore


def test_get_column_name(spark):
    backend = SparkSQLBackend(spark)
    spark.sql("DROP DATABASE IF EXISTS db1 CASCADE")
    spark.sql("CREATE DATABASE db1")
    spark.sql("USE db1")
    spark.sql("CREATE TABLE table1 (column1 INT, column2 INT)")
    spark.sql("CREATE TABLE table2 (columnA INT, columnB INT)")

    backend.use_database("db1")
    table_name = backend.get_table_name()
    assert table_name == ["table1", "table2"]

    for table in table_name:
        column_name = backend.get_column_name(table)
        if table == "table1":
            assert column_name == ["column1", "column2"]
        elif table == "table2":
            assert column_name == ["columnA", "columnB"]


def test_get_column_name_rejects_injected_identifier(spark):
    backend = SparkSQLBackend(spark)
    spark.sql("DROP DATABASE IF EXISTS db1 CASCADE")
    spark.sql("CREATE DATABASE db1")
    backend.use_database("db1")
    spark.sql("CREATE TABLE table1 (safe_column INT)")

    with pytest.raises(ExecutionBackendError):
        backend.get_column_name("table1; DROP TABLE table1")

    assert backend.get_column_name("table1") == ["safe_column"]


def test_create_temporary_table(spark):
    backend = SparkSQLBackend(spark)

    df = pd.DataFrame(
        [
            (1, "uid1", "a", 10),
            (2, "uid2", "b", 20),
            (3, "uid3", "a", 30),
            (4, "uid1", "a", 20),
        ],
        columns=["id", "uid", "attribute", "value"],
    )

    backend.create_temporary_table(df, "test_table")

    assert "test_table" in backend.get_table_name()

    assert backend.get_column_name("test_table") == [
        "index",
        "id",
        "uid",
        "attribute",
        "value",
    ]

    backend.create_temporary_table(df, "test_table2", False)

    assert "test_table2" in backend.get_table_name()

    assert backend.get_column_name("test_table2") == [
        "id",
        "uid",
        "attribute",
        "value",
    ]


@pytest.mark.parametrize("arrow_enabled", [False, True])
@pytest.mark.parametrize("all_null", [False, True])
@pytest.mark.parametrize("group_kind", ["integer", "string", "multi"])
def test_create_temporary_table_preserves_null_group_keys(
    spark, arrow_enabled, all_null, group_kind
):
    spark.conf.set(
        "spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower()
    )
    backend, _, group_by, keys, result = _execute_null_group_query(
        spark, group_kind, all_null
    )

    backend.create_temporary_table(result, "saved_null_groups")
    saved = spark.table("saved_null_groups")
    rows = saved.collect()

    assert {tuple(row[name] for name in group_by) for row in rows} == set(keys)
    assert sum(row.n for row in rows) == 4.0
    if group_kind != "string":
        assert isinstance(saved.schema["g"].dataType, LongType)


@pytest.mark.parametrize(
    "key_dtype,spark_type", [("Int64", LongType), ("string", StringType)]
)
@pytest.mark.parametrize("multi", [False, True])
def test_create_temporary_table_preserves_all_null_typed_index(
    spark, key_dtype, spark_type, multi
):
    keys = pd.Index(pd.array([pd.NA], dtype=key_dtype), name="g")
    if multi:
        keys = pd.MultiIndex.from_arrays(
            [keys, pd.array([pd.NA], dtype="string")], names=["g", "h"]
        )
    result = pd.DataFrame({"n": [1.0]}, index=keys)
    backend = SparkSQLBackend(spark)

    backend.create_temporary_table(result, "all_null_typed_index")
    saved = spark.table("all_null_typed_index")

    assert isinstance(saved.schema["g"].dataType, spark_type)
    assert saved.collect()[0].g is None
    if multi:
        assert isinstance(saved.schema["h"].dataType, StringType)
        assert saved.collect()[0].h is None


def test_create_temporary_table_rejects_invalid_name(spark):
    backend = SparkSQLBackend(spark)
    df = pd.DataFrame([(1,)], columns=["id"])

    with pytest.raises(ExecutionBackendError):
        backend.create_temporary_table(df, "x; DROP VIEW hacked", False)


def test_create_temporary_table_rejects_name_collision(spark):
    backend = SparkSQLBackend(spark)
    spark.sql("DROP DATABASE IF EXISTS db1 CASCADE")
    spark.sql("CREATE DATABASE db1")
    backend.use_database("db1")
    spark.sql("CREATE TABLE existing_table (id INT)")
    df = pd.DataFrame([(1,)], columns=["id"])

    with pytest.raises(ExecutionBackendError):
        backend.create_temporary_table(df, "existing_table", False)
