import tempfile

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from utils import create_test_cases

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

    filtered_df = backend.contribution_bound(df, "uid", params)
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
    ],
    ids=[
        "single column filtering",
        "multi column filtering",
        "empty selected_keys case",
    ],
)
def test_filter_by_selected_keys(
    spark, data, schema, group_by, selected_keys, expected_count
):
    backend = SparkSQLBackend(spark)
    df = backend.spark.createDataFrame(data, schema)
    filtered_df = backend.filter_by_selected_keys(df, group_by, selected_keys)
    assert filtered_df.count() == expected_count


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
