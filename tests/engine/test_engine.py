import tempfile

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType
from pytest_mock import MockerFixture

from dpsql.accountant import Accountant
from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.backend import SparkSQLBackend
from dpsql.dp_params import DPParams
from dpsql.engine import Engine
from dpsql.errors import EngineError, InsufficientPrivacyBudgetError
from dpsql.validator import Validator


def assert_dataframe_equal(df1, df2):
    sorted_df1 = df1.sort(df1.columns)
    sorted_df2 = df2.sort(df2.columns)

    assert sorted_df1.schema == sorted_df2.schema, "Schemas are not equal"

    diff = sorted_df1.subtract(sorted_df2).union(sorted_df2.subtract(sorted_df1))
    assert diff.count() == 0, "DataFrames are not equal"


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


def test_get_db_schema(mocker: MockerFixture, spark):
    # Drop existing databases if they exist
    spark.sql("DROP DATABASE IF EXISTS db1 CASCADE")
    spark.sql("DROP DATABASE IF EXISTS db2 CASCADE")

    # add databases
    spark.sql("CREATE DATABASE db1")
    spark.sql("CREATE DATABASE db2")

    # add tables
    spark.sql("USE db1")
    spark.sql("CREATE TABLE table1 (column1 INT, column2 INT)")
    spark.sql("CREATE TABLE table2 (column1 INT, column2 INT)")

    spark.sql("USE db2")
    spark.sql("CREATE TABLE table3 (column1 INT, column2 INT)")
    spark.sql("CREATE TABLE table4 (column1 INT, column2 INT)")

    # execute the target function
    sql_backend = SparkSQLBackend(spark)
    accountant = mocker.Mock(spec=Accountant)
    engine = Engine(accountant, sql_backend, Validator())

    db_schema1 = engine.get_db_schema("db1")
    db_schema2 = engine.get_db_schema("db2")

    # assert the results
    assert db_schema1 == {
        "table1": ["column1", "column2"],
        "table2": ["column1", "column2"],
    }
    assert db_schema2 == {
        "table3": ["column1", "column2"],
        "table4": ["column1", "column2"],
    }


@pytest.fixture
def setup_engine(mocker: MockerFixture, spark: SparkSession):
    accountant = mocker.Mock(spec=Accountant)
    sql_backend = mocker.Mock(spec=SparkSQLBackend)
    validator = mocker.Mock(spec=Validator)
    engine = Engine(accountant, sql_backend, validator)
    dpparams = mocker.Mock(spec=DPParams)
    dpparams.sigmas = []
    dpparams.noise_amounts = [1]
    return engine, accountant, sql_backend, validator, dpparams, spark


def test_register_database(mocker: MockerFixture, setup_engine):
    engine, _, sql_backend, _, _, _ = setup_engine

    # prepare mocks for successful registration
    database_name = "db"
    privacy_unit_columns = {"table1": "table1_column1"}
    schema_for_db = {
        "table1": ["table1_column1"],
        "table2": ["table2_column1", "table2_column2"],
    }
    mocker.patch.object(engine, "get_db_schema", return_value=schema_for_db)

    # execute the target function
    engine.register_database(database_name, privacy_unit_columns)

    # assert the results
    engine.get_db_schema.assert_called_once_with(database_name)
    assert engine.db_schema[database_name] == schema_for_db
    assert "db.table1" in engine.privacy_unit_columns
    assert engine.privacy_unit_columns["db.table1"] == "table1_column1"


def test_register_database_table_not_found(mocker: MockerFixture, setup_engine):
    engine, _, sql_backend, _, _, _ = setup_engine

    # prepare mocks for table not found
    database_name = "db"
    privacy_unit_columns = {"table1": "table1_column1"}
    schema_for_db = {"table2": ["table2_column1", "table2_column2"]}
    mocker.patch.object(engine, "get_db_schema", return_value=schema_for_db)

    # execute the target function with table not found
    with pytest.raises(EngineError, match="Missing table in database"):
        engine.register_database(database_name, privacy_unit_columns)

    # assert the results
    engine.get_db_schema.assert_called_once_with(database_name)
    assert database_name not in engine.db_schema
    assert database_name not in engine.privacy_unit_columns


def test_register_database_privacy_unit_column_not_found(
    mocker: MockerFixture, setup_engine
):
    engine, _, sql_backend, _, _, _ = setup_engine

    # prepare mocks for privacy unit column not found
    database_name = "db"
    privacy_unit_columns = {"table1": "table1_column1"}
    schema_for_db = {"table1": ["table1_column2"]}
    mocker.patch.object(engine, "get_db_schema", return_value=schema_for_db)

    # execute the target function with privacy unit column not found
    with pytest.raises(
        EngineError,
        match="Missing privacy unit column in table",
    ):
        engine.register_database(database_name, privacy_unit_columns)

    # assert the results
    engine.get_db_schema.assert_called_once_with(database_name)
    assert database_name not in engine.db_schema
    assert database_name not in engine.privacy_unit_columns


def test_execute_query_success(mocker: MockerFixture, setup_engine):
    engine, accountant, sql_backend, validator, dpparams, spark = setup_engine

    # prepare mocks for the registration
    database_name = "db"
    privacy_unit_columns = {"table": "column1"}
    schema_for_db = {"table": ["column1", "column2"]}
    # sql_backend.get_db_schema.return_value = schema_for_db
    mocker.patch.object(engine, "get_db_schema", return_value=schema_for_db)

    # register the database
    engine.register_database(database_name, privacy_unit_columns)

    # prepare mocks for successful execution
    query = "SELECT COUNT(*) FROM db.table"
    intermediate_privacy_unit = "privacy_unit"
    schema = StructType([StructField("column", IntegerType(), True)])
    data = [(1,), (2,), (3,)]
    expected_result = spark.createDataFrame(data, schema)

    # Create AggregationColumn objects for final_result_columns
    agg_column = AggregationColumn(
        aggregation_type=Aggregation.COUNT, columns=["*"], alias=None, parameters=[]
    )

    validator.validate_and_get_final_select_items.return_value = (
        intermediate_privacy_unit,
        "SELECT * FROM db.table",  # inner_sql
        [agg_column],  # final_result_columns
        [],  # group_by_columns
        [],  # ordering_terms
        None,  # limit
        None,  # offset
        None,
    )
    accountant.check_budget.return_value = True
    sql_backend.execute_sql.return_value = expected_result

    # execute the target function
    result = engine.execute_query(query, dpparams)

    # Extract aggregation functions for budget checking
    agg_funcs = [Aggregation.COUNT]

    # assert the results
    validator.validate_and_get_final_select_items.assert_called_once_with(
        query, engine.db_schema, engine.privacy_unit_columns
    )
    accountant.check_budget.assert_called_once_with(agg_funcs, dpparams)
    sql_backend.execute_sql.assert_called_once_with(
        intermediate_privacy_unit,
        dpparams,
        "SELECT * FROM db.table",
        [agg_column],
        [],
        [],
        None,
        None,
    )
    accountant.update_budget.assert_called_once_with(agg_funcs, dpparams)
    assert_dataframe_equal(result, expected_result)


def test_execute_query_validation_failure(setup_engine):
    engine, _, _, validator, dpparams, _ = setup_engine

    # prepare mocks for validation failure on the validator
    query = ""
    validator.validate_and_get_final_select_items.side_effect = ValueError(
        "Validation error"
    )

    # execute the target function with validation failure
    with pytest.raises(ValueError, match="Validation error"):
        engine.execute_query(query, dpparams)

    # assert the results
    validator.validate_and_get_final_select_items.assert_called_once_with(query, {}, {})


def test_execute_query_budget_exceeded(setup_engine):
    engine, accountant, sql_backend, validator, dpparams, _ = setup_engine

    # prepare mocks for budget exceeded
    query = "SELECT COUNT(*) FROM db.table"
    intermediate_privacy_unit = ""

    # Create AggregationColumn object for final_result_columns
    agg_column = AggregationColumn(
        aggregation_type=Aggregation.COUNT, columns=["*"], alias=None, parameters=[]
    )

    validator.validate_and_get_final_select_items.return_value = (
        intermediate_privacy_unit,
        "SELECT * FROM db.table",  # inner_sql
        [agg_column],  # final_result_columns
        [],  # group_by_columns
        [],  # ordering_terms
        None,  # limit
        None,  # offset
        None,
    )
    accountant.check_budget.return_value = False
    accountant.epsilon = 1.0
    accountant.delta = 0.1

    # execute the target function
    with pytest.raises(InsufficientPrivacyBudgetError, match="Privacy budget exceeded"):
        engine.execute_query(query, dpparams)

    # Extract aggregation functions for budget checking
    agg_funcs = [Aggregation.COUNT]

    # assert the results
    validator.validate_and_get_final_select_items.assert_called_once_with(query, {}, {})
    accountant.check_budget.assert_called_once_with(agg_funcs, dpparams)
    accountant.update_budget.assert_not_called()
    sql_backend.execute_sql.assert_not_called()
