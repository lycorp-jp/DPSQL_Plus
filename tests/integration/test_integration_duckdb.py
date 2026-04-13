import duckdb
import pandas as pd
import pytest

import dpsql.backend.secure_sampling as secure_sampling
from dpsql.accountant import RenyiAccountant
from dpsql.backend import DuckDBBackend
from dpsql.dp_params import DPParams
from dpsql.engine import Engine
from dpsql.validator import Validator


@pytest.fixture(scope="function")
def duckdb_singlecolumn():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE table1 (column1 INT, column2 INT)")

    # Insert 10000 records in batches
    values = ", ".join([f"({i}, {i % 5})" for i in range(1, 10001)])
    conn.execute(f"INSERT INTO table1 VALUES {values}")
    conn.commit()

    yield conn
    conn.close()


@pytest.fixture(scope="function")
def duckdb_multicolumn():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE table1 (column1 INT, column2 INT, column3 INT)")

    # Insert 10000 records in batches
    values = ", ".join(
        [f"({i * 100 + j}, {i % 5}, {j % 5})" for i in range(100) for j in range(100)]
    )
    conn.execute(f"INSERT INTO table1 VALUES {values}")
    conn.commit()

    yield conn
    conn.close()


def test_duckdb_execute_query(duckdb_singlecolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = DuckDBBackend(duckdb_singlecolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER from table1) SELECT COUNT(USER) FROM tbl"
    )
    dpparams = DPParams(
        contribution_bound=1,
        tau=1,
        sigma_for_thresholding=1.0,
        sigmas=[1.0],
        clipping_thresholds=[None],
    )

    result_df = engine.execute_query(query, dpparams)

    # check the result
    expected_result = [(secure_sampling.secure_gauss(10000, 1.0),)]
    result = [tuple(row) for _, row in result_df.iterrows()]
    assert result == expected_result


def test_duckdb_execute_query_with_temp_table(duckdb_multicolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(float("inf"), 1.0)
    sql_backend = DuckDBBackend(duckdb_multicolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2, column3 from table1) "
        "SELECT column2, column3, COUNT(USER) FROM tbl GROUP BY column2, column3"
    )
    dpparams = DPParams(
        contribution_bound=1,
        tau=1,
        sigma_for_thresholding=1.0,
        sigmas=[1.0],
        clipping_thresholds=[None],
    )

    engine.execute_query(query, dpparams, "temp_table")

    schema = engine.get_db_schema(None)

    assert "temp_table" in schema

    assert ["column2", "column3", "count(USER)"] == schema["temp_table"]

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2, column3 from table1) "
        "SELECT COUNT(USER) FROM tbl"
    )
    dpparams = DPParams(
        contribution_bound=1,
        tau=1,
        sigma_for_thresholding=1.0,
        sigmas=[1.0],
        clipping_thresholds=[None],
    )

    engine.execute_query(query, dpparams, "temp_table2")

    schema = engine.get_db_schema(None)

    assert "temp_table2" in schema

    assert ["index", "count(USER)"] == schema["temp_table2"]


def test_duckdb_execute_query_with_multiple_columns(duckdb_multicolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(float("inf"), 1.0)
    sql_backend = DuckDBBackend(duckdb_multicolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = "SELECT COVAR(column2, column3) from table1"
    dpparams = DPParams(
        contribution_bound=1,
        tau=1,
        sigma_for_thresholding=1.0,
        sigmas=[0.01],  # Small noise
        clipping_thresholds=[[(0.0, 5.0), (0.0, 5.0)]],
    )

    result_df = engine.execute_query(query, dpparams, "temp_table")
    expected_df = pd.DataFrame([[0.0]], columns=["covar_samp(column2,column3)"])
    pd.testing.assert_frame_equal(result_df, expected_df, atol=1e-6)
