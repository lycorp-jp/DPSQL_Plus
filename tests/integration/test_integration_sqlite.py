import sqlite3

import pandas as pd
import pytest

import dpsql.backend.secure_sampling as secure_sampling
from dpsql.accountant import RenyiAccountant
from dpsql.backend import SQLiteBackend
from dpsql.dp_params import DPParams
from dpsql.engine import Engine
from dpsql.validator import Validator


@pytest.fixture(scope="function")
def sqlite_singlecolumn():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE table1 (column1 INT, column2 INT)")
    cursor.execute("CREATE TABLE table2 (column3 INT, column4 INT)")

    cursor.execute("CREATE TABLE table3 (column5 INT, column6 INT)")
    cursor.execute("CREATE TABLE table4 (column7 INT, column8 INT)")

    # Insert 10000 records in batches
    values = ", ".join([f"({i}, {i % 5})" for i in range(1, 10001)])
    cursor.execute(f"INSERT INTO table1 VALUES {values}")
    cursor.execute(f"INSERT INTO table3 VALUES {values}")
    conn.commit()

    yield conn
    conn.close()


@pytest.fixture(scope="function")
def sqlite_multicolumn():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE table1 (column1 INT, column2 INT, column3 INT)")

    # Insert 10000 records in batches
    values = ", ".join(
        [f"({i * 100 + j}, {i % 5}, {j % 5})" for i in range(100) for j in range(100)]
    )
    cursor.execute(f"INSERT INTO table1 VALUES {values}")
    conn.commit()

    yield conn
    conn.close()


def test_sqlite_execute_query(sqlite_singlecolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_singlecolumn)
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


@pytest.mark.parametrize(
    "order_clause,expected_indices",
    [
        ("ORDER BY column2", [0, 1, 2, 3, 4]),
        ("ORDER BY column2 DESC", [4, 3, 2, 1, 0]),
    ],
    ids=["asc", "desc"],
)
def test_sqlite_execute_query_order_by(
    sqlite_singlecolumn, order_clause, expected_indices
):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_singlecolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2 from table1) "
        f"SELECT column2, COUNT(USER) FROM tbl GROUP BY column2 {order_clause}"
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
    expected_result = [(secure_sampling.secure_gauss(2000, 1.0),) for _ in range(5)]
    result = [tuple(row) for _, row in result_df.iterrows()]
    assert result == expected_result
    assert result_df.index.values.tolist() == expected_indices


@pytest.mark.parametrize(
    "order_clause,expected_indices",
    [
        (
            "ORDER BY column2, column3 DESC",
            [(i, j) for i in range(5) for j in range(4, -1, -1)],
        ),
        (
            "ORDER BY column2 DESC, column3 ASC",
            [(i, j) for i in range(4, -1, -1) for j in range(5)],
        ),
        (
            "ORDER BY column2 DESC, column3 DESC",
            [(i, j) for i in range(4, -1, -1) for j in range(4, -1, -1)],
        ),
        (
            "ORDER BY column2, column3 ASC",
            [(i, j) for i in range(5) for j in range(5)],
        ),
    ],
    ids=["asc_desc", "desc_asc", "desc_desc", "asc_asc"],
)
def test_sqlite_execute_query_order_by_multi_column(
    sqlite_multicolumn, order_clause, expected_indices
):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_multicolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2, column3 from table1) "
        "SELECT column2, column3, COUNT(USER) FROM tbl GROUP BY column2, column3 "
        f"{order_clause}"
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
    expected_result = [(secure_sampling.secure_gauss(400, 1.0),) for _ in range(5 * 5)]
    result = [tuple(row) for _, row in result_df.iterrows()]
    assert result == expected_result
    assert result_df.index.values.tolist() == expected_indices


def test_sqlite_execute_query_order_by_limit(sqlite_singlecolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_singlecolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2 from table1) "
        "SELECT column2, COUNT(USER) FROM tbl GROUP BY column2 "
        "ORDER BY column2 DESC LIMIT 3"
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
    expected_result = [(secure_sampling.secure_gauss(2000, 1.0),) for _ in range(3)]
    result = [tuple(row) for _, row in result_df.iterrows()]
    assert result == expected_result
    assert result_df.index.values.tolist() == [
        4,
        3,
        2,
    ]  # descending order by column2 with limit 3


def test_sqlite_execute_query_order_by_limit_offset(sqlite_singlecolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_singlecolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2 from table1) "
        "SELECT column2, COUNT(USER) AS d FROM tbl GROUP BY column2 "
        "ORDER BY column2 DESC, d DESC LIMIT 3 OFFSET 2"
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
    expected_result = [(secure_sampling.secure_gauss(2000, 1.0),) for _ in range(3)]
    result = [tuple(row) for _, row in result_df.iterrows()]
    assert result == expected_result
    assert result_df.index.values.tolist() == [
        2,
        1,
        0,
    ]  # descending order by column2 with limit 3 and offset 2


def test_sqlite_execute_query_with_temp_table(sqlite_multicolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(float("inf"), 1.0)
    sql_backend = SQLiteBackend(sqlite_multicolumn)
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

    engine.execute_query(query, dpparams, "temp.temp_table")

    schema = engine.get_db_schema(None)

    assert "temp.temp_table" in schema

    assert ["column2", "column3", "count(USER)"] == schema["temp.temp_table"]

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

    engine.execute_query(query, dpparams, "temp.temp_table2")

    schema = engine.get_db_schema(None)

    assert "temp.temp_table2" in schema

    assert ["index", "count(USER)"] == schema["temp.temp_table2"]


def test_sqlite_execute_query_for_empty_result(sqlite_singlecolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(100.0, 1.0)
    sql_backend = SQLiteBackend(sqlite_singlecolumn)
    validator = Validator()

    engine = Engine(accountant, sql_backend, validator)

    # register databases
    engine.register_database(None, privacy_unit_columns={"table1": "column1"})

    query = (
        "WITH tbl AS (SELECT column1 AS USER, column2 from table1) "
        "SELECT column2, COUNT(USER) AS d FROM tbl GROUP BY column2 "
        "ORDER BY column2 DESC, d DESC LIMIT 3 OFFSET 2"
    )
    dpparams = DPParams(
        contribution_bound=1,
        tau=10000,
        sigma_for_thresholding=1.0,
        sigmas=[1.0],
        clipping_thresholds=[None],
    )

    result_df = engine.execute_query(query, dpparams)

    # check the result
    assert result_df.empty


def test_sqlite_execute_query_with_multiple_columns(sqlite_multicolumn):
    secure_sampling.SAMPLING_MODE = secure_sampling.SamplingMode.DEBUG

    accountant = RenyiAccountant(float("inf"), 1.0)
    sql_backend = SQLiteBackend(sqlite_multicolumn)
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

    result_df = engine.execute_query(query, dpparams)
    expected_df = pd.DataFrame([[0.0]], columns=["covar_samp(column2,column3)"])
    pd.testing.assert_frame_equal(result_df, expected_df, atol=1e-6)
