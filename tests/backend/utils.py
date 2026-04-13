"""Test utilities for backend testing."""

from typing import Any

import pandas as pd
import polars as pl
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from dpsql.aggregation import Aggregation


def create_test_cases(backend_type: str) -> tuple[list[tuple[Any, ...]], list[str]]:
    """
    Create test cases for apply_aggregation tests.

    Args:
        backend_type: One of 'sqlite', 'duckdb', or 'spark'

    Returns:
        List of test case tuples
    """
    base_test_cases = [
        # (agg_type, column_name, data, group_by,
        #  clipping_threshold, expected_result, test_id)
        (
            Aggregation.COUNT,
            ["*"],
            {
                "data": [
                    (1, "A", 10.0),
                    (2, "B", 20.0),
                    (3, "A", 30.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            None,
            pd.Series([2, 1], index=["A", "B"]),
            "count_with_star",
        ),
        (
            Aggregation.COUNT,
            ["id"],
            {
                "data": [
                    (1, "A", 10.0),
                    (2, "B", 20.0),
                    (3, "A", 30.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            None,
            pd.Series([2, 1], index=["A", "B"]),
            "count_with_column",
        ),
        (
            Aggregation.COUNT_DISTINCT,
            ["value"],
            {
                "data": [
                    (1, "A", 10.0),
                    (2, "B", 20.0),
                    (3, "A", 10.0),
                    (4, "A", 30.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            None,
            pd.Series([2, 1], index=["A", "B"]),
            "count_distinct",
        ),
        (
            Aggregation.SUM,
            ["value"],
            {
                "data": [
                    (1, "A", 10.0),
                    (2, "B", 20.0),
                    (3, "A", 15.0),
                    (4, "A", 25.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            [(5.0, 50.0)],
            pd.Series([50.0, 20.0], index=["A", "B"]),
            "sum_basic",
        ),
        (
            Aggregation.SUM,
            ["value"],
            {
                "data": [
                    (1, "A", 1.0),
                    (2, "A", 100.0),
                    (3, "B", 200.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            [(10.0, 50.0)],
            pd.Series([60.0, 50.0], index=["A", "B"]),
            "sum_with_clipping",
        ),
        (
            Aggregation.SQUARED_SUM,
            ["value"],
            {
                "data": [
                    (1, "A", 3.0),
                    (2, "A", 4.0),
                    (3, "B", 5.0),
                ],
                "columns": ["id", "category", "value"],
            },
            ["category"],
            [(1.0, 10.0)],
            pd.Series([25.0, 25.0], index=["A", "B"]),
            "squared_sum",
        ),
        (
            Aggregation.PRODUCT_SUM,
            ["value1", "value2"],
            {
                "data": [
                    (1, "A", 2.0, 3.0),
                    (2, "A", 4.0, 5.0),
                    (3, "B", 6.0, 7.0),
                ],
                "columns": ["id", "category", "value1", "value2"],
            },
            ["category"],
            [(1.0, 5.0), (2.0, 6.0)],
            pd.Series([26.0, 30.0], index=["A", "B"]),
            "product_sum",
        ),
        (
            Aggregation.COUNT,
            ["*"],
            {
                "data": [
                    (1, 10.0),
                    (2, 20.0),
                    (3, 30.0),
                ],
                "columns": ["id", "value"],
            },
            [],
            None,
            pd.Series([3], index=[0]),
            "no_group_by",
        ),
    ]

    # Convert data based on backend type
    converted_cases: list[tuple[Any, ...]] = []
    for case in base_test_cases:
        (
            agg_type,
            column_name,
            data_dict,
            group_by,
            clipping_threshold,
            expected_result,
            test_id,
        ) = case

        if backend_type == "sqlite":
            # SQLite uses pandas DataFrame
            df = pd.DataFrame(data_dict["data"], columns=data_dict["columns"])
            converted_cases.append(
                (
                    agg_type,
                    column_name,
                    df,
                    group_by,
                    clipping_threshold,
                    expected_result,
                )
            )

        elif backend_type == "duckdb":
            # DuckDB uses polars DataFrame
            columns: list[str] = data_dict["columns"]  # type: ignore
            data = data_dict["data"]
            df = pl.DataFrame(data, schema=columns, orient="row")
            converted_cases.append(
                (
                    agg_type,
                    column_name,
                    df,
                    group_by,
                    clipping_threshold,
                    expected_result,
                )
            )

        elif backend_type == "spark":
            # Spark uses raw data + schema
            data = data_dict["data"]
            columns: list[str] = data_dict["columns"]  # type: ignore

            schema_list = []
            for col in columns:
                if col == "id":
                    schema_list.append(StructField(col, IntegerType(), True))
                elif "value" in col:
                    schema_list.append(StructField(col, DoubleType(), True))
                else:
                    schema_list.append(StructField(col, StringType(), True))
            schema = StructType(schema_list)

            converted_cases.append(
                (
                    agg_type,
                    column_name,
                    data,
                    schema,
                    group_by,
                    clipping_threshold,
                    expected_result,
                )
            )

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    # Get test IDs
    test_ids = [case[6] for case in base_test_cases]

    return converted_cases, test_ids
