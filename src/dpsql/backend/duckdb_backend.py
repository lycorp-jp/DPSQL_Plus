import logging

import duckdb
import pandas as pd
import polars as pl

from ..aggregation import Aggregation
from ..dp_params import DPParams
from ..errors import (
    AggregationError,
    ExecutionBackendError,
    UnsupportedBackendError,
)
from ..utils import safely_get_threshold
from .sql_backend import DataFrameLike, SQLBackend

logger = logging.getLogger(__name__)


class DuckDBBackend(SQLBackend):
    """
    Backend for executing DuckDB queries with differential privacy mechanisms.

    Args:
        conn (duckdb.DuckDBPyConnection): A live DuckDB connection object.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        logger.debug(
            "DuckDBBackend initialized with connection=%s", type(conn).__name__
        )

    def create_inner_df(self, inner_sql: str) -> pl.DataFrame:
        logger.debug("DuckDB create_inner_df: sql_head=%s", inner_sql[:160])
        try:
            return self.conn.execute(inner_sql).pl()
        except Exception as e:  # noqa
            raise ExecutionBackendError(
                "Failed to execute DuckDB SQL",
                context={"sql_head": inner_sql[:160]},
                hint="Check SQL syntax and referenced tables",
                cause=e,
            ) from e

    def contribution_bound(
        self, inner_df: DataFrameLike, privacy_unit: str, params: DPParams
    ) -> pl.DataFrame:
        logger.debug(
            "DuckDB contribution_bound: privacy_unit=%s bound=%s empty=%s",
            privacy_unit,
            params.contribution_bound,
            getattr(inner_df, "is_empty", lambda: None)(),
        )
        if not isinstance(inner_df, pl.DataFrame):
            raise UnsupportedBackendError(
                "Expected Polars DataFrame",
                context={"actual_type": type(inner_df).__name__},
                hint="Provide a polars.DataFrame",
            )

        # Early return for empty DataFrame to avoid errors
        if inner_df.is_empty():
            return inner_df

        # Polars treats NA values as the key in groups by default
        # Generate random rank within each group and filter by contribution_bound
        filtered_df = (
            inner_df.with_columns(
                pl.int_range(0, pl.len())
                .shuffle()
                .over(privacy_unit)
                .alias("_sample_rank_")
            )
            .filter(pl.col("_sample_rank_") < params.contribution_bound)
            .drop("_sample_rank_")
        )
        logger.debug("DuckDB contribution_bound applied")
        return filtered_df

    def apply_aggregation(
        self,
        agg_type: Aggregation,
        column_name: list[str],
        df: DataFrameLike,
        group_by: list[str],
        clipping_threshold: list[tuple[float, float]] | None = None,
    ) -> pd.Series:
        logger.debug(
            "DuckDB apply_aggregation: agg=%s columns=%s group_by=%s clip=%s",
            getattr(agg_type, "name", str(agg_type)),
            column_name,
            group_by,
            clipping_threshold,
        )
        if not isinstance(df, pl.DataFrame):
            raise UnsupportedBackendError(
                "df must be a Polars DataFrame",
                context={"actual_type": type(df).__name__},
                hint="Convert the input to polars.DataFrame before aggregation",
            )
        # Add a dummy column to group by if group_by is empty
        df = df.with_columns(pl.lit(1).alias("_no_group_key_"))

        # Polars treats NA values as the key in groups by default
        grouped = df.group_by(group_by) if group_by else df.group_by("_no_group_key_")

        # Use alias to avoid column name clashes
        if agg_type == Aggregation.COUNT:
            res = (
                grouped.len()
                if column_name[0] == "*"
                else grouped.agg(
                    pl.col(column_name[0]).alias(f"_{column_name[0]}_").count()
                )
            )
        elif agg_type == Aggregation.COUNT_DISTINCT:
            res = grouped.agg(
                pl.col(column_name[0]).alias(f"_{column_name[0]}_").n_unique()
            )
        elif agg_type == Aggregation.SUM:
            if clipping_threshold is None:
                raise AggregationError(
                    "Missing `clipping_threshold` for `SUM` aggregation",
                    context={"agg_type": agg_type.name, "column": column_name[0]},
                    hint="Provide a list of (lower_bound, upper_bound) tuples",
                )
            lower_bound, upper_bound = safely_get_threshold(
                clipping_threshold, 0, agg_type.name
            )
            res = grouped.agg(
                pl.col(column_name[0])
                .alias(f"_{column_name[0]}_")
                .clip(lower_bound, upper_bound)
                .sum()
            )
        elif agg_type == Aggregation.SQUARED_SUM:
            if clipping_threshold is None:
                raise AggregationError(
                    "Missing `clipping_threshold` for `SQUARED_SUM` aggregation",
                    context={"agg_type": agg_type.name, "column": column_name[0]},
                    hint="Provide a list of (lower_bound, upper_bound) tuples",
                )
            lower_bound, upper_bound = safely_get_threshold(
                clipping_threshold, 0, agg_type.name
            )
            res = grouped.agg(
                (
                    pl.col(column_name[0])
                    .alias(f"_{column_name[0]}_")
                    .clip(lower_bound, upper_bound)
                    ** 2
                ).sum()
            )
        elif agg_type == Aggregation.PRODUCT_SUM:
            lower_bound_x, upper_bound_x = safely_get_threshold(
                clipping_threshold, 0, agg_type.name
            )
            lower_bound_y, upper_bound_y = safely_get_threshold(
                clipping_threshold, 1, agg_type.name
            )
            # Calculate the product of the first two columns
            res = grouped.agg(
                (
                    pl.col(column_name[0])
                    .alias(f"_{column_name[0]}_")
                    .clip(lower_bound_x, upper_bound_x)
                    * pl.col(column_name[1])
                    .alias(f"_{column_name[1]}_")
                    .clip(lower_bound_y, upper_bound_y)
                ).sum()
            )
        else:
            raise AggregationError(
                "Unsupported aggregation type",
                context={"agg_type": getattr(agg_type, "name", str(agg_type))},
                hint="Use COUNT, COUNT_DISTINCT, SUM, SQUARED_SUM, PRODUCT_SUM",
            )
        pd_res = res.to_pandas()  # convert Polars DataFrame to Pandas DataFrame

        # Convert to Pandas Series (If not group_by, [:, 0] is a dummy column)
        logger.debug("DuckDB aggregation result converted to pandas Series")
        return pd_res.set_index(group_by).iloc[:, 0] if group_by else pd_res.iloc[:, 1]

    def use_database(self, database_name: str | None) -> None:
        logger.info("DuckDB use_database: %s", database_name)
        """
        In DuckDB, "USE database" is not directly supported as in MySQL or Spark.
        If you need to switch to a new database file, you can create a new
        connection or attach the database. By default, we do nothing here.
        """
        if database_name is not None:
            raise UnsupportedBackendError(
                "Unsupported database switch in DuckDB backend",
                context={"requested_database": database_name},
                hint="Open a new connection or ATTACH another database file",
            )

    def get_table_name(self) -> list[str]:
        logger.debug("DuckDB get_table_name")
        tables = self.conn.execute("SHOW TABLES").fetchall()
        return [table[0] for table in tables]

    def filter_by_selected_keys(
        self,
        df: DataFrameLike,
        group_by: list[str],
        selected_keys: list[tuple[str, ...]],
    ) -> pl.DataFrame:
        logger.debug(
            "DuckDB filter_by_selected_keys: group_by=%s selected_keys_count=%s",
            group_by,
            len(selected_keys),
        )
        if not isinstance(df, pl.DataFrame):
            raise UnsupportedBackendError(
                "df must be a Polars DataFrame",
                context={"actual_type": type(df).__name__},
                hint="Ensure the input is polars.DataFrame",
            )

        if len(group_by) == 0:
            # If no group by columns, return the original Table
            return df
        elif len(selected_keys) == 0:
            # If selected_keys is empty, return an empty DataFrame with the same schema
            return pl.DataFrame(schema=df.schema)
        else:
            ref_df = pl.DataFrame(selected_keys, schema=group_by, orient="row")
            filtered_df = df.join(ref_df, on=group_by, how="inner")
            logger.debug("DuckDB keys filtered")
            return filtered_df

    def get_column_name(self, table_name: str) -> list[str]:
        logger.debug("DuckDB get_column_name: table=%s", table_name)
        columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [column[0] for column in columns]

    def create_temporary_table(
        self, df: pd.DataFrame, table_name: str, index: bool = True
    ) -> None:
        logger.info(
            "DuckDB create_temporary_table: table=%s index=%s", table_name, index
        )
        temp_df = df.reset_index() if index else df
        self.conn.register(table_name, temp_df)
