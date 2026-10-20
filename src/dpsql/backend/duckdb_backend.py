import logging
import re

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

_SAFE_TEMP_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str) -> str:
    quoted_parts: list[str] = []
    for identifier_part in identifier.split("."):
        escaped_part = identifier_part.replace('"', '""')
        quoted_parts.append(f'"{escaped_part}"')
    return ".".join(quoted_parts)


def _clipped_column(column_name: str, bounds: tuple[float, float]) -> pl.Expr:
    lower_bound, upper_bound = bounds
    column = pl.col(column_name).cast(pl.Float64)
    return column.fill_nan(upper_bound).clip(lower_bound, upper_bound)


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

    def prepare_inner_df(
        self,
        inner_df: DataFrameLike,
        privacy_unit: str,
        group_by: list[str],
    ) -> pl.DataFrame:
        if not isinstance(inner_df, pl.DataFrame):
            raise UnsupportedBackendError(
                "Expected Polars DataFrame",
                context={"actual_type": type(inner_df).__name__},
                hint="Provide a polars.DataFrame",
            )
        for column in [privacy_unit, *group_by]:
            if inner_df.schema[column].is_float():
                raise ExecutionBackendError(
                    "Privacy unit and GROUP BY columns cannot support NaN or infinity",
                    context={"column": column, "dtype": str(inner_df.schema[column])},
                    hint="Use a non-floating-point column for privacy units and groups",
                )
        return inner_df.drop_nulls(subset=[privacy_unit])

    def contribution_bound(
        self,
        inner_df: DataFrameLike,
        privacy_unit: str,
        params: DPParams,
        priority_column: str,
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

        # Rank contributions by the shared random priority within each privacy unit.
        filtered_df = inner_df.filter(
            pl.col(priority_column).rank(method="ordinal").over(privacy_unit)
            <= params.contribution_bound
        )
        logger.debug("DuckDB contribution_bound applied")
        return filtered_df

    def add_random_priority(self, df: DataFrameLike, column_name: str) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise UnsupportedBackendError(
                "Expected Polars DataFrame",
                context={"actual_type": type(df).__name__},
                hint="Provide a polars.DataFrame",
            )
        return df.with_columns(pl.int_range(0, pl.len()).shuffle().alias(column_name))

    def prepare_covar_df(
        self, df: DataFrameLike, columns: list[str]
    ) -> tuple[pl.DataFrame, list[str]]:
        if not isinstance(df, pl.DataFrame):
            raise UnsupportedBackendError("Expected Polars DataFrame")
        x, y = columns
        masked_x = self._unused_column_name(df, "_dpsql_covar_x_")
        masked_y = self._unused_column_name(df, "_dpsql_covar_y_")
        valid = pl.col(x).is_not_null() & pl.col(y).is_not_null()
        result = df.with_columns(
            pl.when(valid).then(pl.col(x)).alias(masked_x),
            pl.when(valid).then(pl.col(y)).alias(masked_y),
        )
        return result, [masked_x, masked_y]

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
        aggregation_keys = group_by
        if not aggregation_keys:
            dummy_column = self._unused_column_name(df, "_no_group_key_")
            df = df.with_columns(pl.lit(1).alias(dummy_column))
            aggregation_keys = [dummy_column]

        # Polars treats NA values as the key in groups by default
        grouped = df.group_by(aggregation_keys)

        if agg_type == Aggregation.COUNT:
            aggregation = (
                pl.len() if column_name[0] == "*" else pl.col(column_name[0]).count()
            )
        elif agg_type == Aggregation.COUNT_DISTINCT:
            aggregation = pl.col(column_name[0]).drop_nulls().n_unique()
        elif agg_type in (Aggregation.SUM, Aggregation.SQUARED_SUM):
            column = _clipped_column(
                column_name[0],
                safely_get_threshold(clipping_threshold, 0, agg_type.name),
            )
            if agg_type == Aggregation.SQUARED_SUM:
                column = column**2
            aggregation = column.sum()
        elif agg_type == Aggregation.PRODUCT_SUM:
            # Calculate the product of the first two columns
            aggregation = (
                _clipped_column(
                    column_name[0],
                    safely_get_threshold(clipping_threshold, 0, agg_type.name),
                )
                * _clipped_column(
                    column_name[1],
                    safely_get_threshold(clipping_threshold, 1, agg_type.name),
                )
            ).sum()
        else:
            raise AggregationError(
                "Unsupported aggregation type",
                context={"agg_type": getattr(agg_type, "name", str(agg_type))},
                hint="Use COUNT, COUNT_DISTINCT, SUM, SQUARED_SUM, PRODUCT_SUM",
            )
        result_column = self._unused_column_name(df, "_dpsql_aggregation_")
        res = grouped.agg(aggregation.alias(result_column))
        pd_res = res.to_pandas()  # convert Polars DataFrame to Pandas DataFrame
        for group_column in group_by:
            if res.schema[group_column].is_integer() and res[group_column].null_count():
                # Keep nullable integers exact, including in pandas MultiIndex.
                pd_res[group_column] = res[group_column].to_pandas(
                    use_pyarrow_extension_array=True
                )

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

        if len(selected_keys) == 0:
            # If no selected keys, return an empty DataFrame with the same schema
            return pl.DataFrame(schema=df.schema)
        if len(group_by) == 0:
            # If no group by columns, return the original Table
            return df
        else:
            ref_df = pl.DataFrame(
                [
                    tuple(None if pd.isna(key) else key for key in keys)
                    for keys in selected_keys
                ],
                schema=df.select(group_by).schema,
                orient="row",
                strict=False,
            )
            filtered_df = df.join(ref_df, on=group_by, how="inner", nulls_equal=True)
            logger.debug("DuckDB keys filtered")
            return filtered_df

    def get_column_name(self, table_name: str) -> list[str]:
        logger.debug("DuckDB get_column_name: table=%s", table_name)
        try:
            columns = self.conn.execute(
                f"DESCRIBE {_quote_identifier(table_name)}"
            ).fetchall()
        except Exception as e:  # noqa
            raise ExecutionBackendError(
                "Failed to describe table",
                context={"table": table_name},
                hint="Verify the relation name exists and is a valid identifier",
                cause=e,
            ) from e
        return [column[0] for column in columns]

    def create_temporary_table(
        self, df: pd.DataFrame, table_name: str, index: bool = True
    ) -> None:
        logger.info(
            "DuckDB create_temporary_table: table=%s index=%s", table_name, index
        )
        if not _SAFE_TEMP_TABLE_NAME.fullmatch(table_name):
            raise ExecutionBackendError(
                "Invalid temporary table name",
                context={"table_name": table_name},
                hint=(
                    "Use only letters, numbers, and underscores, and start "
                    "with a letter or underscore"
                ),
            )
        if table_name in self.get_table_name():
            raise ExecutionBackendError(
                "Temporary table name already exists",
                context={"table_name": table_name},
                hint=(
                    "Choose a new temporary table name that does not collide "
                    "with an existing relation"
                ),
            )
        temp_df = df.reset_index() if index else df
        try:
            self.conn.register(table_name, temp_df)
        except Exception as e:  # noqa
            raise ExecutionBackendError(
                "Failed to create temporary table",
                context={"table_name": table_name, "index_included": index},
                hint="Check the DataFrame schema and temporary table name",
                cause=e,
            ) from e
