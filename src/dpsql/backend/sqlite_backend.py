import logging
import re
import sqlite3

import numpy as np
import pandas as pd

from ..aggregation import Aggregation
from ..dp_params import DPParams
from ..errors import AggregationError, ExecutionBackendError, UnsupportedBackendError
from ..utils import safely_get_threshold
from .sql_backend import DataFrameLike, SQLBackend

logger = logging.getLogger(__name__)


def _clip_values(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values.astype(float).clip(lower=lower, upper=upper)


def _quote_pragma_literal(identifier: str) -> str:
    escaped_identifier = identifier.replace("'", "''")
    return f"'{escaped_identifier}'"


class SQLiteBackend(SQLBackend):
    """
    Backend for executing SQLite queries with differential privacy mechanisms.

    Args:
        conn (sqlite3.Connection): A live SQLite connection object.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        logger.debug(
            "SQLiteBackend initialized with connection=%s", type(conn).__name__
        )

    def create_inner_df(self, inner_sql: str) -> pd.DataFrame:
        logger.debug("SQLite create_inner_df: sql_head=%s", inner_sql[:160])
        return pd.read_sql_query(inner_sql, self.conn)

    def prepare_inner_df(
        self,
        inner_df: DataFrameLike,
        privacy_unit: str,
        group_by: list[str],
    ) -> pd.DataFrame:
        if not isinstance(inner_df, pd.DataFrame):
            raise UnsupportedBackendError("Expected pandas DataFrame in SQLite backend")
        # SQLite uses dynamic typing, so privacy unit and GROUP BY column types
        # are not validated here.
        return inner_df.dropna(subset=[privacy_unit])

    def contribution_bound(
        self,
        inner_df: DataFrameLike,
        privacy_unit: str,
        params: DPParams,
        priority_column: str,
    ) -> pd.DataFrame:
        logger.debug(
            "SQLite contribution_bound: privacy_unit=%s, bound=%s",
            privacy_unit,
            params.contribution_bound,
        )
        if not isinstance(inner_df, pd.DataFrame):
            raise UnsupportedBackendError(
                "Expected pandas DataFrame in SQLite backend",
                context={"actual_type": type(inner_df).__name__},
                hint="Provide a pandas.DataFrame",
            )

        # Rank contributions by the shared random priority within each privacy unit.
        randomized = inner_df.sort_values(by=priority_column)
        grouped = randomized.groupby(
            privacy_unit, dropna=False, group_keys=False, sort=False
        )
        filtered_df = randomized[grouped.cumcount() < params.contribution_bound]
        filtered_df = filtered_df.reset_index(drop=True)
        logger.debug(
            "SQLite contribution_bound: filtered_rows=%s", filtered_df.shape[0]
        )
        return filtered_df

    def add_random_priority(self, df: DataFrameLike, column_name: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise UnsupportedBackendError(
                "Expected pandas DataFrame in SQLite backend",
                context={"actual_type": type(df).__name__},
                hint="Provide a pandas.DataFrame",
            )
        rng = np.random.default_rng()
        return df.assign(**{column_name: rng.permutation(len(df))})

    def prepare_covar_df(
        self, df: DataFrameLike, columns: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        if not isinstance(df, pd.DataFrame):
            raise UnsupportedBackendError("Expected pandas DataFrame in SQLite backend")
        x, y = columns
        masked_x = self._unused_column_name(df, "_dpsql_covar_x_")
        masked_y = self._unused_column_name(df, "_dpsql_covar_y_")
        valid = df[x].notna() & df[y].notna()
        result = df.assign(
            **{
                masked_x: df[x].where(valid),
                masked_y: df[y].where(valid),
            }
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
            "SQLite apply_aggregation: agg=%s columns=%s group_by=%s clip=%s",
            getattr(agg_type, "name", str(agg_type)),
            column_name,
            group_by,
            clipping_threshold,
        )
        # Add a dummy column to group by if group_by is empty
        if not isinstance(df, pd.DataFrame):
            raise UnsupportedBackendError(
                "Expected pandas DataFrame in SQLite backend",
                context={"actual_type": type(df).__name__},
                hint="Convert the input to a pandas.DataFrame before "
                "calling apply_aggregation",
            )
        if not group_by:
            dummy_column = self._unused_column_name(df, "_no_group_key_")
            df = df.assign(**{dummy_column: 0})
            group_by = [dummy_column]
        keys = [df[column] for column in group_by]
        grouped = df.groupby(keys, dropna=False, sort=False)

        if agg_type == Aggregation.COUNT:
            return (
                grouped.size()
                if column_name[0] == "*"
                else grouped[column_name[0]].count()
            )
        elif agg_type == Aggregation.COUNT_DISTINCT:
            return grouped[column_name[0]].nunique()
        elif agg_type in (Aggregation.SUM, Aggregation.SQUARED_SUM):
            lower_bound, upper_bound = safely_get_threshold(
                clipping_threshold, 0, agg_type.name
            )
            values = _clip_values(df[column_name[0]], lower_bound, upper_bound)
            if agg_type == Aggregation.SQUARED_SUM:
                values = values**2
        elif agg_type == Aggregation.PRODUCT_SUM:
            lower_bound_x, upper_bound_x = safely_get_threshold(
                clipping_threshold, 0, agg_type.name
            )
            lower_bound_y, upper_bound_y = safely_get_threshold(
                clipping_threshold, 1, agg_type.name
            )
            values = _clip_values(
                df[column_name[0]], lower_bound_x, upper_bound_x
            ) * _clip_values(df[column_name[1]], lower_bound_y, upper_bound_y)
        else:
            raise AggregationError(
                "Unsupported aggregation type",
                context={"agg_type": getattr(agg_type, "name", str(agg_type))},
                hint="Use one of COUNT, COUNT_DISTINCT, SUM, SQUARED_SUM, PRODUCT_SUM",
            )

        # Sum the numeric Series without coercing results back to the input dtype.
        return values.groupby(keys, dropna=False, sort=False).sum()

    def use_database(self, database_name: str | None) -> None:
        logger.info("SQLite use_database called: %s", database_name)
        """
        In SQLite, "USE database" is not directly supported as in MySQL or Spark.
        If you need to switch to a new database file, you can create a new
        connection or attach the database. By default, we do nothing here.
        """
        if database_name is not None:
            raise UnsupportedBackendError(
                "Unsupported database switch in SQLite backend",
                context={"requested_database": database_name},
                hint="Create a new connection for another database",
            )

    def get_table_name(self) -> list[str]:
        logger.debug("SQLite get_table_name")
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]

    def filter_by_selected_keys(
        self,
        df: DataFrameLike,
        group_by: list[str],
        selected_keys: list[tuple[str, ...]],
    ) -> pd.DataFrame:
        logger.debug(
            "SQLite filter_by_selected_keys: group_by=%s selected_keys_count=%s",
            group_by,
            len(selected_keys),
        )
        if not isinstance(df, pd.DataFrame):
            raise UnsupportedBackendError(
                "df must be a Pandas DataFrame",
                context={"actual_type": type(df).__name__},
                hint="Ensure the upstream backend returns a pandas.DataFrame",
            )

        if len(selected_keys) == 0:
            # If no selected keys, return an empty DataFrame with the same columns
            return df.iloc[0:0]
        if len(group_by) == 0:
            # If no group by columns, return the original DataFrame
            return df
        else:
            ref_df = pd.DataFrame(selected_keys, columns=group_by)
            filtered_df = df.merge(ref_df, on=group_by, how="inner")
            return filtered_df

    def get_column_name(self, table_name: str) -> list[str]:
        logger.debug("SQLite get_column_name: table=%s", table_name)
        cursor = self.conn.execute(
            f"PRAGMA table_info({_quote_pragma_literal(table_name)});"
        )
        # The info columns are typically: (cid, name, type, notnull, dflt_value, pk)
        return [row[1] for row in cursor.fetchall()]

    def is_inmemory_db(self) -> bool:
        logger.debug("SQLite is_inmemory_db check")
        cursor = self.conn.execute("PRAGMA database_list;")
        db_list = cursor.fetchall()
        # The first database is the main database
        if len(db_list) > 0 and db_list[0][2] == "":
            return True
        return False

    def create_temporary_table(
        self, df: pd.DataFrame, table_name: str, index: bool = True
    ) -> None:
        logger.info(
            "SQLite create_temporary_table: table=%s index=%s", table_name, index
        )
        """
        Create a temporary table in the database.
        Supports only in-memory databases.

        Args:
            df (DataFrame): The DataFrame to create the temporary table from.
            table_name (str): The name of the temporary table.
            index (bool): Whether to include the index as
            a column in the temporary table.
        """
        if not self.is_inmemory_db():
            raise UnsupportedBackendError(
                "Temporary tables are allowed only in in-memory databases",
                context={"is_inmemory": self.is_inmemory_db()},
                hint="Use an in-memory (:memory:) SQLite database",
            )

        if (
            table_name.find("temp.") == 0
            and len(table_name) > len("temp.")
            and re.match(r"^temp\.[A-Za-z_][A-Za-z0-9_]*$", table_name)
        ):
            df.to_sql(name=table_name, con=self.conn, if_exists="fail", index=index)
        else:
            raise ExecutionBackendError(
                "Temporary table names must start with 'temp.'.",
                context={"table_name": table_name},
                hint="Prefix the table name with 'temp.', e.g., temp.my_table",
            )
