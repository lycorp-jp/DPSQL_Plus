import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame

from ..aggregation import Aggregation, AggregationColumn
from ..dp_params import DPParams
from ..errors import ExecutionBackendError
from .aggregation_strategies import AggregationFactory
from .secure_sampling import secure_gauss

DataFrameLike: TypeAlias = pd.DataFrame | SparkDataFrame | pl.DataFrame

logger = logging.getLogger(__name__)


class SQLBackend(metaclass=ABCMeta):
    """
    Abstract class for executing SQL queries with differential privacy mechanisms.
    """

    def execute_sql(
        self,
        privacy_unit: str,
        params: DPParams,
        inner_sql: str,
        agg_columns: list[AggregationColumn],
        group_by_columns: list[str],
        ordering_terms: list[dict[str, str | None]],
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query with differential privacy mechanisms.

        Args:
            privacy_unit (str): The column name to use as the privacy unit.
            params (DPParams): The differential privacy parameters to use.
            inner_sql (str): The SQL query to create the intermediate table.
            agg_columns (list[AggregationColumn]):
                The list of aggregation columns to use.
            group_by_columns (list[str]): The list of columns to group by.
            ordering_terms (list[dict[str, str | None]]):
                The list of ordering terms for the final result.
            limit (int | None): The maximum number of rows to return.
            offset (int | None): The number of rows to skip before returning results.

        Returns:
            DataFrame: The result of the SQL query with DP mechanisms applied.
        """
        logger.info("Backend execute_sql start")
        logger.debug(
            "Args: privacy_unit=%s, group_by=%s, ordering_terms=%s,"
            " limit=%s, offset=%s, agg_count=%s",
            privacy_unit,
            group_by_columns,
            ordering_terms,
            limit,
            offset,
            len(agg_columns),
        )
        group_by = group_by_columns

        # Create intermediate table
        logger.debug("Creating inner_df (sql head): %s", inner_sql[:200])
        inner_df = self.create_inner_df(inner_sql)
        logger.debug(
            "inner_df created: type=%s, shape=%s",
            type(inner_df).__name__,
            getattr(inner_df, "shape", None),
        )

        # Reuse one random priority in both bounding stages. This preserves every
        # selected record from the first stage while allowing the second stage to
        # fill capacity released by keys that did not pass thresholding.
        priority_column = self._unused_column_name(
            inner_df, "_dpsql_contribution_priority_"
        )
        prioritized_df = self.add_random_priority(inner_df, priority_column)

        # First contribution bounding
        logger.debug(
            "Applying first contribution_bound: privacy_unit=%s, bound=%s",
            privacy_unit,
            params.contribution_bound,
        )
        logger.info("Applying first contribution_bound...")
        first_filtered_df = self.contribution_bound(
            prioritized_df, privacy_unit, params, priority_column
        )
        logger.debug(
            "First filtered df: shape=%s", getattr(first_filtered_df, "shape", None)
        )

        # Key selection (tau-thresholding) to determine which keys to keep
        logger.debug("Computing sensitivities for %s aggregations", len(agg_columns))
        sensitivities = self._compute_sensitivities(agg_columns, params)
        logger.debug("Sensitivities: %s", sensitivities)
        sigmas, tau, sigma_for_thresholding = params.get_noise_parameters(sensitivities)
        logger.debug(
            "Noise params: sigmas=%s, tau=%s, sigma_for_thresholding=%s",
            sigmas,
            tau,
            sigma_for_thresholding,
        )
        logger.debug(
            "Selecting keys: min_frequency=%s, tau=%s, sigma_th=%s",
            params.min_frequency,
            tau,
            sigma_for_thresholding,
        )
        logger.info("Selecting keys for thresholding...")
        selected_keys = self.key_selection(
            first_filtered_df,
            group_by,
            privacy_unit,
            params.min_frequency,
            sigma_for_thresholding,
            tau,
        )

        # Filter records to only include those with selected keys
        logger.debug("Filtering by selected keys...")
        key_filtered_df = self.filter_by_selected_keys(
            prioritized_df, group_by, selected_keys
        )
        logger.debug(
            "Key-filtered df: shape=%s", getattr(key_filtered_df, "shape", None)
        )

        # Second contribution bounding on the key-filtered data
        logger.info("Applying second contribution_bound...")
        final_filtered_df = self.contribution_bound(
            key_filtered_df, privacy_unit, params, priority_column
        )
        final_filtered_df = self._drop_column(final_filtered_df, priority_column)
        logger.debug(
            "Final filtered df: shape=%s", getattr(final_filtered_df, "shape", None)
        )

        # Apply noisy aggregation
        logger.debug("Creating final_df with %s aggregations", len(agg_columns))
        final_df = self.create_final_df(
            final_filtered_df, agg_columns, group_by, sigmas, params.clipping_thresholds
        )
        logger.debug(
            "final_df created: shape=%s, empty=%s",
            getattr(final_df, "shape", None),
            getattr(final_df, "empty", None),
        )
        logger.info("Aggregation with DP mechanisms applied")

        # Reset index if group by is empty
        if len(group_by) == 0:
            logger.debug("Resetting index (no group_by)")
            final_df = final_df.reset_index(drop=True)

        # Handle empty DataFrame case
        if final_df.empty:
            logger.info("final_df is empty; returning as-is")
            return final_df

        # Sort the final DataFrame based on ordering terms
        sort_keys, sorted_orders = self._sort_values_key(ordering_terms)
        logger.debug("Sort keys=%s, orders=%s", sort_keys, sorted_orders)
        if len(sort_keys) > 0:
            final_df = final_df.sort_values(by=sort_keys, ascending=sorted_orders)

        # Apply limit and offset if specified
        logger.debug("Applying limit/offset: limit=%s, offset=%s", limit, offset)
        if limit is not None:
            if offset is not None:
                final_df = final_df.iloc[offset : offset + limit]
            else:
                final_df = final_df.iloc[:limit]

        logger.info("Backend execute_sql done")
        return final_df

    def _sort_values_key(
        self, ordering_terms: list[dict[str, str | None]]
    ) -> tuple[list[str], list[bool]]:
        logger.debug("Building sort keys from ordering_terms=%s", ordering_terms)
        keys: list[str] = []
        orders: list[bool] = []
        for term in ordering_terms:
            if term["column_name"] is None:
                logger.error("Missing `column_name` in ordering term: %s", term)
                raise ExecutionBackendError(
                    "Missing `column_name` in ordering term",
                    context={"term": term},
                    hint="Specify column_name in each ORDER BY term",
                )
            keys.append(term["column_name"])
            orders.append(term["order"] is None or term["order"] == "ASC")
        logger.debug("Sort keys built: keys=%s, orders=%s", keys, orders)
        return keys, orders

    @abstractmethod
    def create_inner_df(self, inner_sql: str) -> DataFrameLike:
        """
        Create the intermediate DataFrame.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            DataFrameLike: The intermediate DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def contribution_bound(
        self,
        inner_df: DataFrameLike,
        privacy_unit: str,
        params: DPParams,
        priority_column: str,
    ) -> DataFrameLike:
        """
        Apply contribution bounding to the intermediate DataFrame.

        Args:
            inner_df (DataFrameLike): The intermediate DataFrame.
            privacy_unit (str): The column name to use as the privacy unit.
            params (DPParams): The differential privacy parameters to use.
            priority_column (str): A precomputed random-priority column.

        Returns:
            DataFrameLike: The filtered DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def add_random_priority(self, df: DataFrameLike, column_name: str) -> DataFrameLike:
        """Add a random row-priority column used for contribution bounding."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _unused_column_name(df: DataFrameLike, prefix: str) -> str:
        """Return an internal column name that does not collide with input columns."""
        columns = set(df.columns)
        column_name = prefix
        while column_name in columns:
            column_name += "_"
        return column_name

    @staticmethod
    def _drop_column(df: DataFrameLike, column_name: str) -> DataFrameLike:
        """Drop one internal column from a backend DataFrame."""
        if isinstance(df, pd.DataFrame):
            return df.drop(columns=[column_name])
        return df.drop(column_name)

    def create_final_df(
        self,
        filtered_df: DataFrameLike,
        agg_columns: list[AggregationColumn],
        group_by: list[str],
        sigmas: list[float],
        clipping_thresholds: Sequence[list[tuple[float, float]] | None],
    ) -> pd.DataFrame:
        """
        Create the final DataFrame with noisy aggregation applied.

        Args:
            filtered_df (DataFrameLike): The filtered DataFrame.
            agg_columns (list[AggregationColumn]):
                The list of aggregation columns to use.
            group_by (list[str]): The list of columns to group by.
            sigmas (list[float]): The list of noise standard deviations
                for each aggregation.
            clipping_thresholds (Sequence[list[tuple[float, float]] | None]):
                The list of clipping thresholds for each aggregation column.

        Returns:
            The final DataFrame with DP mechanisms applied.
        """
        # Construct the final DataFrame
        final_df = pd.DataFrame()

        for i, agg_column in enumerate(agg_columns):
            # Create strategy using Factory
            strategy = AggregationFactory.create_strategy(
                agg_column.aggregation_type, self
            )

            # Compute single aggregation
            result_series = strategy.compute(
                filtered_df,
                agg_column,
                group_by,
                sigmas[i],
                clipping_thresholds[i],
            )
            # Get column name for the aggregation result
            column_name = strategy.get_column_name(agg_column)

            # Add result to DataFrame
            final_df[column_name] = result_series

        return final_df

    def _compute_sensitivities(
        self, agg_columns: list[AggregationColumn], params: DPParams
    ) -> list[float]:
        """
        Calculate sensitivities for all aggregation columns.
        Args:
            agg_columns (list[AggregationColumn]):
                The list of aggregation columns to use.
            params (DPParams): The differential privacy parameters to use.
        Returns:
            list[float]: The list of sensitivities for each aggregation column.
        """
        return [
            agg_column.aggregation_type.get_sensitivity(
                params.contribution_bound, clipping_threshold
            )
            for agg_column, clipping_threshold in zip(
                agg_columns, params.clipping_thresholds, strict=False
            )
        ]

    @abstractmethod
    def apply_aggregation(
        self,
        agg_type: Aggregation,
        column_name: list[str],
        df: DataFrameLike,
        group_by: list[str],
        clipping_threshold: list[tuple[float, float]] | None = None,
    ) -> pd.Series:
        """
        Apply noisy aggregation to the DataFrame.

        Args:
            agg_type (Aggregation): The type of aggregation to apply.
            column_name (list[str]): The column name(s) to aggregate.
            df (DataFrame): The DataFrame to aggregate.
            group_by (list[str]): The list of columns to group by.
            clipping_threshold (list[tuple[float, float]] | None):
                The clipping thresholds for each column.

        Returns:
            DataFrame: The DataFrame with aggregation applied.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def key_selection(
        self,
        filtered_df: DataFrameLike,
        group_by: list[str],
        privacy_unit: str,
        min_frequency: int,
        sigma: float,
        tau: float,
    ) -> list[tuple[str, ...]]:
        """
        Perform key selection (tau-thresholding) to determine which keys to keep.

        Args:
            filtered_df (DataFrameLike): The filtered DataFrame.
            group_by (list[str]): The list of columns to group by.
            privacy_unit (str): The column name to use as the privacy unit.
            min_frequency (int): The threshold for first thresholding
              before adding noise. It satisfies minimum frequency rule.
            sigma (float): The standard deviation
                for the Gaussian mechanism before the second thresholding.
            tau (float): The threshold for second thresholding after adding noise.

        Returns:
            list[tuple[str, ...]]: List containing
            the selected keys that pass the threshold.
        """
        raw_count = self.apply_aggregation(
            Aggregation.COUNT_DISTINCT, [privacy_unit], filtered_df, group_by
        )
        noisy_count = raw_count.apply(lambda x: secure_gauss(x, sigma))
        selected_keys_series = noisy_count[
            (raw_count >= min_frequency) & (noisy_count >= tau)
        ]
        # Ensure each key is a tuple, even if index is a single column
        return [
            key if isinstance(key, tuple) else (key,)
            for key in selected_keys_series.index.tolist()
        ]

    @abstractmethod
    def filter_by_selected_keys(
        self,
        df: DataFrameLike,
        group_by: list[str],
        selected_keys: list[tuple[str, ...]],
    ) -> DataFrameLike:
        """
        Filter DataFrame to only include records with selected keys.

        Args:
            df (DataFrameLike): The DataFrame to filter.
            group_by (list[str]): The list of columns to group by.
            selected_keys (list[tuple[str, ...]]): List containing the selected keys.

        Returns:
            DataFrameLike: The filtered DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_table_name(self) -> list[str]:
        """
        Get the list of tables in the database.

        Returns:
            list[str]: The list of table names.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def use_database(self, database_name: str | None) -> None:
        """
        Use a database.

        Args:
            database_name (str): The name of the database to use.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_column_name(self, table_name: str) -> list[str]:
        """
        Get the list of columns in the table.

        Args:
            table_name (str): The name of the table.

        Returns:
            list[str]: The list of column names.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def create_temporary_table(
        self, df: pd.DataFrame, table_name: str, index: bool = True
    ) -> None:
        """
        Create a temporary table in the database.

        Args:
            df (DataFrame): The DataFrame to create the temporary table from.
            table_name (str): The name of the temporary table.
            index (bool): Whether to include the index as
                a column in the temporary table.
        """
        raise NotImplementedError("Subclasses must implement this method")
