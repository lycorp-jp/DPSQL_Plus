from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from ..aggregation import Aggregation, AggregationColumn
from ..errors import AggregationError
from ..utils import safely_get_threshold
from .bezier_mechanisms import private_avg, private_covar, private_stddev, private_var
from .secure_sampling import secure_gauss

# Avoid circular imports
if TYPE_CHECKING:
    from .sql_backend import (
        DataFrameLike,
        SQLBackend,
    )

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    """
    Base class for aggregation computations.
    """

    def __init__(self, backend: SQLBackend) -> None:
        self.backend = backend
        logger.debug("AggregationStrategy init: backend=%s", type(backend).__name__)

    @abstractmethod
    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        """
        Execute aggregation computation.

        Args:
            filtered_df (DataFrameLike): The filtered DataFrame.
            agg_column (AggregationColumn): The aggregation column information.
            group_by (list[str]): The list of columns to group by.
            sigma (float): The noise scale.
            clipping_threshold (list[tuple[float, float]] | None):
                The clipping thresholds for each column.

        Returns:
            pd.Series: The Series containing the aggregation results.
        """
        pass

    def get_column_name(self, agg_column: AggregationColumn) -> str:
        """
        Generate the column name for the aggregation result (default implementation).

        Args:
            agg_column (AggregationColumn): The aggregation column information.

        Returns:
            str: The generated column name.
        """
        if agg_column.alias is not None:
            return agg_column.alias

        # Default: Format as "agg_name(column_name)"
        agg_name = agg_column.aggregation_type.name.lower()
        column_name = ",".join(agg_column.columns)
        name = f"{agg_name}({column_name})"
        logger.debug("Generated aggregation column name: %s", name)
        return name


class CountAggregation(AggregationStrategy):
    """
    Implementation of COUNT aggregation.
    """

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "COUNT compute: cols=%s group_by=%s sigma=%s",
            agg_column.columns,
            group_by,
            sigma,
        )
        raw_result = self.backend.apply_aggregation(
            Aggregation.COUNT, agg_column.columns, filtered_df, group_by
        )
        return raw_result.apply(lambda x: secure_gauss(x, sigma))


class CountDistinctAggregation(AggregationStrategy):
    """
    Implementation of COUNT_DISTINCT aggregation.
    """

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "COUNT_DISTINCT compute: cols=%s group_by=%s sigma=%s",
            agg_column.columns,
            group_by,
            sigma,
        )
        raw_result = self.backend.apply_aggregation(
            Aggregation.COUNT_DISTINCT, agg_column.columns, filtered_df, group_by
        )
        return raw_result.apply(lambda x: secure_gauss(x, sigma))


class SumAggregation(AggregationStrategy):
    """
    Implementation of SUM aggregation.
    """

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "SUM compute: cols=%s group_by=%s sigma=%s clip=%s",
            agg_column.columns,
            group_by,
            sigma,
            clipping_threshold,
        )
        raw_result = self.backend.apply_aggregation(
            Aggregation.SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        return raw_result.apply(lambda x: secure_gauss(x, sigma))


class AvgAggregation(AggregationStrategy):
    """
    Implementation of AVG aggregation.
    """

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "AVG compute: cols=%s group_by=%s sigma=%s clip=%s",
            agg_column.columns,
            group_by,
            sigma,
            clipping_threshold,
        )
        raw_sum = self.backend.apply_aggregation(
            Aggregation.SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        raw_count = self.backend.apply_aggregation(
            Aggregation.COUNT, agg_column.columns, filtered_df, group_by
        )

        threshold = safely_get_threshold(clipping_threshold, 0, Aggregation.AVG.name)
        logger.debug("AVG threshold=%s", threshold)
        return pd.DataFrame({"raw_sum": raw_sum, "raw_count": raw_count}).apply(
            lambda x: private_avg(
                x["raw_count"],
                x["raw_sum"],
                sigma,
                threshold,
            ),
            axis=1,
        )


class StddevAggregation(AggregationStrategy):
    """
    Implementation of STDDEV aggregation (both SAMP and POP).
    """

    def __init__(self, backend: SQLBackend, pop: bool = False) -> None:
        super().__init__(backend)
        self.pop = pop
        logger.debug("STDDEV init: pop=%s", pop)

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "STDDEV compute: cols=%s group_by=%s sigma=%s clip=%s pop=%s",
            agg_column.columns,
            group_by,
            sigma,
            clipping_threshold,
            self.pop,
        )
        raw_sum = self.backend.apply_aggregation(
            Aggregation.SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        raw_squared_sum = self.backend.apply_aggregation(
            Aggregation.SQUARED_SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        raw_count = self.backend.apply_aggregation(
            Aggregation.COUNT, agg_column.columns, filtered_df, group_by
        )

        agg_type = Aggregation.STDDEV_POP if self.pop else Aggregation.STDDEV_SAMP
        threshold = safely_get_threshold(clipping_threshold, 0, agg_type.name)
        logger.debug("STDDEV threshold=%s", threshold)
        return pd.DataFrame(
            {
                "raw_sum": raw_sum,
                "raw_squared_sum": raw_squared_sum,
                "raw_count": raw_count,
            }
        ).apply(
            lambda x: private_stddev(
                x["raw_count"],
                x["raw_sum"],
                x["raw_squared_sum"],
                sigma,
                threshold,
                pop=self.pop,
            ),
            axis=1,
        )


class VarAggregation(AggregationStrategy):
    """
    Implementation of VAR aggregation (both SAMP and POP).
    """

    def __init__(self, backend: SQLBackend, pop: bool = False) -> None:
        super().__init__(backend)
        self.pop = pop
        logger.debug("VAR init: pop=%s", pop)

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "VAR compute: cols=%s group_by=%s sigma=%s clip=%s pop=%s",
            agg_column.columns,
            group_by,
            sigma,
            clipping_threshold,
            self.pop,
        )
        raw_sum = self.backend.apply_aggregation(
            Aggregation.SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        raw_squared_sum = self.backend.apply_aggregation(
            Aggregation.SQUARED_SUM,
            agg_column.columns,
            filtered_df,
            group_by,
            clipping_threshold,
        )
        raw_count = self.backend.apply_aggregation(
            Aggregation.COUNT, agg_column.columns, filtered_df, group_by
        )

        agg_type = Aggregation.VAR_POP if self.pop else Aggregation.VAR_SAMP
        threshold = safely_get_threshold(clipping_threshold, 0, agg_type.name)
        logger.debug("VAR threshold=%s", threshold)
        return pd.DataFrame(
            {
                "raw_sum": raw_sum,
                "raw_squared_sum": raw_squared_sum,
                "raw_count": raw_count,
            }
        ).apply(
            lambda x: private_var(
                x["raw_count"],
                x["raw_sum"],
                x["raw_squared_sum"],
                sigma,
                threshold,
                pop=self.pop,
            ),
            axis=1,
        )


class CovarAggregation(AggregationStrategy):
    """
    Implementation of COVAR aggregation (both SAMP and POP).
    """

    def __init__(self, backend: SQLBackend, pop: bool = False) -> None:
        super().__init__(backend)
        self.pop = pop
        logger.debug("COVAR init: pop=%s", pop)

    def compute(
        self,
        filtered_df: DataFrameLike,
        agg_column: AggregationColumn,
        group_by: list[str],
        sigma: float,
        clipping_threshold: list[tuple[float, float]] | None,
    ) -> pd.Series:
        logger.debug(
            "COVAR compute: cols=%s group_by=%s sigma=%s clip=%s",
            agg_column.columns,
            group_by,
            sigma,
            clipping_threshold,
        )
        agg_type = Aggregation.COVAR_POP if self.pop else Aggregation.COVAR_SAMP
        agg_column_x = agg_column.columns[0]
        agg_column_y = agg_column.columns[1]
        threshold_x = safely_get_threshold(clipping_threshold, 0, agg_type.name)
        threshold_y = safely_get_threshold(clipping_threshold, 1, agg_type.name)

        raw_sum_x = self.backend.apply_aggregation(
            Aggregation.SUM,
            [agg_column_x],
            filtered_df,
            group_by,
            [threshold_x],
        )
        raw_sum_y = self.backend.apply_aggregation(
            Aggregation.SUM,
            [agg_column_y],
            filtered_df,
            group_by,
            [threshold_y],
        )
        raw_sum_xy = self.backend.apply_aggregation(
            Aggregation.PRODUCT_SUM,
            [agg_column_x, agg_column_y],
            filtered_df,
            group_by,
            [threshold_x, threshold_y],
        )
        raw_count = self.backend.apply_aggregation(
            Aggregation.COUNT, [agg_column_x, agg_column_y], filtered_df, group_by
        )

        logger.debug("COVAR thresholds: x=%s y=%s", threshold_x, threshold_y)
        return pd.DataFrame(
            {
                "raw_sum_x": raw_sum_x,
                "raw_sum_y": raw_sum_y,
                "raw_sum_xy": raw_sum_xy,
                "raw_count": raw_count,
            }
        ).apply(
            lambda x: private_covar(
                x["raw_count"],
                x["raw_sum_x"],
                x["raw_sum_y"],
                x["raw_sum_xy"],
                sigma,
                threshold_x,
                threshold_y,
                pop=self.pop,
            ),
            axis=1,
        )


class AggregationFactory:
    """
    Factory class to create aggregation strategy instances.
    """

    @staticmethod
    def create_strategy(
        agg_type: Aggregation, backend: SQLBackend
    ) -> AggregationStrategy:
        logger.debug("AggregationFactory.create_strategy: agg_type=%s", agg_type.name)
        """
        Create the appropriate strategy based on the aggregation type.

        Args:
            agg_type (Aggregation): The aggregation type
            backend (SQLBackend): The SQLBackend instance

        Returns:
            AggregationStrategy: The created aggregation strategy instance

        Raises:
            AggregationError: If the aggregation type is not supported
        """
        strategy_map = {
            Aggregation.COUNT: CountAggregation,
            Aggregation.COUNT_DISTINCT: CountDistinctAggregation,
            Aggregation.SUM: SumAggregation,
            Aggregation.AVG: AvgAggregation,
            Aggregation.STDDEV_SAMP: lambda b: StddevAggregation(b, pop=False),
            Aggregation.STDDEV_POP: lambda b: StddevAggregation(b, pop=True),
            Aggregation.VAR_SAMP: lambda b: VarAggregation(b, pop=False),
            Aggregation.VAR_POP: lambda b: VarAggregation(b, pop=True),
            Aggregation.COVAR_SAMP: lambda b: CovarAggregation(b, pop=False),
            Aggregation.COVAR_POP: lambda b: CovarAggregation(b, pop=True),
        }

        if agg_type not in strategy_map:
            logger.error("Unsupported aggregation type: %s", agg_type.name)
            raise AggregationError(
                f"Unsupported aggregation type: {agg_type.name}",
                context={
                    "agg_type": agg_type.name,
                    "supported_types": list(strategy_map.keys()),
                },
                hint="Use one of the supported aggregation types",
            )

        strategy_creator = strategy_map[agg_type]
        return strategy_creator(backend)
