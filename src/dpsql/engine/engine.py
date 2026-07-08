import logging
import re

from pandas import DataFrame

from ..accountant import Accountant
from ..aggregation import Aggregation
from ..backend import SQLBackend
from ..dp_params import DPParams, generate_dpparams
from ..errors import (
    EngineError,
    InsufficientPrivacyBudgetError,
    InvalidPrivacyParametersError,
    UnsupportedBackendError,
)
from ..validator import Validator

logger = logging.getLogger(__name__)

_SAFE_TEMPORARY_TABLE_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


class Engine:
    def __init__(
        self, accountant: Accountant, sql_backend: SQLBackend, validator: Validator
    ):
        self.accountant = accountant
        self.sql_backend = sql_backend
        self.validator = validator
        self.privacy_unit_columns: dict[str, str] = {}
        self.db_schema: dict[str | None, dict[str, list[str]]] = {}

        logger.info("Initializing Engine")
        logger.debug(
            "Components: backend=%s accountant=%s auditor=%s",
            type(self.sql_backend).__name__,
            type(self.accountant).__name__ if self.accountant else None,
            type(self.validator).__name__ if self.validator else None,
        )

    def get_db_schema(self, database_name: str | None) -> dict[str, list[str]]:
        """
        Get the schema of a database.

        Args:
            database_name (str): The name of the database.

        Returns:
            dict: The schema of the database.
            ex) {"table1": ["column1", "column2"], "table2": ["column1"]}
        """

        self.sql_backend.use_database(database_name)
        tables = self.sql_backend.get_table_name()
        db_schema: dict[str, list[str]] = {}
        logger.info("Fetching DB schema")
        logger.debug("Target database: %s", database_name)
        for table_name in tables:
            columns = self.sql_backend.get_column_name(table_name)
            db_schema[table_name] = columns
            logger.debug("Schema: %s -> %s", table_name, columns)
        return db_schema

    def register_database(
        self,
        database_name: str | None = None,
        privacy_unit_columns: dict[str, str] | None = None,
    ) -> None:
        """
        Register a database with privacy unit columns in Engine.

        Args:
            database_name (str): The name of the database.
            privacy_unit_columns (dict): The privacy unit columns of the database.
                ex) {"table1": "column1"}
                if there is no privacy unit column in a table, give {}

        Returns:
            None
        """

        if privacy_unit_columns is None:
            privacy_unit_columns = {}

        db_schema = self.get_db_schema(database_name)

        logger.info("Registering database")
        logger.debug(
            "Database=%s privacy_unit_columns=%s", database_name, privacy_unit_columns
        )
        for table_name, privacy_unit_column in privacy_unit_columns.items():
            if table_name not in db_schema:
                logger.error(
                    "Missing table in database: table=%s database=%s",
                    table_name,
                    database_name,
                )
                raise EngineError(
                    "Missing table in database",
                    context={"database": database_name, "table": table_name},
                    hint="Verify table name via get_db_schema() or SHOW TABLES",
                )
            if privacy_unit_column not in db_schema[table_name]:
                logger.error(
                    "Missing privacy unit column: table=%s column=%s available=%s",
                    table_name,
                    privacy_unit_column,
                    db_schema[table_name],
                )
                raise EngineError(
                    "Missing privacy unit column in table",
                    context={
                        "database": database_name,
                        "table": table_name,
                        "column": privacy_unit_column,
                        "available_columns": db_schema[table_name],
                    },
                    hint="Confirm the column exists or "
                    "adjust privacy_unit_columns mapping",
                )

        self.db_schema[database_name] = db_schema

        logger.debug("Registered DB schema for %s", database_name)
        for table_name, privacy_unit_column in privacy_unit_columns.items():
            if database_name is None:
                self.privacy_unit_columns[f"{table_name}"] = privacy_unit_column
            else:
                self.privacy_unit_columns[f"{database_name}.{table_name}"] = (
                    privacy_unit_column
                )
            logger.debug(
                "Registered privacy unit: %s -> %s",
                f"{table_name}"
                if database_name is None
                else f"{database_name}.{table_name}",
                privacy_unit_column,
            )

    def execute_query(
        self,
        query: str,
        dpparams: DPParams | None,
        temporary_table_name: str | None = None,
    ) -> DataFrame:
        """
        Execute a SQL query with privacy parameters.

        Args:
            query (str): The SQL query to execute.
            dpparams (DPParams | None): The differential privacy parameters.
            temporary_table_name (str | None): The name of the temporary table
              to save the result.

        Returns:
            DataFrame: The result of the query execution.
        """
        (
            intermediate_privacy_unit,
            inner_sql,
            final_result_columns,
            group_by_columns,
            ordering_terms,
            limit,
            offset,
            privacy_params,
        ) = self.validator.validate_and_get_final_select_items(
            query, self.db_schema, self.privacy_unit_columns
        )

        logger.info("Executing private query")
        logger.debug("Query head: %s", query[:200])
        # Filter out non-aggregated columns
        agg_columns = [
            agg_column
            for agg_column in final_result_columns
            if agg_column.aggregation_type != Aggregation.NONE
        ]
        agg_funcs = [agg_column.aggregation_type for agg_column in agg_columns]
        if dpparams is None:
            if privacy_params is None:
                logger.error("Missing privacy parameters for query")
                raise InvalidPrivacyParametersError(
                    "Missing privacy parameters for query",
                    context={"query_head": query[:120]},
                    hint="Add PRIVATE_QUERY OPTIONS or pass dpparams explicitly",
                )
            dpparams = generate_dpparams(privacy_params, agg_columns)
            logger.debug("Generated DPParams from privacy_params")
        else:
            if privacy_params is not None:
                logger.error(
                    "Duplicate privacy parameter sources (explicit and in-query)"
                )
                raise InvalidPrivacyParametersError(
                    "Duplicate privacy parameter sources "
                    "(both explicit and in-query provided)",
                    context={"query_head": query[:120]},
                    hint="Use only one of dpparams argument or PRIVATE_QUERY OPTIONS",
                )

        logger.debug("Aggregations: %s", [a.name for a in agg_funcs])
        if not self.accountant.check_budget(agg_funcs, dpparams):
            raise InsufficientPrivacyBudgetError(
                "Privacy budget exceeded",
                context={
                    "requested_aggregations": [a.name for a in agg_funcs],
                    "epsilon_total": self.accountant.epsilon,
                    "delta_total": self.accountant.delta,
                },
                hint="Reduce per-query budget or increase global accountant budget",
            )

        logger.info("Executing inner SQL on backend")
        logger.debug("Inner SQL head: %s", inner_sql[:200])
        result = self.sql_backend.execute_sql(
            intermediate_privacy_unit,
            dpparams,
            inner_sql,
            agg_columns,
            group_by_columns,
            ordering_terms,
            limit,
            offset,
        )

        logger.info("Backend execution succeeded")
        if temporary_table_name is not None:
            self._validate_temporary_table_name(temporary_table_name)
            logger.info("Creating temporary table: %s", temporary_table_name)
            try:
                self.sql_backend.create_temporary_table(result, temporary_table_name)
                logger.debug("Temporary table created: %s", temporary_table_name)
            except UnsupportedBackendError as e:
                raise e

        self.accountant.update_budget(agg_funcs, dpparams)
        logger.info("Privacy budget updated")
        return result

    def _validate_temporary_table_name(self, temporary_table_name: str) -> None:
        if _SAFE_TEMPORARY_TABLE_NAME.fullmatch(temporary_table_name):
            return
        raise EngineError(
            "Invalid temporary table name",
            context={"temporary_table_name": temporary_table_name},
            hint=(
                "Use dot-separated identifiers containing only letters, "
                "numbers, and underscores"
            ),
        )
