import logging

from ..aggregation import AggregationColumn
from .privparser import PrivSQLParser

logger = logging.getLogger(__name__)


class Validator:
    def _intermediate_queries(self, validator: PrivSQLParser) -> str:
        intermediate_queries = validator.common_table_expressions
        final_table_name = (
            validator.final_table_name
            if validator.final_db_name is None
            else validator.final_db_name + "." + validator.final_table_name
        )
        intermediate_queries += f" SELECT * FROM {final_table_name}"

        logger.debug(
            "Build intermediate queries for final table=%s.%s",
            validator.final_db_name,
            validator.final_table_name,
        )
        logger.debug("Intermediate queries built (len=%s)", len(intermediate_queries))
        return intermediate_queries

    def validate_and_get_final_select_items(
        self,
        query: str,
        db_schema: dict[str | None, dict[str, list[str]]],
        privacy_unit_columns: dict[str, str],
    ) -> tuple[
        str,
        str,
        list[AggregationColumn],
        list[str],
        list[dict[str, str | None]],
        int | None,
        int | None,
        dict[str, float | int] | None,
    ]:
        """
        validate the query on the basis of the db_schema and get the final select items.

        Args:
            query: The SQL query to validate.
            db_schema: The dictionary of the database schema. {db: {table: [columns]}}
            privacy_unit_columns:
                The dictionary of the privacy unit columns. {db.table: column}

        Returns:
            str: The privacy unit to use in the intermideate table for the query.
            str: The intermediate queries to execute before the final query.
            list[AggregationColumn]: The final result columns
            list[str]: The group by columns.
            list[dict[str, str | None]]: The ordering terms.
                Each dict contains key{"column_name", "order"}.
            int | None: The limit for the final query.
            int | None: The offset for the final query.
            dict[str, float | int] | None: The privacy parameters.
        """
        validator = PrivSQLParser()
        logger.info("Validator started for query validation")
        logger.debug(
            "Inputs: schema_dbs=%s privacy_units=%s",
            len(db_schema),
            len(privacy_unit_columns),
        )

        privacy_unit = validator.validate_and_get_intermediate_privacy_unit(
            query, db_schema, privacy_unit_columns
        )
        logger.debug("Intermediate privacy_unit=%s", privacy_unit)

        intermediate_queries = self._intermediate_queries(validator)

        logger.debug(
            "Final: group_by=%s ordering=%s limit=%s offset=%s",
            validator.group_by_columns,
            validator.ordering_terms,
            validator.limit,
            validator.offset,
        )
        logger.debug("Privacy params=%s", validator.privacy_params)
        return (
            privacy_unit,
            intermediate_queries,
            validator.final_result_columns,
            validator.group_by_columns,
            validator.ordering_terms,
            validator.limit,
            validator.offset,
            validator.privacy_params,
        )
