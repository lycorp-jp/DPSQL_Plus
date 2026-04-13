import logging
from typing import Any

from lark import Lark, Transformer
from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedInput,
    UnexpectedToken,
    VisitError,
)
from lark.lexer import Token
from lark.tree import Meta, Tree
from lark.visitors import v_args  # type: ignore

from ..aggregation import AggregationColumn
from ..errors import (
    DPSQLError,
    InternalError,
    PrivacyConstraintError,
    QueryParseError,
    ValidationError,
)
from .alterprivate import AlterPrivateTableParser
from .ctes import CommonTableExpressionParser
from .finalselect import FinalSelectStmtParser
from .sqlgrammar import PrivSQLGrammar
from .utils import is_private_table_

logger = logging.getLogger(__name__)


class PrivSQLValidator(Transformer[Any, Any]):
    """Validator for the PrivSQL grammar."""

    def __init__(self, query: str):
        super().__init__()
        self._privacy_unit_columns: dict[str | None, dict[str, str]] = {}
        self._common_table_expression_parser: list[CommonTableExpressionParser] = []
        self._final_select_stmt_parser: FinalSelectStmtParser | None = None
        self._alter_private_table_str: list[str] = []
        self._common_table_expression_str: list[str] = []
        self._final_select_stmt_str: str | None = None
        self._final_select_core_str: str | None = None
        self._query = query
        logger.debug("PrivSQLValidator initialized (query_len=%s)", len(query))

    def _is_private_table(self, db_name: str | None, table_name: str) -> bool:
        """Check if the table is a private table.
        If the table is not a private table, return False."""
        res = is_private_table_(db_name, table_name, self._privacy_unit_columns)
        logger.debug(
            "Check private table: db=%s table=%s -> %s", db_name, table_name, res
        )
        return res

    def _add_private_column(
        self,
        db_name: str | None,
        table_name: str,
        column_name: str,
        column_name_alias: str | None = None,
    ) -> bool:
        """Add a private table to the collection.
        If the table name already exists, return False.
        If the table name does not exist, add it to the collection and return True."""
        if not self._is_private_table(db_name, table_name):
            if db_name not in self._privacy_unit_columns:
                self._privacy_unit_columns[db_name] = {}
            self._privacy_unit_columns[db_name][table_name] = (
                column_name if column_name_alias is None else column_name_alias
            )
            logger.debug(
                "Add private column: db=%s table=%s col=%s alias=%s",
                db_name,
                table_name,
                column_name,
                column_name_alias,
            )
            return True
        else:
            return False

    @v_args(meta=True)
    def alter_private_table(self, meta: Meta, children: list[Token | Tree[Any]]):
        """Parse the ALTER PRIVATE_TABLE statement and add the private table
        to the collection."""
        parser = AlterPrivateTableParser(children)
        if not self._add_private_column(
            parser.db_name, parser.table_name, parser.column_name
        ):
            raise ValidationError(
                "Duplicate private table registration",
                context={"table": parser.table_name, "db": parser.db_name},
                hint="Use a unique table name or avoid duplicate "
                "ALTER PRIVATE_TABLE statements",
            )
        self._alter_private_table_str.append(self._query[meta.start_pos : meta.end_pos])
        logger.info(
            "Parsed ALTER PRIVATE_TABLE: span=(%s,%s)", meta.start_pos, meta.end_pos
        )

    @v_args(meta=True)
    def common_table_expression(self, meta: Meta, children: list[Token | Tree[Any]]):
        """Parse the common table expression and check if the table is private.
        If the table is private, add the privacy unit column to the collection."""

        parser = CommonTableExpressionParser(children)
        if not parser.privacy_check(self._privacy_unit_columns):
            raise PrivacyConstraintError(
                "Privacy constraint failed in CTE",
                context={"cte": parser.table_name},
                hint="Ensure the CTE projects exactly one privacy unit column "
                "if private",
            )

        if parser.is_private_table(self._privacy_unit_columns):
            privacy_unit_column = parser.privacy_unit_column(self._privacy_unit_columns)
            if privacy_unit_column is None:
                raise PrivacyConstraintError(
                    "Missing privacy unit column in CTE",
                    context={"cte": parser.table_name},
                    hint="Select exactly one privacy unit column in the CTE projection",
                )
            if not self._add_private_column(
                None, parser.table_name, privacy_unit_column, None
            ):
                raise ValidationError(
                    "Duplicate private table registration in CTE",
                    context={"cte": parser.table_name},
                    hint="Use a distinct table name for each private CTE",
                )

        self._common_table_expression_str.append(
            self._query[meta.start_pos : meta.end_pos]
        )
        logger.info("Parsed CTE: span=(%s,%s)", meta.start_pos, meta.end_pos)
        logger.debug(
            "CTE privacy check: table=%s is_private=%s",
            parser.table_name,
            parser.is_private_table(self._privacy_unit_columns),
        )

    @v_args(meta=True)
    def final_select_stmt(self, meta: Meta, children: list[Token | Tree[Any]]):
        """Parse the final select statement and check if the table is private."""
        self._final_select_stmt_parser = FinalSelectStmtParser(children)
        parser = self._final_select_stmt_parser.final_select_core_parser
        if self._is_private_table(
            parser.db_name,
            parser.table_name,
        ):
            if not self._final_select_stmt_parser.privacy_check(
                self._privacy_unit_columns
            ):
                raise PrivacyConstraintError(
                    "Privacy constraint failed in FINAL SELECT",
                    context={"table": parser.table_name, "db": parser.db_name},
                    hint="Ensure exactly one privacy unit column and valid joins",
                )
        self._final_select_stmt_str = self._query[meta.start_pos : meta.end_pos]
        logger.info("Parsed FINAL SELECT: span=(%s,%s)", meta.start_pos, meta.end_pos)
        logger.debug(
            "FINAL SELECT target: db=%s table=%s", parser.db_name, parser.table_name
        )

    @property
    def privacy_unit_columns(self):
        return self._privacy_unit_columns

    @property
    def alter_private_tables_str(self) -> str:
        return "\n".join(self._alter_private_table_str)

    @property
    def common_table_expressions_str(self) -> str:
        if len(self._common_table_expression_str) == 0:
            return ""
        logger.debug(
            "common_table_expressions_str count=%s",
            len(self._common_table_expression_str),
        )
        return "WITH " + ", ".join(self._common_table_expression_str)

    @property
    def final_select_stmt_str(self) -> str | None:
        logger.debug(
            "final_select_stmt_str exists=%s", self._final_select_stmt_str is not None
        )
        return self._final_select_stmt_str

    @property
    def final_privacy_unit_column(self) -> str | None:
        if self._final_select_stmt_parser is None:
            raise InternalError(
                "Internal: FINAL SELECT parser not initialized",
                hint="Call analyze() before accessing final_privacy_unit_column",
            )
        if not self.is_private_query():
            return None
        parser = self._final_select_stmt_parser.final_select_core_parser
        logger.debug(
            "final_privacy_unit_column queried (is_private=%s)", self.is_private_query()
        )
        return self._privacy_unit_columns[parser.db_name][parser.table_name]

    def is_private_query(self) -> bool:
        if self._final_select_stmt_parser is None:
            raise InternalError(
                "Internal: FINAL SELECT parser not initialized",
                hint="Call analyze() before querying privacy status",
            )
        parser = self._final_select_stmt_parser.final_select_core_parser
        res = False
        if parser.db_name in self._privacy_unit_columns:
            if parser.table_name in self._privacy_unit_columns[parser.db_name]:
                res = True
        logger.debug("is_private_query -> %s", res)
        return res

    @property
    def final_select_stmt_parser(self) -> FinalSelectStmtParser:
        if self._final_select_stmt_parser is None:
            raise InternalError(
                "Internal: FINAL SELECT parser not initialized",
                hint="Call analyze() before accessing final_select_stmt_parser",
            )
        return self._final_select_stmt_parser


class PrivSQLParser:
    """Parser for PrivSQL queries.

    This class uses the Lark parser to parse PrivSQL queries and validate them.
    It provides methods to analyze the query, extract final select statements,
    and check if the query is private.
    """

    def __init__(self, private_mode: bool = True):
        self._parser = Lark(
            PrivSQLGrammar().grammar,
            parser="earley",
            start="query_statement",
            strict=True,
            propagate_positions=True,
        )
        self._validator: PrivSQLValidator | None = None
        self._private_mode = private_mode
        logger.debug("PrivSQLParser initialized (private_mode=%s)", private_mode)

    def analyze(self, query: str) -> bool:
        """
        Analyze the query and return the parse tree
        Args:
            query (str): The SQL query to analyze.
        Returns:
            bool: True if the query is private, False otherwise.
        """
        try:
            self._validator = PrivSQLValidator(query)
            self._validator.transform(self._parser.parse(query))
        except VisitError as e:
            orig = getattr(e, "orig_exc", None)
            if isinstance(orig, DPSQLError):
                raise orig from e
            raise QueryParseError(
                "Unexpected error during parse/validation",
                context={"wrapped_type": type(orig).__name__ if orig else None},
                hint="Check SQL syntax and privacy constraints.",
                cause=orig or e,
            ) from orig or e
        except (UnexpectedCharacters, UnexpectedToken, UnexpectedInput) as e:
            raise QueryParseError(
                "Syntax error in query",
                hint="Grammar violation. Verify tokens / structure.",
                cause=e,
            ) from e
        logger.info("Analyze query (len=%s)", len(query))
        logger.debug(
            "Analyze result (is_private=%s)",
            self._validator.is_private_query() if self._validator else None,
        )
        if self._private_mode and not self._validator.is_private_query():
            return False
        return True

    @property
    def final_select_stmt(self) -> str:
        """
        Return the final select statement string
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing final_select_stmt",
            )
        if self._validator.final_select_stmt_str is None:
            raise QueryParseError(
                "Missing FINAL SELECT statement",
                hint="Provide a FINAL SELECT statement in the query",
            )
        logger.debug("Access final_select_stmt")
        return self._validator.final_select_stmt_str

    @property
    def alter_private_table(self) -> str:
        """
        Return the alter private tables string
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing alter_private_table",
            )
        logger.debug("Access alter_private_table")
        return self._validator.alter_private_tables_str

    @property
    def common_table_expressions(self) -> str:
        """
        Return the common table expressions string
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing common_table_expressions",
            )
        logger.debug("Access common_table_expressions")
        return self._validator.common_table_expressions_str

    @property
    def privacy_unit_columns(self) -> dict[str | None, dict[str, str]]:
        """
        Return the privacy unit columns
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing privacy_unit_columns",
            )
        logger.debug("Access privacy_unit_columns")
        return self._validator.privacy_unit_columns

    @property
    def final_result_columns(self) -> list[AggregationColumn]:
        """
        Return the final result columns
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing final_result_columns",
            )
        logger.debug("Access final_result_columns")
        parser = self._validator.final_select_stmt_parser.final_select_core_parser
        return parser.final_result_columns

    @property
    def final_db_name(self) -> str | None:
        """
        Return the final database name
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing final_db_name",
            )
        logger.debug("Access final_db_name")
        parser = self._validator.final_select_stmt_parser.final_select_core_parser
        return parser.db_name

    @property
    def final_table_name(self) -> str:
        """
        Return the final table name
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing final_table_name",
            )
        logger.debug("Access final_table_name")
        parser = self._validator.final_select_stmt_parser.final_select_core_parser
        return parser.table_name

    @property
    def group_by_columns(self) -> list[str]:
        """
        Return the group by columns
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing group_by_columns",
            )
        logger.debug("Access group_by_columns")
        parser = self._validator.final_select_stmt_parser.final_select_core_parser
        return parser.group_by_columns

    @property
    def privacy_params(self) -> dict[str, float | int] | None:
        """
        Return the privacy parameters
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing privacy_params",
            )
        logger.debug("Access privacy_params")
        parser = self._validator.final_select_stmt_parser.final_select_core_parser
        return parser.privacy_params

    @property
    def final_privacy_unit_column(self) -> str | None:
        """
        Return the final privacy unit column
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing final_privacy_unit_column",
            )
        logger.debug("Access final_privacy_unit_column (property)")
        return self._validator.final_privacy_unit_column

    @property
    def limit(self) -> int | None:
        """
        Return the limit value if exists
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing limit",
            )
        logger.debug("Access limit")
        parser = self._validator.final_select_stmt_parser
        return parser.limit

    @property
    def offset(self) -> int | None:
        """
        Return the offset value if exists
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing offset",
            )
        logger.debug("Access offset")
        parser = self._validator.final_select_stmt_parser
        return parser.offset

    @property
    def ordering_terms(self) -> list[dict[str, str | None]]:
        """
        Return the ordering terms if exists
        """
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Call analyze() before accessing ordering_terms",
            )
        logger.debug("Access ordering_terms")
        parser = self._validator.final_select_stmt_parser
        return parser.ordering_terms

    def validate_and_get_intermediate_privacy_unit(
        self,
        query: str,
        db_schema: dict[str | None, dict[str, list[str]]],
        privacy_unit_columns: dict[str, str],
    ) -> str:
        """
        Validate the query and return the intermediate privacy unit column.

        Args:
            query (str): The SQL query to validate.
            db_schema (dict[str | None, dict[str, list[str]]]): The database schema.
            privacy_unit_columns (dict[str, str]): The privacy unit columns of
              the database.

        Returns:
            str: The intermediate privacy unit column.
        """
        alter_private_tables_queries = ""
        for table_name, column in privacy_unit_columns.items():
            alter_private_tables_queries += (
                f"ALTER PRIVATE_TABLE {table_name} OPTIONS "
                f"(PRIVACY_UNIT_COLUMN = {column});\n"
            )
        query = query.rstrip()
        query += ";" if query[-1] != ";" else ""
        query = f"{alter_private_tables_queries} {query}"
        logger.debug(
            "Validate and get intermediate privacy unit (db_schema_dbs=%s, pucount=%s)",
            len(db_schema),
            len(privacy_unit_columns),
        )
        logger.debug(
            "Injected ALTER PRIVATE_TABLE statements for %s tables",
            len(privacy_unit_columns),
        )
        if not self.analyze(query):
            raise PrivacyConstraintError(
                "Query does not reference a private table",
                hint="Register a private table and ensure the FINAL SELECT references "
                "exactly one privacy unit column",
            )
        if self._validator is None:
            raise InternalError(
                "Internal: Validator not initialized",
                hint="Unexpected state after analyze(); file an issue",
            )
        if self._validator.final_privacy_unit_column is None:
            raise PrivacyConstraintError(
                "Missing final privacy unit column",
                hint="Ensure the private table projects "
                "exactly one privacy unit column",
            )
        logger.debug(
            "Intermediate privacy unit resolved: %s",
            self._validator.final_privacy_unit_column if self._validator else None,
        )
        return self._validator.final_privacy_unit_column
