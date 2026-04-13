import logging
from collections.abc import Iterator
from typing import Any

from lark import Transformer
from lark.lexer import Token
from lark.tree import Tree

from ..errors import (
    PrivacyConstraintError,
    QueryParseError,
    UnsupportedQueryError,
)
from .ctes_join import PrivateJoinConstraintParser
from .sqlgrammar import PrivacyChecker
from .utils import (
    is_private_table_,
    resolve_children_value,
    resolve_instance_3_values,
    resolve_instance_value,
    token_concat,
)

logger = logging.getLogger(__name__)


class ResultColumnParser(Transformer[Any, Any]):
    """This parser is used to parse result column in SELECT statement.
    It is used to check if the result column is a pure result column,
     which means it contains only one expression.
    """

    def __init__(self) -> None:
        super().__init__()
        self._expr_count = 0
        self._is_pure_result_column = True
        self._column_names: list[str] = []
        self._table_names: list[str] = []
        logger.debug("Init ResultColumnParser")

    def expr(self, items: list[Token | Tree[Any]]):
        self._expr_count += 1
        logger.debug("ResultColumnParser.expr count=%s", self._expr_count)
        if self._expr_count > 1024:
            raise QueryParseError(
                "Too many expressions in result column",
                context={"expr_count": self._expr_count},
                hint="Limit pure result columns to a single expression",
            )

    def column_name(self, items: list[Token | Tree[Any]]):
        self._column_names.append(resolve_children_value(items))
        logger.debug("ResultColumnParser.column_name -> %s", self._column_names[-1])

    def table_name(self, items: list[Token | Tree[Any]]):
        self._table_names.append(resolve_children_value(items))
        logger.debug("ResultColumnParser.table_name -> %s", self._table_names[-1])

    @property
    def is_pure_result_column(self) -> bool:
        res = self._expr_count == 1
        logger.debug("ResultColumnParser.is_pure_result_column -> %s", res)
        return res

    @property
    def column_names(self) -> list[str]:
        logger.debug("ResultColumnParser.column_names -> %s", self._column_names)
        return self._column_names

    @property
    def table_names(self) -> list[str]:
        logger.debug("ResultColumnParser.table_names -> %s", self._table_names)
        return self._table_names


def _is_privacy_unit_column(
    column_name: str,
    private_unit_columns: dict[str | None, dict[str, str]],
    db_name: str | None,
    table_name: str,
) -> bool:
    logger.debug(
        "_is_privacy_unit_column: db=%s table=%s column=%s",
        db_name,
        table_name,
        column_name,
    )
    if not is_private_table_(db_name, table_name, private_unit_columns):
        return False
    res = column_name == private_unit_columns[db_name][table_name]
    logger.debug("_is_privacy_unit_column -> %s", res)
    return res


def _is_include_privacy_unit_column(
    column_names: list[str],
    privacy_unit_columns: dict[str | None, dict[str, str]],
    db_name: str | None,
    table_name: str,
) -> bool:
    logger.debug(
        "_is_include_privacy_unit_column: db=%s table=%s columns=%s",
        db_name,
        table_name,
        column_names,
    )
    for column_name in column_names:
        if _is_privacy_unit_column(
            column_name, privacy_unit_columns, db_name, table_name
        ):
            logger.debug("Privacy unit column included: %s", column_name)
            return True
    return False


def get_privacy_unit_column_with_privacy_check(
    item: Tree[Any],
    privacy_unit_columns: dict[str | None, dict[str, str]],
    from_db_name: str | None,
    join_db_name: str | None,
    from_table_name: str,
    join_table_name: str | None,
) -> str | None:
    logger.debug("Resolve privacy unit column with privacy check")
    logger.debug(
        "Inputs: from=%s.%s join=%s.%s",
        from_db_name,
        from_table_name,
        join_db_name,
        join_table_name,
    )
    privacy_unit_column_name = None
    isExhausted = False
    item_iterator = iter(item.children)
    """    This function resolves the privacy unit column name from the item."""

    try:
        node = next(item_iterator)
        while True:
            while isinstance(node, Token):
                if node.type != "AS":
                    raise QueryParseError(
                        "Unexpected token in result column",
                        context={"token": node.type, "value": str(node)},
                        hint="Expect alias tokens (AS) or expression trees",
                    )
                node = next(item_iterator)
            match node.data:
                case "expr":
                    parser = ResultColumnParser()
                    parser.transform(node)
                    if join_table_name is not None and len(parser.table_names) != len(
                        parser.column_names
                    ):
                        raise QueryParseError(
                            "Mismatch between table and column references "
                            "in join expression",
                            context={
                                "tables": parser.table_names,
                                "columns": parser.column_names,
                            },
                            hint="Ensure each column is qualified by its table "
                            "in join queries",
                        )

                    privacy_unit_column_name = _resolve_privacy_column_name(
                        privacy_unit_columns,
                        from_db_name,
                        join_db_name,
                        from_table_name,
                        join_table_name,
                        node,
                        parser,
                    )
                case "column_name_alias":
                    if privacy_unit_column_name is not None:
                        privacy_unit_column_name = resolve_instance_value(
                            node, "column_name_alias"
                        )
                case _:
                    raise QueryParseError(
                        "`*` wildcard or unsupported structure "
                        "in private table column list",
                        context={"node_type": node.data},
                        hint="Avoid using * expansions in private tables",
                    )
            node = next(item_iterator)
    except StopIteration:
        isExhausted = True

    if isExhausted:
        logger.debug("Resolved privacy unit column -> %s", privacy_unit_column_name)
        return privacy_unit_column_name
    else:
        raise QueryParseError(
            "Unexpected trailing input in privacy unit column",
            hint="Ensure result column specification terminates correctly",
        )


def _resolve_privacy_column_name(
    privacy_unit_columns: dict[str | None, dict[str, str]],
    from_db_name: str | None,
    join_db_name: str | None,
    from_table_name: str,
    join_table_name: str | None,
    node: Tree[Any],
    parser: ResultColumnParser,
) -> str | None:
    logger.debug(
        "_resolve_privacy_column_name: from=%s.%s join=%s.%s pure=%s",
        from_db_name,
        from_table_name,
        join_db_name,
        join_table_name,
        parser.is_pure_result_column,
    )
    if join_table_name is None:
        privacy_unit_column_name = _get_privacy_unit_column_name(
            privacy_unit_columns, from_db_name, from_table_name, node, parser
        )
    else:
        privacy_unit_column_name = _get_privacy_unit_column_name(
            privacy_unit_columns, from_db_name, from_table_name, node, parser
        )
        if privacy_unit_column_name is None:
            privacy_unit_column_name = _get_privacy_unit_column_name(
                privacy_unit_columns, join_db_name, join_table_name, node, parser
            )
    logger.debug("_resolve_privacy_column_name -> %s", privacy_unit_column_name)
    return privacy_unit_column_name


def _get_privacy_unit_column_name(
    privacy_unit_columns: dict[str | None, dict[str, str]],
    db_name: str | None,
    table_name: str,
    node: Tree[Any],
    parser: ResultColumnParser,
) -> str | None:
    logger.debug(
        "_get_privacy_unit_column_name: db=%s table=%s pure=%s",
        db_name,
        table_name,
        parser.is_pure_result_column,
    )
    privacy_unit_column_name: str | None = None
    if parser.is_pure_result_column:
        if len(parser.column_names) != 1:
            raise QueryParseError(
                "Expected exactly one column in pure result column",
                context={"columns": parser.column_names},
                hint="Simplify the expression or remove extra columns",
            )
        column_name = parser.column_names[0]
        privacy_unit_column_name = (
            column_name
            if _is_privacy_unit_column(
                column_name,
                privacy_unit_columns,
                db_name,
                table_name,
            )
            else None
        )
    else:
        if _is_include_privacy_unit_column(
            parser.column_names,
            privacy_unit_columns,
            db_name,
            table_name,
        ):
            raise PrivacyConstraintError(
                "Privacy unit column in composite expression is disallowed",
                context={
                    "columns": parser.column_names,
                    "table": table_name,
                },
                hint="Project the privacy unit column alone",
            )
    if privacy_unit_column_name is None:
        logger.debug(
            "privacy_unit_column_name not found for %s.%s", db_name, table_name
        )
    return privacy_unit_column_name


class SelectCoreParser(PrivacyChecker):
    """This parser is used to parse select core in SELECT statement.
    It is used to check if the select core is a private table.
    """

    def __init__(self, items: list[Token | Tree[Any]]):
        logger.debug("Init SelectCoreParser (items=%s)", len(items))
        self._from_table_name: str | None = None
        self._from_table_name_alias: str | None = None
        self._from_db_name: str | None = None
        self._join_table_name: str | None = None
        self._join_table_name_alias: str | None = None
        self._join_db_name: str | None = None
        self._join_operator: str | None = None
        self._join_constraint: list[Tree[Any] | Token] | None = None
        self._equijoin_conditions: PrivateJoinConstraintParser | None = None
        self._raw_result_columns: list[Tree[Any]] = []
        self._privacy_unit_column: str | None = None
        self._used_spark_sql_extensions = False
        isExhausted = False

        try:
            items_iterator = iter(items)
            item = next(items_iterator)
            while isinstance(item, Token):
                match item.type:
                    case "SELECT":
                        item, items_iterator = self._select(items_iterator)
                    case "FROM":
                        item, items_iterator = self._from(items_iterator)
                    case "PIVOT" | "UNPIVOT" | "LATERAL":
                        item, items_iterator = self._spark_sql_extensions(
                            item, items_iterator
                        )
                    case "WHERE":
                        item, items_iterator = self._where(items_iterator)
                    case "GROUP":
                        item, items_iterator = self._group_by(items_iterator)
                    case "HAVING":
                        item, items_iterator = self._having(items_iterator)
                    case _:
                        raise QueryParseError(
                            "Unexpected clause token in select core",
                            context={"token_type": item.type, "value": str(item)},
                            hint="Allowed: SELECT, FROM, PIVOT, UNPIVOT, LATERAL, "
                            "WHERE, GROUP, HAVING",
                        )
        except StopIteration:
            isExhausted = True

        if not isExhausted:
            raise QueryParseError(
                "Unexpected trailing input in select core",
                hint="Remove extraneous tokens after the last clause",
            )
        else:
            logger.debug(
                "SelectCore parsed: from=%s join=%s op=%s spark_ext=%s",
                self._from_table_name,
                self._join_table_name,
                self._join_operator,
                self._used_spark_sql_extensions,
            )

    def _select(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse SELECT list")
        item = next(items_iterator)
        if isinstance(item, Tree) and item.data == "hints_clause":
            item = next(items_iterator)
        if isinstance(item, Token):
            if item.type != "DISTINCT" and item.type != "ALL":
                raise QueryParseError(
                    "Unexpected set quantifier",
                    context={"token_type": item.type},
                    hint="Use DISTINCT or ALL",
                )
            item = next(items_iterator)
        while isinstance(item, Tree):
            match item.data:
                case "result_column":
                    self._raw_result_columns.append(item)
                case _:
                    raise QueryParseError(
                        "Unexpected element in select list",
                        context={"node": item.data},
                        hint="Provide only result_column entries",
                    )
            item = next(items_iterator)
        return item, items_iterator

    def _from(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse FROM clause")
        item = next(items_iterator)
        if not isinstance(item, Tree):
            raise QueryParseError(
                "Unexpected token in `FROM` clause",
                context={"value": str(item)},
                hint="Expect table_expr or join_clause tree",
            )
        match item.data:
            case "table_expr":
                (
                    self._from_db_name,
                    self._from_table_name,
                    self._from_table_name_alias,
                ) = resolve_instance_3_values(item, "table_expr")
            case "join_clause":
                if len(item.children) != 4:
                    raise QueryParseError(
                        "Unexpected child count in join clause",
                        context={"child_count": len(item.children)},
                        hint="join_clause must have 4 children",
                    )
                (
                    from_table_expr,
                    join_operator,
                    join_table_expr,
                    join_constraint,
                ) = item.children
                (
                    self._from_db_name,
                    self._from_table_name,
                    self._from_table_name_alias,
                ) = resolve_instance_3_values(from_table_expr, "table_expr")
                (
                    self._join_db_name,
                    self._join_table_name,
                    self._join_table_name_alias,
                ) = resolve_instance_3_values(join_table_expr, "table_expr")
                self._join_operator = token_concat(join_operator.children, is_type=True)
                self._join_constraint = join_constraint.children
            case _:
                raise QueryParseError(
                    "Unexpected element in `FROM` clause",
                    context={"node": item.data},
                    hint="Use table_expr or join_clause",
                )
        item = next(items_iterator)
        logger.debug(
            "FROM parsed: from=%s join=%s op=%s",
            self._from_table_name,
            self._join_table_name,
            self._join_operator,
        )
        return item, items_iterator

    def _where(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse WHERE clause")
        item = next(items_iterator)
        if not isinstance(item, Tree) or item.data != "expr":
            raise QueryParseError(
                "Unexpected element in `WHERE` clause",
                context={"value": str(item)},
                hint="Expect an expr tree",
            )
        item = next(items_iterator)

        return item, items_iterator

    def _group_by(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse GROUP BY clause")
        item = next(items_iterator)
        if not isinstance(item, Token) or item.type != "BY":
            raise QueryParseError(
                "Missing BY token in GROUP BY clause",
                context={"value": str(item)},
                hint="GROUP must be followed by BY",
            )
        item = next(items_iterator)
        while isinstance(item, Tree):
            match item.data:
                case "expr":
                    pass
                case _:
                    raise QueryParseError(
                        "Unexpected element in `GROUP BY` list",
                        context={"node": item.data},
                        hint="Only expr nodes allowed",
                    )
            item = next(items_iterator)
        return item, items_iterator

    def _having(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse HAVING clause")
        item = next(items_iterator)
        if not isinstance(item, Tree) or item.data != "expr":
            raise QueryParseError(
                "Unexpected element in `HAVING` clause",
                context={"value": str(item)},
                hint="Expect an expr tree",
            )
        item = next(items_iterator)

        return item, items_iterator

    def _spark_sql_extensions(
        self, clause_type: Token, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        self._used_spark_sql_extensions = True
        match clause_type.type:
            case "PIVOT":
                item = next(items_iterator)
                if not isinstance(item, Tree) or item.data != "pivot_clause":
                    raise QueryParseError(
                        "Unexpected element after `PIVOT`",
                        context={"value": str(item)},
                        hint="Expect pivot_clause",
                    )
                item = next(items_iterator)
            case "UNPIVOT":
                item = next(items_iterator)
                if not isinstance(item, Tree) or item.data != "unpivot_clause":
                    raise QueryParseError(
                        "Unexpected element after `UNPIVOT`",
                        context={"value": str(item)},
                        hint="Expect unpivot_clause",
                    )
                item = next(items_iterator)
            case "LATERAL":
                item = next(items_iterator)
                if not isinstance(item, Token) or item.type != "VIEW":
                    raise QueryParseError(
                        "Unexpected element after LATERAL",
                        context={"value": str(item)},
                        hint="Expect VIEW token",
                    )
                item = next(items_iterator)
                if not isinstance(item, Tree) or item.data != "lateral_view_clause":
                    raise QueryParseError(
                        "Unexpected element after `LATERAL VIEW`",
                        context={"value": str(item)},
                        hint="Expect lateral_view_clause",
                    )
                item = next(items_iterator)
            case _:
                raise QueryParseError(
                    "Unsupported Spark SQL extension token",
                    context={"token": clause_type.type},
                    hint="Allowed: PIVOT, UNPIVOT, LATERAL",
                )
        return item, items_iterator

    def _join_private_table_privacy_check(
        self,
        from_table_name: str,
        privacy_unit_columns: dict[str | None, dict[str, str]],
    ):
        """Check if the join clause is a private table."""
        logger.debug(
            "Join privacy check: from=%s join=%s op=%s",
            from_table_name,
            self._join_table_name,
            self._join_operator,
        )
        if is_private_table_(
            self._from_db_name, from_table_name, privacy_unit_columns
        ) and is_private_table_(
            self._join_db_name, self._join_table_name, privacy_unit_columns
        ):
            if self._join_operator == "CROSS_JOIN":
                raise PrivacyConstraintError(
                    "CROSS JOIN between private tables is disallowed",
                    context={
                        "from_table": from_table_name,
                        "join_table": self._join_table_name,
                    },
                    hint="Use equi-join on privacy unit columns",
                )
            if self._join_constraint is None:
                raise PrivacyConstraintError(
                    "Missing join constraint between private tables",
                    context={
                        "from_table": from_table_name,
                        "join_table": self._join_table_name,
                    },
                    hint="Provide an equi-join condition on the privacy unit columns",
                )
            if self._join_table_name is not None and not PrivateJoinConstraintParser(
                self._join_constraint,
                self._from_db_name,
                self._join_db_name,
                from_table_name,
                self._join_table_name,
                self._from_table_name_alias,
                self._join_table_name_alias,
            ).privacy_check(privacy_unit_columns):
                raise PrivacyConstraintError(
                    "Privacy unit join constraint validation failed",
                    context={
                        "from_table": self._from_table_name,
                        "join_table": self._join_table_name,
                    },
                    hint="Ensure exactly one equi-join matches "
                    "both privacy unit columns",
                )

    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("SelectCore privacy_check")
        is_privacy_check = False
        if self._from_table_name is None:
            raise QueryParseError(
                "Missing FROM table name in select core",
                hint="Ensure a table or subquery appears after FROM",
            )
        if not self.is_private_table(privacy_unit_columns):
            return True

        self._join_private_table_privacy_check(
            self._from_table_name, privacy_unit_columns
        )

        for raw_result_column in self._raw_result_columns:
            privacy_unit_column = get_privacy_unit_column_with_privacy_check(
                raw_result_column,
                privacy_unit_columns,
                self._from_db_name,
                self._join_db_name,
                self._from_table_name,
                self._join_table_name,
            )
            if (
                self._privacy_unit_column is not None
                and privacy_unit_column is not None
            ):
                is_privacy_check = False
                raise PrivacyConstraintError(
                    "Duplicate privacy unit column in projection",
                    context={
                        "existing": self._privacy_unit_column,
                        "duplicate": privacy_unit_column,
                    },
                    hint="Project the privacy unit column only once",
                )
            if self._privacy_unit_column is None and privacy_unit_column is not None:
                self._privacy_unit_column = privacy_unit_column
                is_privacy_check = True

        if self._used_spark_sql_extensions:
            raise UnsupportedQueryError(
                "Unsupported Spark SQL extensions (`PIVOT`,`UNPIVOT`,`LATERAL VIEW`) "
                "for private tables",
                hint="Remove these extensions or materialize intermediate results "
                "outside private tables",
            )

        logger.debug("SelectCore privacy_check -> %s", is_privacy_check)
        return is_privacy_check

    def is_private_table(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        res = is_private_table_(
            self._from_db_name, self._from_table_name, privacy_unit_columns
        ) or is_private_table_(
            self._join_db_name, self._join_table_name, privacy_unit_columns
        )
        logger.debug("SelectCore is_private_table -> %s", res)
        return res

    def privacy_unit_column(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> str | None:
        logger.debug(
            "SelectCore privacy_unit_column query for %s", self._from_table_name
        )
        if not self.is_private_table(privacy_unit_columns):
            return None
        if self._privacy_unit_column is None:
            raise PrivacyConstraintError(
                "Missing privacy unit column in private table selection",
                hint="Select exactly one column that serves as the privacy unit",
            )
        return self._privacy_unit_column


class SelectStmtParser(Transformer[Any, Any]):
    """This parser is used to parse select statement in SELECT clause.
    It is used to check if the select statement is a private table."""

    def __init__(self) -> None:
        super().__init__()
        self._select_cores: list[SelectCoreParser] = []
        self._compound_operator: str | None = None
        logger.debug("Init SelectStmtParser")

    def select_core(self, items: list[Token | Tree[Any]]):
        logger.debug("Append select_core (items=%s)", len(items))
        selectCoreParser = SelectCoreParser(items)
        self._select_cores.append(selectCoreParser)

    def compound_operator(self, items: list[Token | Tree[Any]]):
        logger.debug("Parse compound operator (count=%s)", len(items))
        if len(items) == 1:
            if not isinstance(items[0], Token):
                raise QueryParseError(
                    "Unexpected element in compound operator",
                    context={"value": str(items[0])},
                    hint="Expect a single token (e.g. UNION)",
                )
            self._compound_operator = items[0].type
        elif len(items) == 2:
            if not isinstance(items[0], Token) or not isinstance(items[1], Token):
                raise QueryParseError(
                    "Unexpected element count in compound operator",
                    context={"values": [str(x) for x in items]},
                    hint="Expect one or two tokens",
                )
            self._compound_operator = items[0].type + " " + items[1].type
        else:
            raise QueryParseError(
                "Too many tokens in compound operator",
                context={"count": len(items)},
                hint="Use one (UNION) or two tokens (UNION ALL)",
            )
        logger.debug("Compound operator -> %s", self._compound_operator)

    @property
    def select_cores(self) -> list[SelectCoreParser]:
        return self._select_cores

    @property
    def compound_operator_(self) -> str | None:
        return self._compound_operator


class CommonTableExpressionParser(PrivacyChecker):
    """This parser is used to parse common table expression in WITH clause.
    It is used to check if the common table expression is a private table."""

    def __init__(self, items: list[Token | Tree[Any]]):
        logger.debug("Init CTE Parser")
        self._table_name = None
        self._select_stmt_parser = None
        isExhausted = False
        items_iterator = iter(items)
        try:
            item = next(items_iterator)
            if not isinstance(item, Tree):
                raise QueryParseError(
                    "Unexpected token at CTE start",
                    context={"value": str(item)},
                    hint="Expect table name tree",
                )
            self._table_name = resolve_instance_value(item, "table_name")
            item = next(items_iterator)
            while isinstance(item, Token):
                match item.type:
                    case "AS" | "NOT" | "MATERIALIZED":
                        pass
                    case _:
                        raise QueryParseError(
                            "Unexpected token in CTE modifier list",
                            context={"token": item.type},
                            hint="Allowed: AS, NOT, MATERIALIZED",
                        )
                item = next(items_iterator)
            if item.data != "select_stmt":
                raise QueryParseError(
                    "Unexpected element in CTE body",
                    context={"node": item.data},
                    hint="Expect select_stmt",
                )
            self._select_stmt_parser = SelectStmtParser()
            self._select_stmt_parser.transform(item)
            next(items_iterator)
        except StopIteration:
            isExhausted = True

        if not isExhausted:
            raise QueryParseError(
                "Unexpected trailing input in CTE clause",
                hint="Remove extra tokens after select_stmt",
            )
        else:
            logger.debug("CTE parsed: table=%s", self._table_name)

    @property
    def table_name(self) -> str:
        logger.debug("CTE.table_name access")
        if self._table_name is None:
            raise QueryParseError(
                "Missing CTE table name",
                hint="Provide an identifier before AS",
            )
        return self._table_name

    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("CTE privacy_check: table=%s", self._table_name)
        if self._select_stmt_parser is None:
            raise QueryParseError(
                "Missing select statement parser for CTE",
                context={"cte": self._table_name},
                hint="Ensure the CTE defines a SELECT statement",
            )

        for select_core in self._select_stmt_parser.select_cores:
            if not select_core.privacy_check(privacy_unit_columns):
                raise PrivacyConstraintError(
                    "Missing privacy unit column in CTE",
                    context={"cte": self._table_name},
                    hint="Select exactly one privacy unit column in private table CTEs",
                )
        logger.debug("CTE privacy_check passed")
        return True

    def is_private_table(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("CTE is_private_table query: %s", self._table_name)
        if self._select_stmt_parser is None:
            raise QueryParseError(
                "Missing select statement parser for CTE",
                context={"cte": self._table_name},
                hint="Ensure the CTE defines a SELECT statement",
            )

        for select_core in self._select_stmt_parser.select_cores:
            if select_core.is_private_table(privacy_unit_columns):
                return True
        return False

    def privacy_unit_column(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> str | None:
        logger.debug("CTE privacy_unit_column query: %s", self._table_name)
        if self._select_stmt_parser is None:
            raise QueryParseError(
                "Missing select statement parser for CTE",
                context={"cte": self._table_name},
                hint="Ensure the CTE defines a SELECT statement",
            )
        if not self.is_private_table(privacy_unit_columns):
            return None
        if len(self._select_stmt_parser.select_cores) == 1:
            return self._select_stmt_parser.select_cores[0].privacy_unit_column(
                privacy_unit_columns
            )
        elif len(self._select_stmt_parser.select_cores) == 2:
            first_col = self._select_stmt_parser.select_cores[0].privacy_unit_column(
                privacy_unit_columns
            )
            second_col = self._select_stmt_parser.select_cores[1].privacy_unit_column(
                privacy_unit_columns
            )
            if first_col != second_col or first_col is None or second_col is None:
                raise PrivacyConstraintError(
                    "Mismatched privacy unit columns across set operation",
                    context={
                        "cte": self._table_name,
                        "first": first_col,
                        "second": second_col,
                    },
                    hint="Ensure both sides project the same privacy unit column",
                )
            return first_col
        else:
            raise QueryParseError(
                "Unsupported number of SELECT cores in CTE",
                context={
                    "cte": self._table_name,
                    "count": len(self._select_stmt_parser.select_cores),
                },
                hint="Use one or two SELECT cores",
            )
