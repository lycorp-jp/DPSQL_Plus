import logging
from collections.abc import Iterator
from typing import Any

from lark.lexer import Token
from lark.tree import Tree

from ..aggregation import Aggregation, AggregationColumn
from ..errors import (
    PrivacyConstraintError,
    QueryParseError,
    UnsupportedQueryError,
    ValidationError,
)
from .sqlgrammar import PrivacyChecker
from .utils import (
    get_privacy_definition,
    is_private_table_,
    resolve_instance_3_values,
    resolve_instance_value,
)

logger = logging.getLogger(__name__)


def _is_matched_table_name(
    table_name_value: str, table_name: str, table_name_alias: str | None
):
    logger.debug(
        "Check table qualifier: value=%s table=%s alias=%s",
        table_name_value,
        table_name,
        table_name_alias,
    )
    if table_name_value != table_name and table_name_value != table_name_alias:
        raise QueryParseError(
            "Table qualifier does not match FROM table",
            context={
                "value": table_name_value,
                "table": table_name,
                "alias": table_name_alias,
            },
            hint="Qualify columns with the selected table or its alias",
        )


def _parse_final_aggregate_function_argument(
    item: Tree[Any], table_name: str, table_name_alias: str | None = None
) -> tuple[list[str], bool, list[float]]:
    """
    Parse the final aggregate function argument from the given item.

    Args:
        item (Tree[Any]): The tree item to parse.
        table_name (str): The name of the table.
        table_name_alias (str | None): The alias of the table, if any.

    Returns:
        tuple: A tuple containing a list of column names, a boolean indicating if
        DISTINCT is used, and a list of parameters for aggregate function.
    """
    logger.debug(
        "Parse final aggregate arg: table=%s alias=%s children=%s",
        table_name,
        table_name_alias,
        len(item.children),
    )
    item_iterator = iter(item.children)
    column_names: list[str] = []
    is_distinct = False
    parameters: list[float] = []
    isExhausted = False

    try:
        item = next(item_iterator)
        while True:
            if isinstance(item, Token):
                match item.type:
                    case "DISTINCT":
                        is_distinct = True
                    case "ASTERISK":
                        column_names.append("*")
                    case _:
                        raise QueryParseError(
                            "Unexpected token in aggregate argument",
                            context={"token_type": item.type, "value": str(item)},
                            hint="Use DISTINCT, * or column / literal expressions",
                        )
            else:
                match item.data:
                    case "column_name":
                        column_names.append(resolve_instance_value(item, "column_name"))
                    case "table_name":
                        table_name_value = resolve_instance_value(item, "table_name")
                        _is_matched_table_name(
                            table_name_value, table_name, table_name_alias
                        )
                    case "literal_value":
                        parameters.append(
                            float(resolve_instance_value(item, "literal_value"))
                        )
                    case _:
                        raise QueryParseError(
                            "Unexpected element in aggregate argument",
                            context={"node": item.data},
                            hint="Allowed: column_name, table_name, literal_value",
                        )
            item = next(item_iterator)
    except StopIteration:
        isExhausted = True

    if isExhausted:
        return column_names, is_distinct, parameters
    else:
        raise QueryParseError(
            "Unexpected trailing input in aggregate function argument",
            hint="Remove extra tokens after the last argument element",
        )


def get_final_result_column(
    items: list[Token | Tree[Any]], table_name: str, table_name_alias: str | None
) -> tuple[list[str], str | None, bool, list[float]]:
    """
    Extracts the final result column from the provided items.

    Args:
        items (list[Token | Tree[Any]]): The list of items to parse.
        table_name (str): The name of the table.
        table_name_alias (str | None): The alias of the table, if any.

    Returns:
        tuple: A tuple containing the column name(s), aggregate function name,
        whether DISTINCT is used, and a list of parameters for aggregate funtion.
    """
    logger.debug(
        "Get final result column: items=%s table=%s alias=%s",
        len(items),
        table_name,
        table_name_alias,
    )
    aggregate_functions_column_names: list[str] = []
    column_name: str | None = None
    aggregate_function_name = None
    is_distinct = False
    parameters: list[float] = []
    isExhausted = False

    item_iterator = iter(items)
    try:
        item = next(item_iterator)
        while True:
            if not isinstance(item, Tree):
                raise QueryParseError(
                    "Unexpected token in result column",
                    context={"value": str(item), "type": type(item).__name__},
                    hint="Expect syntax tree nodes for result column components",
                )
            match item.data:
                case "table_name":
                    table_name_value = resolve_instance_value(item, "table_name")
                    if (
                        table_name_value != table_name
                        and table_name_value != table_name_alias
                    ):
                        raise QueryParseError(
                            "Table qualifier does not match FROM table",
                            context={
                                "value": table_name_value,
                                "table": table_name,
                                "alias": table_name_alias,
                            },
                            hint="Qualify columns with the selected table or its alias",
                        )
                case "column_name":
                    column_name = resolve_instance_value(item, "column_name")
                case "final_aggregate_function_name":
                    aggregate_function_name = resolve_instance_value(
                        item, "final_aggregate_function_name", True
                    )
                case "final_aggregate_function_argument":
                    aggregate_functions_column_names, is_distinct, parameters = (
                        _parse_final_aggregate_function_argument(
                            item, table_name, table_name_alias
                        )
                    )
                case _:
                    raise QueryParseError(
                        "Unexpected element in result column",
                        context={"node": item.data},
                        hint="Allowed: table_name, column_name, "
                        "final_aggregate_function_*",
                    )
            item = next(item_iterator)
    except StopIteration:
        isExhausted = True

    if isExhausted:
        return (
            aggregate_functions_column_names if column_name is None else [column_name],
            aggregate_function_name,
            is_distinct,
            parameters,
        )
    raise QueryParseError(
        "Unexpected trailing input in result column",
        hint="Remove extra tokens after the last result column element",
    )


def parse_final_result_column(
    item: Tree[Any], table_name: str, table_name_alias: str | None
) -> tuple[str | None, list[str], str | None, list[float]]:
    """
    Parse the final result column from the given item.

    Args:
        item (Tree[Any]): The tree item to parse.
        table_name (str): The name of the table.
        table_name_alias (str | None): The alias of the table, if any.

    Returns:
        tuple: A tuple containing the aggregate function name, column name(s),
        column name alias, and a list of parameters for aggregate function.

    Raises:
        PrivSQLError: If the input is invalid or does not match the expected format.
    """
    logger.debug(
        "Parse final result column: table=%s alias=%s children=%s",
        table_name,
        table_name_alias,
        len(item.children),
    )
    if len(item.children) != 1 and len(item.children) != 3:
        raise QueryParseError(
            "Unexpected child count in final result column",
            context={
                "count": len(item.children),
                "children": [str(c) for c in item.children],
            },
            hint="Use 1 child (expression) or 3 (expression AS alias)",
        )

    aggregate_function_name = None
    is_distinct = False
    column_name_alias = None

    if isinstance(item.children[0], Token):
        if item.children[0].type != "ASTERISK":
            raise QueryParseError(
                "Unexpected token in final result column",
                context={"token_type": item.children[0].type},
                hint="Use * or an expression tree",
            )
        return aggregate_function_name, ["ASTERISK"], None, []

    final_expr: Tree[Any] = item.children[0]
    if final_expr.data != "final_expr":
        raise QueryParseError(
            "Unexpected element in final result column",
            context={"node": final_expr.data},
            hint="Expected final_expr tree",
        )

    column_names, aggregate_function_name, is_distinct, parameters = (
        get_final_result_column(
            final_expr.children,
            table_name,
            table_name_alias,
        )
    )

    if is_distinct:
        if aggregate_function_name is None:
            raise ValidationError(
                "DISTINCT is allowed only inside an aggregate function",
                hint="Wrap DISTINCT inside an aggregate (e.g. COUNT(DISTINCT col))",
            )
        aggregate_function_name = f"{aggregate_function_name}_DISTINCT"

    if len(item.children) == 3:
        if aggregate_function_name is None:
            raise ValidationError(
                "Alias on raw column without aggregate is disallowed",
                hint="Use an aggregate or remove the alias",
            )
        if (
            not isinstance(item.children[2], Tree)
            or item.children[2].data != "column_name_alias"
        ):
            raise QueryParseError(
                "Unexpected element for column alias",
                context={"child": str(item.children[2].data)},
                hint="Provide a column_name_alias tree",
            )
        column_name_alias = resolve_instance_value(
            item.children[2], "column_name_alias"
        )

    logger.debug(
        "Parsed final result: agg=%s columns=%s alias=%s params=%s",
        aggregate_function_name,
        column_names,
        column_name_alias,
        parameters,
    )
    return aggregate_function_name, column_names, column_name_alias, parameters


class FinalSelectCoreParser(PrivacyChecker):
    """
    Parser for the final select core in SQL queries.

    This class parses the final select core from a list of items and checks
    if the final result columns and group by columns satisfy the privacy unit columns.
    """

    def __init__(self, items: list[Token | Tree[Any]]):
        logger.debug("Init FinalSelectCoreParser (items=%s)", len(items))
        self._final_result_columns: list[AggregationColumn] = []
        self._group_by_columns: list[str] = []
        self._db_name: str | None = None
        self._table_name: str | None = None
        self._table_name_alias: str | None = None
        raw_final_result_columns: list[Tree[Any]] = []
        raw_privacy_params: list[tuple[Tree[Any], Tree[Any]]] = []
        self._privacy_params: dict[str, float | int] | None = None

        self._parse_final_select_core(
            items, raw_final_result_columns, raw_privacy_params
        )
        logger.debug(
            "FinalSelectCore parsed: table=%s alias=%s group_by=%s",
            self._table_name,
            self._table_name_alias,
            self._group_by_columns,
        )

        if self._table_name is None:
            raise QueryParseError(
                "Missing table name in FINAL SELECT core",
                hint="Provide a FROM clause with a table_expr",
            )

        for final_result_column in raw_final_result_columns:
            aggregate_func_name, column_name, column_name_alias, parameters = (
                parse_final_result_column(
                    final_result_column, self._table_name, self._table_name_alias
                )
            )

            # Convert string to Aggregation enum
            aggregation_type = Aggregation.from_str(aggregate_func_name)
            agg_column = AggregationColumn(
                aggregation_type=aggregation_type,
                columns=column_name,
                alias=column_name_alias,
                parameters=parameters,
            )
            self._final_result_columns.append(agg_column)
            logger.debug(
                "Append result column: agg=%s cols=%s alias=%s params=%s",
                aggregation_type.name,
                column_name,
                column_name_alias,
                parameters,
            )
        if len(raw_privacy_params) > 0:
            logger.debug("Parsing privacy params: count=%s", len(raw_privacy_params))
            self._privacy_params = self._parse_privacy_params(raw_privacy_params)

    def _parse_privacy_params(
        self, raw_privacy_params: list[tuple[Tree[Any], Tree[Any]]]
    ) -> dict[str, float | int]:
        logger.debug("FinalSelectCore: _parse_privacy_params start")
        privacy_params: dict[str, float | int] = {}
        for privacy_param, literal_value in raw_privacy_params:
            param_name = resolve_instance_value(privacy_param, "privacy_param", True)
            param_value = resolve_instance_value(literal_value, "literal_value")
            match param_name:
                case "EPSILON" | "DELTA":
                    privacy_params[param_name] = float(param_value)
                case "MIN_FREQUENCY" | "CONTRIBUTION_BOUND":
                    privacy_params[param_name] = int(param_value)
                case _:
                    raise ValidationError(
                        "Unsupported privacy parameter",
                        context={"param": param_name},
                        hint="Allowed: EPSILON, DELTA, MIN_FREQUENCY,"
                        " CONTRIBUTION_BOUND",
                    )
        match get_privacy_definition(privacy_params):
            case "DP":
                return privacy_params
            case "DP_MIN_FREQUENCY":
                return privacy_params
            case "MIN_FREQUENCY":
                raise UnsupportedQueryError(
                    "Privacy definition MIN_FREQUENCY is not supported",
                    hint="Use DP or supply DP-compatible parameters",
                )
            case _:
                raise ValidationError(
                    "Invalid privacy parameters",
                    context={"params": privacy_params},
                    hint="Provide EPSILON and DELTA (optionally MIN_FREQUENCY, "
                    "CONTRIBUTION_BOUND)",
                )

    def _parse_final_select_core(
        self,
        items: list[Token | Tree[Any]],
        raw_final_result_columns: list[Tree[Any]],
        raw_privacy_params: list[tuple[Tree[Any], Tree[Any]]],
    ):
        logger.debug("Parse final_select_core: items=%s", len(items))
        isExhausted = False
        items_iterator = iter(items)
        try:
            item = next(items_iterator)
            while isinstance(item, Token):
                match item.type:
                    case "SELECT":
                        item = next(items_iterator)
                        if isinstance(item, Token) and item.type == "PRIVATE_QUERY":
                            item, items_iterator = self._parse_private_query(
                                items_iterator, raw_privacy_params
                            )
                        while (
                            isinstance(item, Tree)
                            and item.data == "final_result_column"
                        ):
                            raw_final_result_columns.append(item)
                            item = next(items_iterator)
                    case "GROUP":
                        item, items_iterator = self._parse_group_by(items_iterator)
                    case "FROM":
                        item = next(items_iterator)
                        if not (isinstance(item, Tree) and item.data == "table_expr"):
                            raise QueryParseError(
                                "Unexpected element after `FROM`",
                                context={"value": str(item)},
                                hint="Expect table_expr",
                            )
                        self._db_name, self._table_name, self._table_name_alias = (
                            resolve_instance_3_values(item, "table_expr")
                        )
                        item = next(items_iterator)
                    case _:
                        raise QueryParseError(
                            "Unexpected clause token in FINAL SELECT core",
                            context={"token": item.type},
                            hint="Allowed: SELECT, GROUP, FROM",
                        )
        except StopIteration:
            isExhausted = True

        if not isExhausted:
            raise QueryParseError(
                "Unexpected trailing input in FINAL SELECT core",
                hint="Remove extra tokens after the last clause",
            )

    def _parse_private_query(
        self,
        items_iterator: Iterator[Token | Tree[Any]],
        raw_privacy_params: list[tuple[Tree[Any], Tree[Any]]],
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse PRIVATE_QUERY OPTIONS ...")
        item = next(items_iterator)
        if not (isinstance(item, Token) and item.type == "OPTIONS"):
            raise QueryParseError(
                "Missing `OPTIONS` token after PRIVATE_QUERY",
                context={"value": str(item)},
                hint="Use PRIVATE_QUERY OPTIONS param=value ...",
            )
        privacy_param = next(items_iterator)
        while isinstance(privacy_param, Tree) and privacy_param.data == "privacy_param":
            literal_value = next(items_iterator)
            if (
                isinstance(literal_value, Tree)
                and literal_value.data == "literal_value"
            ):
                raw_privacy_params.append((privacy_param, literal_value))
            else:
                raise QueryParseError(
                    "Missing literal value for privacy parameter",
                    context={"value": str(literal_value)},
                    hint="Provide numeric literal after parameter name",
                )
            privacy_param = next(items_iterator)
        item = privacy_param
        logger.debug("PRIVATE_QUERY parsed: params=%s", len(raw_privacy_params))
        return item, items_iterator

    def _parse_group_by(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("Parse GROUP BY ...")
        item = next(items_iterator)
        if isinstance(item, Token) and item.type != "BY":
            raise QueryParseError(
                "Missing BY token in GROUP BY clause",
                context={"value": str(item)},
                hint="Use GROUP BY <columns>",
            )
        item = next(items_iterator)
        while isinstance(item, Tree):
            match item.data:
                case "table_name":
                    table_name = resolve_instance_value(item, "table_name")
                    if (
                        self._table_name != table_name
                        and self._table_name_alias != table_name
                    ):
                        raise QueryParseError(
                            "Table qualifier in GROUP BY does not match FROM table",
                            context={
                                "value": table_name,
                                "table": self._table_name,
                                "alias": self._table_name_alias,
                            },
                            hint="Qualify columns with the selected table or its alias",
                        )
                case "column_name":
                    self._group_by_columns.append(
                        resolve_instance_value(item, "column_name")
                    )
                case _:
                    raise QueryParseError(
                        "Unexpected element in `GROUP BY` list",
                        context={"node": item.data},
                        hint="Provide column_name (optionally qualified)",
                    )
            item = next(items_iterator)
        logger.debug("GROUP BY parsed: columns=%s", self._group_by_columns)
        return item, items_iterator

    def _check_group_by_columns(self, privacy_column: str) -> None:
        logger.debug("Check GROUP BY against privacy column: %s", privacy_column)
        for group_by_column_name in self._group_by_columns:
            if privacy_column == group_by_column_name:
                raise PrivacyConstraintError(
                    "Privacy unit column in `GROUP BY` is disallowed",
                    context={"column": group_by_column_name},
                    hint="Remove the privacy unit column from GROUP BY",
                )

    def _check_final_result_columns(self, privacy_column: str) -> None:
        logger.debug(
            "Check final result columns against privacy column: %s", privacy_column
        )
        for agg_column in self._final_result_columns:
            columns = agg_column.columns
            aggregation_type = agg_column.aggregation_type
            # No aggregate function
            match aggregation_type:
                case Aggregation.NONE:
                    if len(columns) != 1:
                        raise ValidationError(
                            "Expected a single column for non-aggregate projection",
                            context={"columns": columns},
                            hint="Project exactly one column or use an aggregate",
                        )
                    column = columns[0]
                    if len(self._group_by_columns) == 0:
                        raise ValidationError(
                            "Non-aggregate column without `GROUP BY` is disallowed",
                            context={"column": column},
                            hint="Add GROUP BY or apply an aggregate function",
                        )
                    if column == "ASTERISK":
                        raise ValidationError(
                            "`*` wildcard is disallowed in final result column",
                            hint="List columns explicitly",
                        )
                    if privacy_column == column:
                        raise PrivacyConstraintError(
                            "Privacy unit column as raw projection is disallowed",
                            context={"column": column},
                            hint="Aggregate or omit the privacy unit column",
                        )
                    if column not in self._group_by_columns:
                        raise ValidationError(
                            "Projected column not in `GROUP BY`",
                            context={"column": column},
                            hint="Add the column to GROUP BY or aggregate it",
                        )
                # Allow COUNT and COUNT_DISTINCT functions
                case Aggregation.COUNT | Aggregation.COUNT_DISTINCT:
                    continue
                case _:
                    # Other aggregate functions
                    if privacy_column in columns:
                        raise PrivacyConstraintError(
                            "Privacy unit column inside aggregate is disallowed",
                            context={
                                "aggregation": aggregation_type.name,
                                "columns": columns,
                                "privacy_unit_column": privacy_column,
                            },
                            hint="Exclude the privacy unit column from aggregates",
                        )

    def _check_group_by_in_select(self) -> None:
        """Check that all GROUP BY keys are included in SELECT clause"""
        logger.debug("Verify GROUP BY columns are in SELECT")
        if not self._group_by_columns:
            return  # No GROUP BY, so no check needed

        # Get non-aggregate columns from SELECT clause
        select_columns = []
        for agg_column in self._final_result_columns:
            if agg_column.aggregation_type == Aggregation.NONE:
                select_columns.extend(agg_column.columns)

        # Check if each GROUP BY column is included in SELECT clause
        for group_by_column in self._group_by_columns:
            if group_by_column not in select_columns:
                raise ValidationError(
                    "GROUP BY column must be included in SELECT clause",
                    context={"column": group_by_column},
                    hint="Add the GROUP BY column to SELECT clause",
                )

    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("FinalSelectCore privacy_check")
        if self._table_name is None:
            raise QueryParseError(
                "Missing table name in FINAL SELECT core",
                hint="Provide a FROM clause with a table_expr",
            )

        if not self.is_private_table(privacy_unit_columns):
            return True

        privacy_column = privacy_unit_columns[self._db_name][self._table_name]
        self._check_group_by_columns(privacy_column)
        self._check_final_result_columns(privacy_column)
        self._check_group_by_in_select()

        logger.debug("Privacy check passed")
        return True

    def privacy_unit_column(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> str | None:
        logger.debug("FinalSelectCore privacy_unit_column query")
        if not self.is_private_table(privacy_unit_columns):
            return None
        return privacy_unit_columns[self.db_name][self.table_name]

    def is_private_table(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        res = is_private_table_(self._db_name, self._table_name, privacy_unit_columns)
        logger.debug("FinalSelectCore is_private_table -> %s", res)
        return res

    @property
    def db_name(self) -> str | None:
        return self._db_name

    @property
    def table_name(self) -> str:
        if self._table_name is None:
            raise QueryParseError(
                "Table name not found",
                hint="Provide a FROM clause with a table_expr",
            )
        return self._table_name

    @property
    def privacy_params(self) -> dict[str, float | int] | None:
        return self._privacy_params

    @property
    def group_by_columns(self) -> list[str]:
        return self._group_by_columns

    @property
    def final_result_columns(self) -> list[AggregationColumn]:
        return self._final_result_columns


class FinalSelectStmtParser(PrivacyChecker):
    """Parser for the final select statement in SQL queries.

    This class parses the final select statement from a list of items
    and checks if the final result columns, group by columns, and ordering terms
    satisfy the privacy unit columns.
    """

    def __init__(self, items: list[Token | Tree[Any]]):
        logger.debug("Init FinalSelectStmtParser (items=%s)", len(items))
        self._limit: int | None = None
        self._offset: int | None = None
        self._ordering_terms: list[dict[str, str | None]] = []
        self._parse_final_select_stmt(items)

    def _parse_final_select_stmt(self, items: list[Token | Tree[Any]]) -> None:
        isExhausted = False
        items_iterator = iter(items)
        try:
            item = next(items_iterator)
            if not (isinstance(item, Tree) and item.data == "final_select_core"):
                raise QueryParseError(
                    "Missing `final_select_core` at start of FINAL SELECT",
                    context={"value": str(item)},
                    hint="Begin with final_select_core tree",
                )
            self._final_select_core_parser = FinalSelectCoreParser(item.children)
            item = next(items_iterator)
            while isinstance(item, Token):
                match item.type:
                    case "ORDER":
                        item = next(items_iterator)
                        if not isinstance(item, Token) or item.type != "BY":
                            raise QueryParseError(
                                "Missing BY token after ORDER",
                                context={"value": str(item)},
                                hint="Use ORDER BY <ordering_terms>",
                            )
                        item = next(items_iterator)
                        while isinstance(item, Tree) and item.data == "ordering_term":
                            # Process ordering terms if needed
                            self._ordering_terms.append(
                                self._parse_ordering_terms(item.children)
                            )
                            item = next(items_iterator)
                    case "LIMIT":
                        # Handle LIMIT clause if needed
                        item = next(items_iterator)
                        self._limit = self._get_int_value(item)
                        item = next(items_iterator)
                        if not isinstance(item, Token) or item.type != "OFFSET":
                            raise QueryParseError(
                                "Missing OFFSET token after LIMIT",
                                context={"value": str(item)},
                                hint="Use LIMIT n OFFSET m",
                            )
                        item = next(items_iterator)
                        self._offset = self._get_int_value(item)
                        item = next(items_iterator)
                    case _:
                        raise QueryParseError(
                            "Unexpected clause token in FINAL SELECT",
                            context={"token": item.type},
                            hint="Allowed: ORDER, LIMIT",
                        )
        except StopIteration:
            isExhausted = True

        if not isExhausted:
            raise QueryParseError(
                "Unexpected trailing input in FINAL SELECT clause",
                hint="Remove extra tokens after the last clause",
            )

    def _get_int_value(self, item: Token | Tree[Any]) -> int:
        if not isinstance(item, Token) or item.type != "INT":
            raise QueryParseError(
                "Expected integer literal",
                context={"value": str(item)},
                hint="Use an INT token for numeric limits/offsets",
            )
        return int(item.value)

    def _parse_column_name_for_ordering(
        self, item: Tree[Any], table_name: str
    ) -> str | None:
        column_name = None
        match item.data:
            case "column_name":
                column_name = resolve_instance_value(item, "column_name")
            case "table_name":
                table_name = resolve_instance_value(item, "table_name")
                if table_name != self._final_select_core_parser.table_name:
                    raise QueryParseError(
                        "Table qualifier in ORDER BY does not match FROM table",
                        context={
                            "value": table_name,
                            "table": self._final_select_core_parser.table_name,
                        },
                        hint="Qualify columns with the selected table or its alias",
                    )
            case _:
                raise QueryParseError(
                    "Unexpected element in ordering term",
                    context={"node": item.data},
                    hint="Allowed: column_name or table_name",
                )
        return column_name

    def _parse_ordering_terms(
        self, children: list[Token | Tree[Any]]
    ) -> dict[str, str | None]:
        logger.debug("Parse ordering_terms start (children=%s)", len(children))
        ordering_terms: dict[str, str | None] = {}
        ordering_terms["order"] = None
        ordering_terms["column_name"] = None
        items_iterator = iter(children)
        item = next(items_iterator)
        isExhausted = False
        try:
            while True:
                if isinstance(item, Token):
                    match item.type:
                        case "ASC" | "DESC":
                            ordering_terms["order"] = item.type
                        case _:
                            raise QueryParseError(
                                "Unexpected token in ordering term",
                                context={"token": item.type},
                                hint="Use ASC or DESC for ordering direction",
                            )
                else:
                    ordering_terms["column_name"] = (
                        self._parse_column_name_for_ordering(
                            item, self._final_select_core_parser.table_name
                        )
                    )
                item = next(items_iterator)
        except StopIteration:
            isExhausted = True

        if isExhausted:
            logger.debug("Parsed ordering_term: %s", ordering_terms)
            return ordering_terms
        else:
            raise QueryParseError(
                "Unexpected trailing input in ORDER BY clause",
                hint="Remove extra tokens after the last ordering term",
            )

    @property
    def final_select_core_parser(self) -> FinalSelectCoreParser:
        return self._final_select_core_parser

    @property
    def limit(self) -> int | None:
        return self._limit

    @property
    def offset(self) -> int | None:
        return self._offset

    @property
    def ordering_terms(self) -> list[dict[str, str | None]]:
        return self._ordering_terms

    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("FinalSelectStmtParser privacy_check delegated")
        return self._final_select_core_parser.privacy_check(privacy_unit_columns)

    def privacy_unit_column(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> str | None:
        logger.debug("FinalSelectStmtParser privacy_unit_column delegated")
        return self._final_select_core_parser.privacy_unit_column(privacy_unit_columns)

    def is_private_table(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        logger.debug("FinalSelectStmtParser is_private_table delegated")
        return self._final_select_core_parser.is_private_table(privacy_unit_columns)
