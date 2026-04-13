import logging
from typing import Any

from lark.lexer import Token
from lark.tree import Tree

from ..errors import PrivacyConstraintError, QueryParseError
from .utils import resolve_instance_value

logger = logging.getLogger(__name__)


def _handle_table_name(node: Tree[Any], contents: list[str], counter: int):
    logger.debug(
        "_handle_table_name: counter=%s node=%s", counter, getattr(node, "data", None)
    )
    if counter != 0 and counter != 3:
        raise QueryParseError(
            "Unexpected position for table name",
            context={"node": str(node), "position": counter},
            hint="table_name allowed at positions 0 or 3",
        )
    table_name = resolve_instance_value(node, "table_name")
    contents.append(table_name)


def _handle_column_name(
    node: Tree[Any],
    contents: list[str],
    counter: int,
    ans: list[tuple[str, str, str, str]],
):
    logger.debug(
        "_handle_column_name: counter=%s node=%s contents_len=%s",
        counter,
        getattr(node, "data", None),
        len(contents),
    )
    if counter != 1 and counter != 4:
        raise QueryParseError(
            "Unexpected position for column name",
            context={"node": str(node), "position": counter},
            hint="column_name allowed at positions 1 or 4",
        )
    column_name = resolve_instance_value(node, "column_name")
    contents.append(column_name)
    if counter == 4:
        if len(contents) != 4:
            raise QueryParseError(
                "Incomplete join expression tuple",
                context={"contents": contents},
                hint="Expect exactly 4 elements: from_table, from_column, "
                "join_table, join_column",
            )
        ans.append((contents[0], contents[1], contents[2], contents[3]))
        contents.clear()
    logger.debug(
        "_handle_column_name: tuple_appended=%s total=%s", counter == 4, len(ans)
    )


def _handle_binary_operator(node: Tree[Any], counter: int):
    logger.debug("_handle_binary_operator: counter=%s", counter)
    if counter != 2 and counter != 5:
        raise QueryParseError(
            "Unexpected position for binary operator",
            context={"node": str(node), "position": counter},
            hint="Operator at positions 2 or 5 only",
        )
    binary_op = resolve_instance_value(node, "binary_operator", istype=True)
    if binary_op == "EQUAL":
        if counter != 2:
            raise QueryParseError(
                "Unexpected position for `=` operator",
                context={"position": counter},
                hint="First operator (position 2) must be `=`",
            )
    elif binary_op == "AND":
        if counter != 5:
            raise QueryParseError(
                "Unexpected position for `AND` operator",
                context={"position": counter},
                hint="Second operator (position 5) must be AND",
            )
    else:
        raise QueryParseError(
            "Unsupported binary operator",
            context={"operator": binary_op},
            hint="Only `=` and AND are allowed",
        )
    logger.debug("_handle_binary_operator: validated")


def private_join_constraint(item: Tree[Any]) -> list[tuple[str, str, str, str]]:
    """Parse the private join constraint from the given Tree item.
    The expected structure is:
    expr: table_name "." column_name = table_name "." column_name
         (AND table_name "." column_name = table_name "." column_name)*
    """
    logger.debug("private_join_constraint: start item=%s", getattr(item, "data", None))
    base_err_message = (
        "Private join constraints only allow equi-joins and "
        "conjunctions using AND. "
        "For example: table1.col1 = table2.col1 AND "
        "table1.col2 = table2.col2"
    )
    counter = 0

    def depth_first_search(
        node: Tree[Any],
        contents: list[str],
        ans: list[tuple[str, str, str, str]],
    ) -> None:
        nonlocal counter
        if isinstance(node, Token):
            raise QueryParseError(
                "Unexpected token in join constraint",
                context={"token": str(node)},
                hint="Expect expression tree nodes, not raw tokens",
            )
        for child in node.children:
            try:
                match node.data:
                    case "expr":
                        depth_first_search(child, contents, ans)
                    case "table_name":
                        _handle_table_name(node, contents, counter)
                        counter += 1
                        counter %= 6
                    case "column_name":
                        _handle_column_name(node, contents, counter, ans)
                        counter += 1
                        counter %= 6
                    case "binary_operator":
                        _handle_binary_operator(node, counter)
                        counter += 1
                        counter %= 6
                    case _:
                        raise QueryParseError(
                            "Unexpected node type in join constraint",
                            context={"node_type": node.data},
                            hint="Allowed: expr, table_name, column_name, "
                            "binary_operator",
                        )
            except QueryParseError as e:
                raise QueryParseError(
                    f"{str(e)} {base_err_message}",
                    context=e.context,
                    hint=e.hint,
                ) from e

    ans: list[tuple[str, str, str, str]] = []
    depth_first_search(item, [], ans)
    logger.debug("private_join_constraint: result_count=%s", len(ans))
    return ans


class PrivateJoinConstraintParser:
    """Parser for private join constraints in SQL queries.

    This class parses the private join constraints from a list of items
    and checks if the constraints satisfy the privacy unit columns.
    """

    def __init__(
        self,
        items: list[Token | Tree[Any]],
        from_db_name: str | None,
        join_db_name: str | None,
        from_table_name: str,
        join_table_name: str,
        from_table_name_alias: str | None = None,
        join_table_name_alias: str | None = None,
    ):
        """
        Parse the private join constraints from the given items.

        Args:
            items (list[Token | Tree[Any]]): The list of items to parse.
            from_db_name (str | None): The name of the database for the from table.
            join_db_name (str | None): The name of the database for the join table.
            from_table_name (str): The name of the from table.
            join_table_name (str): The name of the join table.
            from_table_name_alias (str | None): The alias for the from table.
            join_table_name_alias (str | None): The alias for the join table.
        """
        logger.debug(
            "Init PrivateJoinConstraintParser: from=%s.%s join=%s.%s",
            from_db_name,
            from_table_name,
            join_db_name,
            join_table_name,
        )
        logger.debug(
            "Aliases: from_alias=%s join_alias=%s items_len=%s",
            from_table_name_alias,
            join_table_name_alias,
            len(items),
        )
        self._equijoin_column_name: list[tuple[str, str, str, str]] = []
        self._from_db_name = from_db_name
        self._join_db_name = join_db_name
        self._from_table_name = from_table_name
        self._join_table_name = join_table_name
        self._from_table_name_alias = from_table_name_alias
        self._join_table_name_alias = join_table_name_alias
        isExhausted = False
        try:
            items_iterator = iter(items)
            item = next(items_iterator)
            if not isinstance(item, Token):
                raise QueryParseError(
                    "Unexpected element (expected token)",
                    context={"element": str(item)},
                    hint="Join constraint must start with ON or USING token",
                )
            match item.type:
                case "ON":
                    logger.debug("Join constraint via ON")
                    item = next(items_iterator)
                    if not isinstance(item, Tree):
                        raise QueryParseError(
                            "Unexpected element after ON (expected expression Tree)",
                            context={"element": str(item)},
                            hint="Provide an equality expression after ON",
                        )
                    if item.data != "expr":
                        raise QueryParseError(
                            "Unexpected tree type after `ON`",
                            context={"tree_type": item.data},
                            hint="Expect expr tree",
                        )
                    self._equijoin_column_name = private_join_constraint(item)
                    logger.debug(
                        "ON parsed: equijoin_count=%s", len(self._equijoin_column_name)
                    )
                    next(items_iterator)
                case "USING":
                    logger.debug("Join constraint via USING")
                    item = next(items_iterator)
                    while isinstance(item, Tree):
                        match item.data:
                            case "column_name":
                                column_name = resolve_instance_value(
                                    item, "column_name"
                                )
                                self._equijoin_column_name.append(
                                    (
                                        from_table_name,
                                        column_name,
                                        join_table_name,
                                        column_name,
                                    )
                                )
                                logger.debug("USING column appended: %s", column_name)
                            case _:
                                raise QueryParseError(
                                    "Unexpected element in `USING` clause",
                                    context={"tree_type": item.data},
                                    hint="Only column_name entries allowed",
                                )
                        item = next(items_iterator)
                case _:
                    logger.error("Unexpected starting token: %s", item.type)
                    raise QueryParseError(
                        "Unexpected starting token in join constraint",
                        context={"token_type": item.type},
                        hint="Expected ON or USING",
                    )
        except StopIteration:
            isExhausted = True

        if not isExhausted:
            logger.error("Unexpected trailing input in join constraint")
            raise QueryParseError(
                "Unexpected trailing input in join constraint",
                hint="Remove extra tokens after ON/USING clause",
            )

    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        """
        Check if the private join constraint satisfies the privacy unit columns.

        Args:
            privacy_unit_columns (dict[str | None, dict[str, str]]):
                The privacy unit columns of the database.
                ex) {"db1", {"table1": "column1"}}
                if there is no privacy unit column in a table, give {}

        Returns:
            bool: True if the private join constraint satisfies
            the privacy unit columns,
                    False otherwise.
        """
        logger.debug("PrivateJoinConstraintParser.privacy_check start")
        logger.debug("Equi-join tuples=%s", len(self._equijoin_column_name))
        privacy_check = False
        for columns in self._equijoin_column_name:
            from_table, from_column, join_table, join_column = None, None, None, None
            tmp_from_table, tmp_from_column, tmp_join_table, tmp_join_column = columns
            if tmp_from_table in (self._from_table_name, self._from_table_name_alias):
                from_table = self._from_table_name
                from_column = tmp_from_column

            if tmp_join_table in (self._join_table_name, self._join_table_name_alias):
                join_table = self._join_table_name
                join_column = tmp_join_column

            if tmp_from_table in (self._join_table_name, self._join_table_name_alias):
                join_table = self._join_table_name
                join_column = tmp_from_column

            if tmp_join_table in (self._from_table_name, self._from_table_name_alias):
                from_table = self._from_table_name
                from_column = tmp_join_column

            if from_table is None or join_table is None:
                logger.error("Failed to resolve participating tables: raw=%s", columns)
                raise PrivacyConstraintError(
                    "Failed to resolve table names in join constraint",
                    context={
                        "from_table": self._from_table_name,
                        "join_table": self._join_table_name,
                        "raw": columns,
                    },
                    hint="Ensure only participating tables or aliases are referenced",
                )

            if from_table == join_table:
                logger.error("Privacy unit self-join disallowed: table=%s", from_table)
                raise PrivacyConstraintError(
                    "Privacy unit self-join disallowed",
                    context={"table": from_table, "constraint": columns},
                    hint="Use distinct tables or aliases",
                )

            if (
                from_column == privacy_unit_columns[self._from_db_name][from_table]
                and join_column == privacy_unit_columns[self._join_db_name][join_table]
            ):
                logger.debug(
                    "Privacy unit equi-join found: %s.%s = %s.%s",
                    from_table,
                    from_column,
                    join_table,
                    join_column,
                )
                if privacy_check:
                    logger.error("Duplicate privacy unit equi-join: %s", columns)
                    raise PrivacyConstraintError(
                        "Duplicate privacy unit equi-join",
                        context={"constraint": columns},
                        hint="Specify only one equi-join on privacy unit columns",
                    )
                privacy_check = True

        logger.debug("privacy_check result=%s", privacy_check)
        return privacy_check
