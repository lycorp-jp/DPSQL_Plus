import logging
from typing import Any

from lark.lexer import Token
from lark.tree import Tree

from ..errors import QueryParseError

logger = logging.getLogger(__name__)


def _parse_token_literal(token: Token) -> str:
    """Parse a token literal value."""
    logger.debug(
        "_parse_token_literal: token=%s type=%s",
        str(token),
        getattr(token, "type", None),
    )
    match token:
        case "NULL":
            return "NULL"
        case "CURRENT_TIMESTAMP":
            return "CURRENT_TIMESTAMP"
        case "CURRENT_TIME":
            return "CURRENT_TIME"
        case "CURRENT_DATE":
            return "CURRENT_DATE"
        case "TRUE":
            return "TRUE"
        case "FALSE":
            return "FALSE"
        case _:
            logger.error("Invalid literal token: %s", str(token))
            raise QueryParseError(
                "Invalid literal token",
                context={"token": str(token)},
                hint="Allowed: NULL, CURRENT_TIMESTAMP, CURRENT_TIME, CURRENT_DATE, "
                "TRUE, FALSE",
            )


def _parse_tree_literal(tree: Tree[Any]) -> str:
    """Parse a tree literal value."""
    logger.debug(
        "_parse_tree_literal: data=%s children=%s", tree.data, len(tree.children)
    )
    match tree.data:
        case "string_literal" | "numeric_literal":
            return resolve_children_value(tree.children)
        case "blob_literal":
            if len(tree.children) == 0:
                logger.error("Empty blob literal")
                raise QueryParseError(
                    "Empty blob literal",
                    context={"tree": str(tree)},
                    hint="Provide at least one blob byte",
                )
            value = f"0x{''.join(str(c) for c in tree.children)}"
            logger.debug("_parse_tree_literal blob -> %s", value)
            return value
        case _:
            logger.error("Invalid literal tree type: %s", tree.data)
            raise QueryParseError(
                "Invalid literal tree type",
                context={"tree_type": tree.data},
                hint="Allowed: string_literal, numeric_literal, blob_literal",
            )


def resolve_children_value(
    children: list[Token | Tree[Any]], istype: bool = False
) -> str:
    logger.debug("resolve_children_value: count=%s istype=%s", len(children), istype)
    if len(children) != 1:
        logger.error("Invalid child count (expected 1): %s", len(children))
        raise QueryParseError(
            "Invalid child count (expected 1)",
            context={
                "children_count": len(children),
                "children": [str(c) for c in children],
            },
            hint="Simplify the parse node",
        )

    if istype:
        if not isinstance(children[0], Token):
            logger.error(
                "Expected token node for type extraction: %s",
                type(children[0]).__name__,
            )
            raise QueryParseError(
                "Expected token node for type extraction",
                context={"child": str(children[0]), "type": type(children[0]).__name__},
                hint="Pass a token node when istype=True",
            )
        result = str(children[0].type)
        logger.debug("resolve_children_value (type) -> %s", result)
        return result
    else:
        result = f"{children[0]}"
        logger.debug("resolve_children_value -> %s", result)
        return result


def resolve_instance_value(tree: Tree[Any], rule: str, istype: bool = False) -> str:
    logger.debug(
        "resolve_instance_value: rule=%s tree.data=%s istype=%s",
        rule,
        tree.data,
        istype,
    )
    allowed_rules = {
        "table_name",
        "table_name_alias",
        "column_name",
        "db_name",
        "column_name_alias",
        "binary_operator",
        "literal_value",
        "final_aggregate_function_name",
        "collation_name",
        "privacy_param",
    }
    if rule not in allowed_rules:
        logger.error("Invalid parse rule: %s", rule)
        raise QueryParseError(
            "Invalid parse rule",
            context={"rule": rule, "allowed_rules": allowed_rules},
            hint="Use one of the allowed rule names",
        )

    if tree.data != rule:
        logger.error("Parse rule mismatch: expected=%s actual=%s", rule, tree.data)
        raise QueryParseError(
            "Parse rule mismatch",
            context={"expected": rule, "actual": tree.data},
            hint="Ensure the parse tree node matches the requested rule",
        )

    if tree.data == "literal_value":
        result = parse_literal_value(tree)
        logger.debug("resolve_instance_value (literal) -> %s", result)
        return result

    result = resolve_children_value(tree.children, istype)
    logger.debug("resolve_instance_value -> %s", result)
    return result


def resolve_instance_3_values(
    tree: Tree[Any], rule: str
) -> tuple[str | None, str | None, str | None]:
    logger.debug("resolve_instance_3_values: rule=%s tree=%s", rule, tree)
    if rule not in ["table_expr"]:
        logger.error("Invalid rule for 3-value resolution: %s", rule)
        raise QueryParseError(
            "Invalid rule",
            context={"rule": rule, "expected": "table_expr"},
            hint="Use table_expr for 3-value resolution",
        )

    match [tree.data, rule]:
        case [rule, "table_expr"]:
            children = tree.children
            match len(children):
                case 1:
                    return (
                        None,
                        resolve_instance_value(children[0], "table_name"),
                        None,
                    )
                case 2:
                    return (
                        resolve_instance_value(children[0], "db_name"),
                        resolve_instance_value(children[1], "table_name"),
                        None,
                    )
                case 3:
                    return (
                        None,
                        resolve_instance_value(children[0], "table_name"),
                        resolve_instance_value(children[2], "table_name_alias"),
                    )
                case 4:
                    return (
                        resolve_instance_value(children[0], "db_name"),
                        resolve_instance_value(children[1], "table_name"),
                        resolve_instance_value(children[3], "table_name_alias"),
                    )
                case _:
                    logger.error("Invalid child count in table_expr: %s", len(children))
                    raise QueryParseError(
                        "Invalid child count in `table_expr`",
                        context={
                            "count": len(children),
                            "children": [str(c) for c in children],
                        },
                        hint="Expected 1, 2, 3, or 4 children",
                    )
        case _:
            logger.error("Parse rule mismatch for table_expr: %s", tree.data)
            raise QueryParseError(
                "Parse rule mismatch",
                context={"expected": "table_expr", "actual": tree.data},
                hint="Ensure node is a table_expr",
            )


def token_concat(
    items: list[Token | Tree[Any]], sep: str = "_", is_type: bool = False
) -> str:
    logger.debug("token_concat: count=%s sep='%s' is_type=%s", len(items), sep, is_type)
    items_iterator = iter(items)
    raw_result: list[str] = []
    try:
        while True:
            item = next(items_iterator)
            if not isinstance(item, Token):
                logger.error(
                    "Expected token in token list: %s (%s)",
                    str(item),
                    type(item).__name__,
                )
                raise QueryParseError(
                    "Expected token in token list",
                    context={"item": str(item), "type": type(item).__name__},
                    hint="Ensure only tokens are passed",
                )
            raw_result.append(str(item.type) if is_type else str(item))
    except StopIteration:
        pass
    result = sep.join(raw_result)
    logger.debug("token_concat -> %s", result)
    return result


def parse_literal_value(item: Tree[Any]) -> str:
    """Parse the literal value from the tree."""
    logger.debug("parse_literal_value: data=%s", item.data)
    if item.data != "literal_value":
        logger.error("Expected `literal_value` node: got %s", item.data)
        raise QueryParseError(
            "Expected `literal_value` node",
            context={"node": item.data},
            hint="Pass a literal_value parse tree",
        )

    if len(item.children) != 1:
        logger.error("Invalid child count in `literal_value`: %s", len(item.children))
        raise QueryParseError(
            "Invalid child count in `literal_value`",
            context={"count": len(item.children)},
            hint="Provide exactly one literal child",
        )

    child = item.children[0]

    if isinstance(child, Token):
        result = _parse_token_literal(child)
        logger.debug("parse_literal_value (token) -> %s", result)
        return result
    else:
        result = _parse_tree_literal(child)
        logger.debug("parse_literal_value (tree) -> %s", result)
        return result


def is_private_table_(
    db_name: str | None,
    table_name: str | None,
    privacy_unit_columns: dict[str | None, dict[str, str]],
) -> bool:
    """Check if the table is a private table.
    Args:
        db_name (str | None): The name of the database.
        table_name (str | None): The name of the table.
        privacy_unit_columns (dict[str | None, dict[str, str]]): The privacy unit
          columns of the database.
    Returns:
        bool: True if the table is a private table, False otherwise.
    """
    if db_name not in privacy_unit_columns:
        return False
    return table_name in privacy_unit_columns[db_name]


def get_privacy_definition(privacy_params: dict[str, float | int]) -> str:
    """
    Decide the privacy technique based on the privacy parameters.

    Args:
        privacy_params (dict): The privacy parameters.
            {"EPSILON": float, "DELTA": float,
            "CONTRIBUTION_BOUND": int, "MIN_FREQUENCY": int}
    Returns:
        str: The privacy technique to use.
    """
    logger.debug("get_privacy_definition: keys=%s", privacy_params.keys())
    if privacy_params.keys() == {"EPSILON", "DELTA", "CONTRIBUTION_BOUND"}:
        return "DP"
    elif privacy_params.keys() == {
        "EPSILON",
        "DELTA",
        "CONTRIBUTION_BOUND",
        "MIN_FREQUENCY",
    }:
        return "DP_MIN_FREQUENCY"
    elif privacy_params.keys() == {"MIN_FREQUENCY"}:
        return "MIN_FREQUENCY"
    else:
        return "UNKNOWN"
