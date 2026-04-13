import logging
from collections.abc import Iterator
from typing import Any

from lark.lexer import Token
from lark.tree import Tree

from ..errors import QueryParseError, ValidationError
from .utils import resolve_instance_value

logger = logging.getLogger(__name__)


class AlterPrivateTableParser:
    """Parser for ALTER PRIVATE_TABLE statement in PrivSQL."""

    def __init__(self, items: list[Token | Tree[Any]]):
        logger.debug("Init AlterPrivateTableParser (items=%s)", len(items))
        items_iterator = iter(items)
        self._table_name: str | None = None
        self._column_name: str | None = None
        self._db_name: str | None = None
        isExhausted = False
        try:
            item = next(items_iterator)
            while isinstance(item, Token):
                match item.type:
                    case "ALTER":
                        logger.debug("Encounter ALTER token")
                        item, items_iterator = self._alter_parser(items_iterator)
                    case "OPTIONS":
                        logger.debug("Encounter OPTIONS token")
                        item, items_iterator = self._options_parser(items_iterator)
                    case _:
                        logger.error(
                            "Unexpected token in ALTER PRIVATE_TABLE: %s (%s)",
                            item.type,
                            str(item),
                        )
                        raise QueryParseError(
                            "Unexpected token in `ALTER PRIVATE_TABLE` clause",
                            context={"token_type": item.type, "token_value": str(item)},
                            hint="Start with ALTER PRIVATE_TABLE ... then OPTIONS "
                            "PRIVACY_UNIT_COLUMN(table.column)",
                        )
        except StopIteration:
            logger.debug("End of input reached in ALTER PRIVATE_TABLE parser")
            isExhausted = True
        if not isExhausted:
            logger.error("Unexpected trailing input in ALTER PRIVATE_TABLE clause")
            raise QueryParseError(
                "Unexpected trailing input in `ALTER PRIVATE_TABLE` clause",
                hint="Terminate the clause after the OPTIONS section",
            )

    def _alter_parser(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("_alter_parser start")
        item = next(items_iterator)
        if not isinstance(item, Token) or item.type != "PRIVATE_TABLE":
            logger.error(
                "Missing PRIVATE_TABLE after ALTER: got %s (%s)",
                getattr(item, "type", None),
                str(item),
            )
            raise QueryParseError(
                "Missing `PRIVATE_TABLE` after `ALTER`",
                context={
                    "token_type": getattr(item, "type", None),
                    "token_value": str(item),
                },
                hint="Use: ALTER PRIVATE_TABLE <db_name?>.<table_name> OPTIONS ...",
            )
        item = next(items_iterator)
        while isinstance(item, Tree):
            match item.data:
                case "db_name":
                    self._db_name = resolve_instance_value(item, "db_name")
                    logger.debug("Resolved db_name=%s", self._db_name)
                case "table_name":
                    self._table_name = resolve_instance_value(item, "table_name")
                    logger.debug("Resolved table_name=%s", self._table_name)
                case _:
                    logger.error("Unexpected element in identifier list: %s", item.data)
                    raise QueryParseError(
                        "Unexpected element in private table identifier list",
                        context={"element": item.data},
                        hint="Provide optional db_name then table_name",
                    )
            item = next(items_iterator)
        if item.type != "OPTIONS":
            logger.error(
                "Missing OPTIONS after table definition: got %s (%s)",
                getattr(item, "type", None),
                str(item),
            )
            raise QueryParseError(
                "Missing `OPTIONS` after table definition",
                context={
                    "token_type": getattr(item, "type", None),
                    "token_value": str(item),
                },
                hint="Add: OPTIONS PRIVACY_UNIT_COLUMN(table.column)",
            )
        logger.debug("_alter_parser done -> next token: %s", item.type)
        return item, items_iterator

    def _options_parser(
        self, items_iterator: Iterator[Token | Tree[Any]]
    ) -> tuple[Token | Tree[Any], Iterator[Token | Tree[Any]]]:
        logger.debug("_options_parser start")
        item = next(items_iterator)
        if not isinstance(item, Token) or item.type != "PRIVACY_UNIT_COLUMN":
            logger.error(
                "Missing PRIVACY_UNIT_COLUMN in OPTIONS: got %s (%s)",
                getattr(item, "type", None),
                str(item),
            )
            raise QueryParseError(
                "Missing `PRIVACY_UNIT_COLUMN` in `OPTIONS` clause",
                context={
                    "token_type": getattr(item, "type", None),
                    "token_value": str(item),
                },
                hint="Use: OPTIONS PRIVACY_UNIT_COLUMN(table.column)",
            )
        item = next(items_iterator)
        while isinstance(item, Tree):
            match item.data:
                case "table_name":
                    table_name = resolve_instance_value(item, "table_name")
                    logger.debug(
                        "Resolved PRIVACY_UNIT_COLUMN.table_name=%s", table_name
                    )
                    if self._table_name != table_name:
                        logger.error(
                            "Mismatch between ALTER table and"
                            " PRIVACY_UNIT_COLUMN: %s != %s",
                            self._table_name,
                            table_name,
                        )
                        raise ValidationError(
                            "Mismatch between table name in `ALTER` and "
                            "`PRIVACY_UNIT_COLUMN`",
                            context={
                                "alter_table": self._table_name,
                                "privacy_unit_table": table_name,
                            },
                            hint="Use identical table name in both locations",
                        )
                case "column_name":
                    self._column_name = resolve_instance_value(item, "column_name")
                    logger.debug(
                        "Resolved PRIVACY_UNIT_COLUMN.column_name=%s", self._column_name
                    )
                case _:
                    break
            item = next(items_iterator)
        logger.debug(
            "_options_parser done -> next token: %s", getattr(item, "type", None)
        )
        return item, items_iterator

    @property
    def table_name(self) -> str:
        logger.debug("Access table_name")
        if self._table_name is None:
            logger.error("Missing table_name in ALTER PRIVATE_TABLE")
            raise QueryParseError(
                "Missing table name in `ALTER PRIVATE_TABLE` statement",
                hint="Specify table_name after PRIVATE_TABLE",
            )
        return self._table_name

    @property
    def column_name(self) -> str:
        logger.debug("Access column_name")
        if self._column_name is None:
            logger.error("Missing privacy unit column in PRIVACY_UNIT_COLUMN")
            raise QueryParseError(
                "Missing privacy unit column in `PRIVACY_UNIT_COLUMN`",
                hint="Provide column_name inside PRIVACY_UNIT_COLUMN(table.column)",
            )
        return self._column_name

    @property
    def db_name(self) -> str | None:
        logger.debug("Access db_name -> %s", self._db_name)
        return self._db_name
