from unittest.mock import Mock

import pytest
from lark.lexer import Token
from lark.tree import Tree

from dpsql.errors import QueryParseError
from dpsql.validator.utils import (
    is_private_table_,
    parse_literal_value,
    resolve_children_value,
    resolve_instance_3_values,
    resolve_instance_value,
    token_concat,
)


class TestResolveChildrenValue:
    def test_resolve_children_value_single_token(self):
        token = Token("NAME", "test_value")
        result = resolve_children_value([token])
        assert result == "test_value"

    def test_resolve_children_value_single_token_with_type(self):
        token = Token("NAME", "test_value")
        result = resolve_children_value([token], istype=True)
        assert result == "NAME"

    def test_resolve_children_value_multiple_children_raises_error(self):
        token1 = Token("NAME", "value1")
        token2 = Token("NAME", "value2")
        with pytest.raises(QueryParseError, match="Invalid child count"):
            resolve_children_value([token1, token2])

    def test_resolve_children_value_empty_list_raises_error(self):
        with pytest.raises(QueryParseError, match="Invalid child count"):
            resolve_children_value([])

    def test_resolve_children_value_non_token_with_type_raises_error(self):
        tree = Mock(spec=Tree)
        with pytest.raises(
            QueryParseError, match="Expected token node for type extraction"
        ):
            resolve_children_value([tree], istype=True)


class TestResolveInstanceValue:
    def test_resolve_instance_value_valid_rule(self):
        token = Token("NAME", "test_table")
        tree = Mock(spec=Tree)
        tree.data = "table_name"
        tree.children = [token]

        result = resolve_instance_value(tree, "table_name")
        assert result == "test_table"

    def test_resolve_instance_value_invalid_rule_raises_error(self):
        tree = Mock(spec=Tree)
        tree.data = "invalid_rule"

        with pytest.raises(QueryParseError, match="Invalid parse rule"):
            resolve_instance_value(tree, "invalid_rule")

    def test_resolve_instance_value_mismatched_rule_raises_error(self):
        tree = Mock(spec=Tree)
        tree.data = "column_name"

        with pytest.raises(QueryParseError, match="Parse rule mismatch"):
            resolve_instance_value(tree, "table_name")

    def test_resolve_instance_value_literal_value(self):
        token = Token("NULL", "NULL")
        child_tree = Mock(spec=Tree)
        child_tree.data = "literal_value"
        child_tree.children = [token]

        tree = Mock(spec=Tree)
        tree.data = "literal_value"
        tree.children = [token]

        result = resolve_instance_value(tree, "literal_value")
        assert result == "NULL"


class TestResolveInstance3Values:
    def test_resolve_instance_3_values_invalid_rule_raises_error(self):
        tree = Mock(spec=Tree)
        with pytest.raises(QueryParseError, match="Invalid rule"):
            resolve_instance_3_values(tree, "invalid_rule")

    def test_resolve_instance_3_values_one_child(self):
        table_token = Token("NAME", "users")
        table_tree = Mock(spec=Tree)
        table_tree.data = "table_name"
        table_tree.children = [table_token]

        tree = Mock(spec=Tree)
        tree.data = "table_expr"
        tree.children = [table_tree]

        result = resolve_instance_3_values(tree, "table_expr")
        assert result == (None, "users", None)

    def test_resolve_instance_3_values_two_children(self):
        db_token = Token("NAME", "mydb")
        table_token = Token("NAME", "users")

        db_tree = Mock(spec=Tree)
        db_tree.data = "db_name"
        db_tree.children = [db_token]

        table_tree = Mock(spec=Tree)
        table_tree.data = "table_name"
        table_tree.children = [table_token]

        tree = Mock(spec=Tree)
        tree.data = "table_expr"
        tree.children = [db_tree, table_tree]

        result = resolve_instance_3_values(tree, "table_expr")
        assert result == ("mydb", "users", None)


class TestTokenConcat:
    def test_token_concat_basic(self):
        tokens = [Token("NAME", "hello"), Token("NAME", "world")]
        result = token_concat(tokens)
        assert result == "hello_world"

    def test_token_concat_custom_separator(self):
        tokens = [Token("NAME", "hello"), Token("NAME", "world")]
        result = token_concat(tokens, sep=".")
        assert result == "hello.world"

    def test_token_concat_with_type(self):
        tokens = [Token("NAME", "hello"), Token("NUMBER", "123")]
        result = token_concat(tokens, is_type=True)
        assert result == "NAME_NUMBER"

    def test_token_concat_non_token_raises_error(self):
        tree = Mock(spec=Tree)
        tokens = [Token("NAME", "hello"), tree]
        with pytest.raises(QueryParseError, match="Expected token in token list"):
            token_concat(tokens)

    def test_token_concat_empty_list(self):
        result = token_concat([])
        assert result == ""


class TestParseLiteralValue:
    def test_parse_literal_value_wrong_data_raises_error(self):
        tree = Mock(spec=Tree)
        tree.data = "not_literal_value"
        with pytest.raises(QueryParseError, match=r"Expected `literal_value` node"):
            parse_literal_value(tree)

    def test_parse_literal_value_wrong_children_count_raises_error(self):
        tree = Mock(spec=Tree)
        tree.data = "literal_value"
        tree.children = []
        with pytest.raises(
            QueryParseError, match=r"Invalid child count in `literal_value`"
        ):
            parse_literal_value(tree)

    def test_parse_literal_value_token_child(self):
        token = Token("NULL", "NULL")
        tree = Mock(spec=Tree)
        tree.data = "literal_value"
        tree.children = [token]

        result = parse_literal_value(tree)
        assert result == "NULL"

    def test_parse_literal_value_tree_child(self):
        child_tree = Mock(spec=Tree)
        child_tree.data = "string_literal"
        child_tree.children = [Token("STRING", "'hello'")]

        tree = Mock(spec=Tree)
        tree.data = "literal_value"
        tree.children = [child_tree]

        result = parse_literal_value(tree)
        assert result == "'hello'"


class TestIsPrivateTable:
    def test_is_private_table_db_not_in_privacy_columns(self):
        privacy_unit_columns = {"db1": {"table1": "user_id"}}
        result = is_private_table_("db2", "table1", privacy_unit_columns)
        assert result is False

    def test_is_private_table_table_not_in_privacy_columns(self):
        privacy_unit_columns = {"db1": {"table1": "user_id"}}
        result = is_private_table_("db1", "table2", privacy_unit_columns)
        assert result is False

    def test_is_private_table_table_in_privacy_columns(self):
        privacy_unit_columns = {"db1": {"table1": "user_id"}}
        result = is_private_table_("db1", "table1", privacy_unit_columns)
        assert result is True

    def test_is_private_table_none_db_name(self):
        privacy_unit_columns = {None: {"table1": "user_id"}}
        result = is_private_table_(None, "table1", privacy_unit_columns)
        assert result is True
