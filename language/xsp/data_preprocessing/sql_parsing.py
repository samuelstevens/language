# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""For converting from a SQL query to output/decoder actions."""
from __future__ import absolute_import, division, print_function

import re
from typing import Dict, List, Optional, Sequence, Union, overload

import sqlparse
from prelude import flattened
from typing_extensions import Literal

from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample
from language.xsp.data_preprocessing.schema_utils import DatabaseTable, TableColumn
from language.xsp.data_preprocessing.sql_utils import SchemaEntityCopy, SQLAction


def _is_valid_generated_token(tok: str) -> bool:
    # numbers are ok to generate
    if re.match(r"\d+", tok):
        return True

    return False


class ParseError(Exception):
    pass


def _is_select(item):
    return item.ttype is sqlparse.tokens.DML and item.value.lower() == "select"


def _is_function(item):
    return isinstance(item, sqlparse.sql.Function)


def _is_from(item):
    return item.ttype is sqlparse.tokens.Keyword and item.value.lower() == "from"


def _is_identifier(item):
    return isinstance(item, sqlparse.sql.Identifier)


def _is_identifier_list(item):
    return isinstance(item, sqlparse.sql.IdentifierList)


def _is_parenthesis(item):
    return isinstance(item, sqlparse.sql.Parenthesis)


def _is_operation(item):
    return isinstance(item, sqlparse.sql.Operation)


def _is_operator(item):
    return item.ttype is sqlparse.tokens.Operator


def _is_punctuation(item):
    return item.ttype is sqlparse.tokens.Punctuation


def _is_wildcard(item):
    return item.ttype is sqlparse.tokens.Wildcard


def _is_keyword(item):
    return item.ttype is sqlparse.tokens.Keyword


def _is_name(item):
    return item.ttype is sqlparse.tokens.Name


def _is_order(item):
    return item.ttype is sqlparse.tokens.Keyword.Order


def _is_where(item):
    return isinstance(item, sqlparse.sql.Where)


def _is_comparison(item):
    return isinstance(item, sqlparse.sql.Comparison)


def _is_comparison_operator(item):
    return item.ttype is sqlparse.tokens.Operator.Comparison


def _is_literal(item):
    return item.ttype in (
        sqlparse.tokens.Literal.String.Single,
        sqlparse.tokens.Literal.Number.Integer,
        sqlparse.tokens.Literal.String.Symbol,
        sqlparse.tokens.Literal.Number.Float,
    )


def _is_integer(item):
    return item.ttype is sqlparse.tokens.Literal.Number.Integer


def _is_table_alias(item):
    return _is_name(item) and re.match(r"t\d+", item.value.lower())


def _is_whitespace(item):
    return item.ttype is sqlparse.tokens.Whitespace


def _add_simple_step(item, example: NLToSQLExample) -> None:
    example.gold_sql_query.actions.append(SQLAction(symbol=item.value.lower()))


def _debug_sql_item(item):
    print("--- debugging... ---")
    print("Item is " + repr(item))
    print("Type is " + str(item.ttype))
    print("Value is " + str(item.value))
    print("String value is " + str(item))
    # breakpoint()


def _debug_state(item, example):
    # breakpoint()
    del example
    _debug_sql_item(item)
    if hasattr(item, "_pprint_tree"):
        item._pprint_tree()  # pylint: disable=protected-access


def _is_single_arg_literal(sql) -> bool:
    """Checks if it's ( 1 ) or ( * ), etc."""
    assert _is_parenthesis(sql)

    open_paren, *raw_args, close_paren = list(sql)

    assert _is_punctuation(open_paren)
    assert _is_punctuation(close_paren)

    raw_args = [arg for arg in raw_args if not _is_whitespace(arg)]

    if len(raw_args) != 1:
        return False

    arg = raw_args[0]

    return _is_wildcard(arg) or _is_literal(arg)


def _parse_single_arg(sql, example, anonymize_values: bool) -> bool:
    assert _is_parenthesis(sql)

    open_paren, *raw_args, close_paren = list(sql)

    assert _is_punctuation(open_paren)
    assert _is_punctuation(close_paren)

    raw_args = [arg for arg in raw_args if not _is_whitespace(arg)]

    literal = raw_args[0]

    _add_simple_step(open_paren, example)
    success = _add_simple_value(
        literal, example, anonymize_values
    ) or _is_valid_generated_token(literal.value)
    _add_simple_step(close_paren, example)

    return success


def _parse_function(sql, example, anonymize_values: bool) -> bool:
    """Parse the part relative to a Function in the SQL query."""
    successful_copy = True
    for item in sql:
        if _is_parenthesis(item):
            # (samuelstevens) Check if it's just a single literal inside the function (typically for a COUNT( 1 ) sort of thing).
            if _is_single_arg_literal(item):
                successful_copy = (
                    _parse_single_arg(item, example, anonymize_values)
                    and successful_copy
                )
            else:
                successful_copy = (
                    populate_sql(item, example, anonymize_values) and successful_copy
                )
            continue

        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue

        if _is_identifier(item) and item.value.lower() in (
            "count",
            "avg",
            "min",
            "max",
            "sum",
            "all",
            "distinct",
        ):
            _add_simple_step(item, example)
            continue

        _debug_state(item, example)
        raise ParseError("Incomplete _parse_function")

    return successful_copy


@overload
def _find_all_entities(
    token,
    example: NLToSQLExample,
    include_tables: Literal[False],
    include_columns: Literal[True],
    restrict_to_table: DatabaseTable,
) -> Sequence[TableColumn]:
    ...


@overload
def _find_all_entities(
    token,
    example: NLToSQLExample,
    include_tables: Literal[True],
    include_columns: Literal[False],
    restrict_to_table: None,
) -> Sequence[DatabaseTable]:
    ...


@overload
def _find_all_entities(
    token,
    example: NLToSQLExample,
    include_tables: bool,
    include_columns: bool,
    restrict_to_table: None,
) -> Sequence[Union[DatabaseTable, TableColumn]]:
    ...


def _find_all_entities(
    token: str,
    example: NLToSQLExample,
    include_tables: bool,
    include_columns: bool,
    restrict_to_table: Optional[DatabaseTable],
) -> Sequence[Union[DatabaseTable, TableColumn]]:
    """
    Tries to find schema entities that match the token.
    """
    if restrict_to_table:
        columns: List[TableColumn] = []
        assert include_columns
        assert not include_tables
        for column in restrict_to_table.table_columns:
            if column.original_column_name.lower() == token.lower():
                columns.append(column)
        return columns
    else:
        matching: List[Union[DatabaseTable, TableColumn]] = []
        for table in example.model_input.tables:
            # Allow lowercase matching because the schema and gold query may not match
            if table.original_table_name.lower() == token.lower() and include_tables:
                matching.append(table)

            for column in table.table_columns:
                if (
                    column.original_column_name.lower() == token.lower()
                    and include_columns
                ):
                    matching.append(column)

        return matching


@overload
def _find_simple_entity(
    token,
    example: NLToSQLExample,
    include_tables: Literal[False],
    include_columns: Literal[True],
) -> Optional[TableColumn]:
    ...


@overload
def _find_simple_entity(
    token,
    example: NLToSQLExample,
    include_tables: Literal[True],
    include_columns: Literal[False],
) -> Optional[DatabaseTable]:
    ...


@overload
def _find_simple_entity(
    token, example: NLToSQLExample, include_tables: bool, include_columns: bool,
) -> Union[None, DatabaseTable, TableColumn]:
    ...


def _find_simple_entity(
    token, example: NLToSQLExample, include_tables: bool, include_columns: bool
) -> Union[None, DatabaseTable, TableColumn]:
    """Finds entities in the schema that a token may be referring to."""
    matching = _find_all_entities(token, example, include_tables, include_columns, None)

    if not matching:
        return None

    if len(matching) == 1:
        return matching[0]

    return None


def _get_tokens(item):
    if hasattr(item, "tokens"):
        return [
            t for t in item.tokens if t.ttype is not sqlparse.tokens.Text.Whitespace
        ]
    else:
        return []


def _find_table_annotations(item, example: NLToSQLExample) -> List[DatabaseTable]:
    """Finds a reference to a table in the schema."""
    if _is_name(item):
        entity = _find_simple_entity(
            item.value, example, include_tables=True, include_columns=False
        )
        assert entity is not None
        return [entity]

    tokens = _get_tokens(item)

    if len(tokens) == 1:
        return _find_table_annotations(tokens[0], example)

    # Assume it's AS
    if _is_identifier(item) and len(tokens) == 3:
        return _find_table_annotations(tokens[0], example)

    # (samuelstevens) It could be an identifier list? Each identifier in the list could also be "table AS alias".
    if _is_identifier_list(item):
        return flattened(
            [
                _find_table_annotations(ident, example)
                for ident in item.get_identifiers()
            ]
        )

    _debug_sql_item(item)
    raise ParseError("Cannot find table annotation")


def _find_column_entities(
    token: str, example: NLToSQLExample, table: DatabaseTable
) -> Optional[TableColumn]:
    all_entities: Sequence[TableColumn] = _find_all_entities(
        token, example, False, True, table
    )

    # Make sure that there was only one found
    assert len(all_entities) <= 1

    if all_entities:
        return all_entities[0]
    return None


def _iterate_sql(sql_parse):
    for item in sql_parse:
        if hasattr(item, "tokens"):
            for child in _iterate_sql(item):
                yield child
        else:
            yield item


def _get_all_aliases(sql, example: NLToSQLExample) -> Dict[str, DatabaseTable]:
    """Returns a dictionary of aliases for the tables."""
    root = sql
    while True:
        if root.parent is None:
            break
        root = root.parent

    aliases = {}
    for item in _iterate_sql(root):
        if not _is_table_alias(item):
            continue

        tokens = _get_tokens(item.parent.parent)
        if len(tokens) == 3 and tokens[1].value.lower() == "as":
            table_annotation = _find_table_annotations(tokens[0], example)[0]
            aliases[tokens[2].value.lower()] = table_annotation

    return aliases


def _resolve_reference(item: sqlparse.sql.Token, example: NLToSQLExample) -> None:
    """
    Resolves an ambiguous token that matches multiple annotations.

    Args:
        item: position in the SQL parse where the search will start.
        example: the QuikExample containing table and column annotations.

    Raises:
        ParseError: if the ambiguity cannot be resolved.
    """
    prev_symbol: str = example.gold_sql_query.actions[
        len(example.gold_sql_query.actions) - 1
    ].symbol

    if prev_symbol in ("join", "from"):
        table_annotations = _find_table_annotations(item, example)
        assert table_annotations, (
            "Cannot find a table annotation for item %s" % item.value
        )
        for table in table_annotations:
            example.gold_sql_query.actions.append(
                SQLAction(entity_copy=SchemaEntityCopy(copied_table=table))
            )
        return

    parent = item.parent

    # We try the simple case, that is aliases
    parent_tokens = _get_tokens(parent)
    if _is_table_alias(parent_tokens[0]) and parent_tokens[1].value == ".":
        aliases = _get_all_aliases(item, example)

        table_annotation = aliases[parent.tokens[0].value.lower()]
        found_column = _find_column_entities(item.value, example, table_annotation)

        assert found_column, "Could not find column with name " + str(item.value)
        example.gold_sql_query.actions.append(
            SQLAction(entity_copy=SchemaEntityCopy(copied_column=found_column))
        )
        return

    def _find_direction(reverse: bool) -> bool:
        """Finds a column annotation in a given direction."""
        table_entities = _find_from_table(item, example, reverse=reverse)

        if not table_entities:
            return False

        for table_entity in table_entities:
            entity = _find_column_entities(item.value, example, table_entity)

            if entity:
                example.gold_sql_query.actions.append(
                    SQLAction(entity_copy=SchemaEntityCopy(copied_column=entity))
                )
                return True
        raise ParseError("Unable to find annotation of table " + str(item))

    if prev_symbol in ("where", "by") or _is_where(parent) or _is_where(parent.parent):
        if _find_direction(reverse=True):
            return

    if _find_direction(reverse=False):
        return

    if _find_direction(reverse=True):
        return

    raise ParseError("Unable to find final annotation in any direction")


def _find_from_table(
    item, example: NLToSQLExample, reverse: bool = False
) -> Optional[List]:
    """
    Finds the table that is being queried.

    Args:
        item: position in the SQL parse where the search will start.
        example: the QuikExample containing the table annotations.
        reverse: if True will use backward search, otherwise forward.

    Returns:
        a list of QuikAnnotation that are tables or None.
    """
    parent = item.parent
    next_token_index = parent.token_index(item)
    next_item = item

    # Here we iterate over al the possible tokens in the current level until we
    # reach a FROM statement.
    def _next_token():
        if reverse:
            return parent.token_prev(next_token_index)
        else:
            return parent.token_next(next_token_index)

    down_exploration = False

    while True:
        prev_item = next_item
        next_token_index, next_item = _next_token()

        if next_item is None:
            # Go up
            if not down_exploration and (
                _is_identifier_list(parent)
                or _is_identifier(parent)
                or _is_comparison(parent)
                or _is_where(parent)
                or _is_operation(parent)
                or _is_parenthesis(parent)
                or _is_function(parent)
                or _is_select(parent)
            ):
                next_token_index = parent.parent.token_index(parent)
                next_item = parent
                parent = parent.parent

            # Go down
            elif hasattr(prev_item, "tokens") and len(prev_item.tokens) > 1:
                next_token_index = prev_item.token_index(prev_item.tokens[0])
                next_item = prev_item.tokens[0]
                parent = prev_item
                down_exploration = True
            else:
                return None

        if _is_from(next_item):
            next_token_index, next_item = parent.token_next(next_token_index)

            tables: List[DatabaseTable] = []
            annotation = _find_table_annotations(next_item, example)
            if not annotation:
                raise ValueError("No annotation found!")
            tables.extend(_find_table_annotations(next_item, example))

            next_token_index, next_item = parent.token_next(next_token_index)

            if next_item and next_item.value.lower() == "join":
                next_token_index, next_item = parent.token_next(next_token_index)
                tables.extend(_find_table_annotations(next_item, example))

            return tables


def _add_simple_value(item, example: NLToSQLExample, anonymize: bool) -> bool:
    """
    Adds a value action to the output.

    Args:
        item: A string value present in the SQL query.
        example: The NLToSQLExample being constructed.
        anonymize: Whether to anonymize values (i.e., replace them with a 'value' placeholder).  Returns a boolean indicating whether the value could be copied from the input.

    Returns:
        Successful copy.
    """
    if anonymize:
        example.gold_sql_query.actions.append(SQLAction(symbol="value"))
        return True

    # Commenting out the code that anonymizes.
    item_str = str(item)

    # Add quotes if [1] there aren't quotes and [2] it's not numeric
    if (
        not item_str.replace(".", "", 1).isdigit()
        and item_str.count('"') < 2
        and item_str.count("'") < 2
    ):
        item_str = "'" + item_str + "'"

    def find_and_add_copy_from_text(substr: str):
        """Finds a substring in the utterance and adds a copying action."""
        # It's a substring of the original utterance, but not so sure it could be
        # composed of wordpieces.
        found = False
        start_wordpiece = -1
        end_wordpiece = -1
        for i in range(len(example.model_input.utterance_wordpieces)):
            for j in range(i + 1, len(example.model_input.utterance_wordpieces) + 1):
                # Compose a string. If it has ##, that means it's a wordpiece, so should
                # not have a space in front.
                composed_pieces = (
                    " ".join(
                        [
                            piece.wordpiece
                            for piece in example.model_input.utterance_wordpieces[i:j]
                        ]
                    )
                    .replace(" ##", "")
                    .replace(" .", ".")
                    .replace(" - ", "-")
                    .replace(" !", "!")
                )
                if substr.lower() == composed_pieces:
                    start_wordpiece = i
                    end_wordpiece = j
                    found = True
                    break
            if found:
                break

        if start_wordpiece >= 0 and end_wordpiece >= 0:
            # Found wordpiece(s, when put together) comprising this item
            for i in range(start_wordpiece, end_wordpiece):
                action = SQLAction(
                    utterance_copy=example.model_input.utterance_wordpieces[i]
                )
                example.gold_sql_query.actions.append(action)
            return True
        return False

    # First, check if this string could be copied from the wordpiece-tokenized inputs.
    quote_type = ""
    if item_str.lower() in example.model_input.original_utterance.lower():
        success = find_and_add_copy_from_text(item_str)

        # QUESTION(samuelstevens): why do they use "not" success? Shouldn't they add it only if it was successfully found?
        # ANSWER(samuelstevens): Because find_and_add_copy_from_text already modifies example (appends an action)
        if not success or _is_valid_generated_token(item_str):
            example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

        return success or _is_valid_generated_token(item_str)

    elif item_str.startswith("'") and item_str.endswith("'"):
        quote_type = "'"
    elif item_str.startswith('"') and item_str.endswith('"'):
        quote_type = '"'

    if quote_type:
        if item_str[1:-1].lower() in example.model_input.original_utterance.lower():
            example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

            success = find_and_add_copy_from_text(item_str[1:-1])
            if not success or _is_valid_generated_token(item_str):
                example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

            example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

            return success or _is_valid_generated_token(item_str)
        elif (
            item_str[1] == "%"
            and item_str[-2] == "%"
            and item_str[2:-2].lower() in example.model_input.original_utterance.lower()
        ):
            example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))
            example.gold_sql_query.actions.append(SQLAction(symbol="%"))

            success = find_and_add_copy_from_text(item_str[2:-2])
            if not success or _is_valid_generated_token(item_str):
                example.gold_sql_query.actions.append(SQLAction(symbol=item_str[2:-2]))

            example.gold_sql_query.actions.append(SQLAction(symbol="%"))
            example.gold_sql_query.actions.append(SQLAction(symbol=quote_type))

            return success or _is_valid_generated_token(item_str)

    # Just add it as choice from the output vocabulary
    if "u s a" in item_str:
        raise ValueError("WHAT????????")

    example.gold_sql_query.actions.append(SQLAction(symbol=item_str))

    # A value of 1 is used for things like LIMIT 1 when ordering.
    return _is_valid_generated_token(item_str)


def _parse_identifier(sql, example: NLToSQLExample, anonymize_values: bool) -> bool:
    """Parse the part relative to an Identifier in the SQL query."""
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue

        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_order(item):
            _add_simple_step(item, example)
            continue

        if _is_table_alias(item) or (_is_punctuation(item) and item.value == "."):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value in ("as",):
            _add_simple_step(item, example)
            continue

        if _is_name(item):
            entity = _find_simple_entity(item.value, example, True, True)
            if entity is not None:
                schema_copy_action = None

                if isinstance(entity, DatabaseTable):
                    schema_copy_action = SchemaEntityCopy(copied_table=entity)
                elif isinstance(entity, TableColumn):
                    schema_copy_action = SchemaEntityCopy(copied_column=entity)
                else:
                    raise ValueError(
                        "Type of entity is unexpected: " + str(type(entity))
                    )

                copy_action = SQLAction(entity_copy=schema_copy_action)
                example.gold_sql_query.actions.append(copy_action)
            else:
                try:
                    _resolve_reference(item, example)
                except AttributeError as e:
                    # Generally this means the reference wasn't found i.e., in WikiSQL, a
                    # value didn't have quotes, so just add it as a value
                    print(e)
                    successful_copy = (
                        _add_simple_value(item, example, anonymize_values)
                        and successful_copy
                    )
            continue

        if _is_function(item):
            successful_copy = (
                _parse_function(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_literal(item):
            prev_len = len(example.gold_sql_query.actions)
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )
            if len(example.gold_sql_query.actions) == prev_len:
                raise ValueError(
                    "Gold query did not change length when adding simple value!"
                )
            continue

        _debug_state(item, example)
        raise ParseError("Incomplete _parse_identifier")

    return successful_copy


def _parse_operation(sql, example: NLToSQLExample, anonymize_values: bool):
    """Parse the part relative to an Operation in the SQL query."""
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue
        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue
        if _is_operator(item):
            _add_simple_step(item, example)
            continue
        if _is_function(item):
            successful_copy = (
                _parse_function(item, example, anonymize_values) and successful_copy
            )
            continue

        _debug_sql_item(item)
        raise ParseError("Incomplete _parse_operation")

    return successful_copy


def _parse_identifier_list(sql, example, anonymize_values):
    """Parse the part relative to an IdentifierList in the SQL query."""
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue

        if _is_punctuation(item) and (item.value in (",",)):
            _add_simple_step(item, example)
            continue

        if _is_function(item):
            _parse_function(item, example, anonymize_values)
            continue

        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_operation(item):
            successful_copy = (
                _parse_operation(item, example, anonymize_values) and successful_copy
            )
            continue
        if _is_keyword(item) and item.value.lower() in (
            "count",
            "avg",
            "min",
            "max",
            "sum",
        ):
            _add_simple_step(item, example)
            continue

        if _is_literal(item):
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )
            continue

        _debug_sql_item(item)
        raise ParseError("Incomplete _parse_identifier_list")

    return successful_copy


def _parse_comparison(sql, example, anonymize_values):
    """Parse the part relative to a comparison in the SQL query."""
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue
        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue
        if _is_comparison_operator(item):
            _add_simple_step(item, example)
            continue
        if _is_literal(item):
            prev_len = len(example.gold_sql_query.actions)
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )

            if len(example.gold_sql_query.actions) == prev_len:
                raise ValueError(
                    "Gold query did not change length when adding simple value!"
                )
            continue

        if _is_parenthesis(item):
            successful_copy = (
                populate_sql(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_function(item):
            successful_copy = (
                _parse_function(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_operation(item):
            successful_copy = (
                _parse_operation(item, example, anonymize_values) and successful_copy
            )
            continue

        _debug_state(item, example)
        raise ParseError("Incomplete _parse_comparison")

    return successful_copy


def _parse_where(sql, example: NLToSQLExample, anonymize_values: bool) -> bool:
    """Parse the part relative to the WHERE clause of the SQL query."""
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue

        if _is_keyword(item) and item.value.lower() in ("where",):
            _add_simple_step(item, example)
            continue

        if _is_comparison(item):
            successful_copy = (
                _parse_comparison(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_comparison_operator(item):
            _add_simple_step(item, example)
            continue

        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_identifier_list(item):
            successful_copy = (
                _parse_identifier_list(item, example, anonymize_values)
                and successful_copy
            )
            continue

        if _is_keyword(item) and item.value.lower() in ("between", "and", "or"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("is", "not null"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("not", "in", "intersect"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("distinct",):
            _add_simple_step(item, example)
            continue

        if _is_integer(item):
            prev_len = len(example.gold_sql_query.actions)
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )
            if len(example.gold_sql_query.actions) == prev_len:
                raise ValueError(
                    "Gold query did not change length when adding simple value!"
                )
            continue

        # Like '%asd%'
        if _is_keyword(item) and item.value.lower() in ("like",):
            _add_simple_step(item, example)
            continue

        if _is_literal(item):
            prev_len = len(example.gold_sql_query.actions)
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )
            if len(example.gold_sql_query.actions) == prev_len:
                raise ValueError(
                    "Gold query did not change length when adding simple value!"
                )
            continue

        if _is_keyword(item) and item.value.lower() in ("join", "on"):
            _add_simple_step(item, example)
            continue

        if _is_parenthesis(item):
            successful_copy = (
                populate_sql(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_select(item) or _is_from(item):
            _add_simple_step(item, example)
            continue

        _debug_sql_item(item)
        raise ParseError("Incomplete _parse_where")

    return successful_copy


def populate_sql(
    sql: sqlparse.sql.Statement, example: NLToSQLExample, anonymize_values: bool
) -> bool:
    """
    Creates a sequence of output / decoder actions from a raw SQL query.

    Args:
        sql: The SQL query to convert.
        example: The NLToSQLExample object to add output actions.
        anonymize_values: Whether to anonymize values by replacing with a placeholder.

    Raises:
        ParseError: if the SQL query can't be parsed.

    Returns:
        Boolean indicating whether all actions copying values from the input utterance were successfully completed.
  """
    successful_copy = True
    for item in sql:
        if item.ttype == sqlparse.tokens.Text.Whitespace:
            continue

        if _is_punctuation(item) and (item.value in ("(", ")")):
            _add_simple_step(item, example)
            continue

        if _is_punctuation(item) and (item.value in (",",)):
            _add_simple_step(item, example)
            continue

        if _is_parenthesis(item):
            successful_copy = (
                populate_sql(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_wildcard(item):
            _add_simple_step(item, example)
            continue

        if _is_select(item) or _is_from(item):
            _add_simple_step(item, example)
            continue

        if _is_where(item):
            successful_copy = (
                _parse_where(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_function(item):
            successful_copy = (
                _parse_function(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_identifier(item):
            successful_copy = (
                _parse_identifier(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_identifier_list(item):
            successful_copy = (
                _parse_identifier_list(item, example, anonymize_values)
                and successful_copy
            )
            continue

        if _is_keyword(item) and item.value.lower() in (
            "group",
            "order",
            "by",
            "having",
            "order by",
            "group by",
        ):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in (
            "count",
            "avg",
            "min",
            "max",
            "sum",
        ):
            _add_simple_step(item, example)
            continue

        if _is_operation(item):
            successful_copy = (
                _parse_operation(item, example, anonymize_values) and successful_copy
            )
            continue

        if _is_keyword(item) and item.value.lower() in ("between", "and", "or"):
            _add_simple_step(item, example)
            continue

        if _is_order(item):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("is", "not null", "in", "not"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("distinct",):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("limit",):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("join", "on"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("intersect", "union"):
            _add_simple_step(item, example)
            continue

        if _is_keyword(item) and item.value.lower() in ("except",):
            _add_simple_step(item, example)
            continue

        if _is_integer(item) and example.gold_sql_query.actions[
            len(example.gold_sql_query.actions) - 1
        ].symbol in ("limit", "between", "and"):
            prev_len = len(example.gold_sql_query.actions)
            successful_copy = (
                _add_simple_value(item, example, anonymize_values) and successful_copy
            )
            if len(example.gold_sql_query.actions) == prev_len:
                raise ValueError(
                    "Gold query did not change length when adding simple value!"
                )
            continue

        if _is_comparison(item):
            successful_copy = (
                _parse_comparison(item, example, anonymize_values) and successful_copy
            )
            continue

        _debug_state(item, example)
        raise ParseError("Incomplete _parse_sql")

    return successful_copy
