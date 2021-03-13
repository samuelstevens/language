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
# Lint as: python3
"""Self-contained library for converting SQL w/ under-specified FROM clause.

This module relies on the sqlparse library, which is a non-validating
SQL parser, and additional heuristics.
TODO(petershaw): A proper parser and grammar for SQL would improve robustness!
"""

from __future__ import absolute_import, division, print_function

import collections
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import sqlparse

from language.xsp.data_preprocessing import sqlparse_keyword_utils

sqlparse_keyword_utils.remove_bad_sqlparse_keywords()

# Queries that are malformed and won't parse.
BAD_SQL = [
    (  # This one has a bad reference to outer alias T2 in nested statement.
        "SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  ="
        "  T2.stuid WHERE T2.dormid IN (SELECT T2.dormid FROM dorm AS T3 JOIN "
        "has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 "
        "ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC "
        "LIMIT 1)"
    ),
    (  # References non-existent table Ref_Company_Types.
        "SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN "
        "Maintenance_Contracts AS T2 ON T1.company_id  =  "
        "T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON "
        "T1.company_type_code  =  T3.company_type_code ORDER BY "
        "T2.contract_end_date DESC LIMIT 1"
    ),
]

# Represents a column.
SqlColumn = NamedTuple("SqlColumn", [("column_name", str), ("table_name", str)])


class SqlSpan:
    """
    This represents either a single token/identifier in the SQL query, or a sequence of tokens representing a nested statement.
    Nested statements can appear in constructions using EXCLUDE, INTERSECTION, and UNION, or can appear in WHERE clauses such as IN (...).
    A list of SqlSpan tuples represents an entire SQL query.
    *Only one of the fields can be non-None* TODO(samuelstevens): encode this constraint in the type system somehow.
    """

    sql_token: Any  # Special SQL token from output vocabulary.
    value_literal: Any  # Literal copied from query.
    column: SqlColumn
    table_name: str
    nested_statement: List["SqlSpan"]

    def __init__(
        self,
        sql_token=None,
        value_literal=None,
        column=None,
        table_name=None,
        nested_statement=None,
    ):
        """SqlSpan constructor where only one argument should be non-None."""

        args = [sql_token, value_literal, column, table_name, nested_statement]
        if sum(arg is not None for arg in args) != 1:
            raise ParseError("Only one argument should be set: %s" % args)

        self.sql_token = sql_token
        self.value_literal = value_literal
        self.column = column
        self.table_name = table_name
        self.nested_statement = nested_statement

    def __str__(self) -> str:
        return repr(self)


# Represents a table and its alias in a query.
SqlTable = collections.namedtuple("SqlTable", ["table_name", "alias"])

# Represents a table in the corresponding schema.
TableSchema = collections.namedtuple(
    "TableSchema",
    [
        "table_name",  # String name of table.
        "column_names",  # List of strings for columns in table.
    ],
)

# Represents a foreign-key relation between 2 tables.
ForeignKeyRelation = NamedTuple(
    "ForeignKeyRelation",
    [
        ("child_table", str),  # String name of child table.
        ("parent_table", str),  # String name of parent table.
        ("child_column", str),  # String name of child column to join on.
        ("parent_column", str),  # String name of parent column to join on.
    ],
)

# Token to use to denote an under-specified from clause.
FROM_CLAUSE_PLACEHOLDER = "<from_clause_placeholder>"


class ParseError(Exception):
    """This exception indicates an unexpected condition during parsing."""

    pass


class UnsupportedSqlError(Exception):
    """For SQL queries that are not supported in Abstract SQL."""

    pass


class NoPathError(Exception):
    """If there is no path between an unvisited table and a visited table"""

    pass


def _get_tables(tokens):
    """Adds table(s) in sqlparse.sql.Token as SqlTable to sql_tables."""
    sql_tables = []
    table_names = set()
    previous_token = None
    in_from_clause_context = False
    for token in tokens:
        # Don't recurse into nested statements.
        if isinstance(token, list):
            continue
        elif _is_underspecified_table_mention(previous_token, token):
            for token_value in token.value.split(" "):
                sql_table = _get_sql_table(token_value)
                if sql_table.table_name in table_names:
                    raise UnsupportedSqlError(
                        "Abstract SQL does not support joining a table with itself!"
                    )
                table_names.add(sql_table.table_name)
                sql_tables.append(sql_table)
        elif _is_table_mention(previous_token, token, in_from_clause_context):
            in_from_clause_context = True
            sql_table = _get_sql_table(token.value)
            if sql_table.table_name in table_names:
                raise UnsupportedSqlError(
                    "Abstract SQL does not support joining a table with itself!"
                )
            table_names.add(sql_table.table_name)
            sql_tables.append(sql_table)
        elif token.value != ",":
            in_from_clause_context = False
        previous_token = token
    return sql_tables


def _is_underspecified_table_mention(previous_token, current_token):
    """Returns True if current_token represents a table."""
    if (
        previous_token
        and previous_token.value == FROM_CLAUSE_PLACEHOLDER
        and isinstance(current_token, sqlparse.sql.Identifier)
    ):
        return True
    return False


def _is_table_mention(previous_token, current_token, in_from_clause_context=False):
    """Returns True if current_token represents a table."""
    # Handle FROM clauses that use both `join` to separate table mentions
    # or commas.
    if previous_token and (
        previous_token.value in ("from", "join", "inner join")
        or (in_from_clause_context and previous_token.value == ",")
    ):
        if current_token.value == "(":
            # Nested query.
            return False
        elif not isinstance(current_token, sqlparse.sql.Identifier):
            raise ParseError(
                "Unexpected current token %s %s given previous token %s %s"
                % (
                    current_token.ttype,
                    current_token.value,
                    previous_token.ttype,
                    previous_token.value,
                )
            )
        return True
    return False


def _is_column_mention(sqlparse_token):
    """Returns True if sqlparse_token represents a column."""
    if not isinstance(sqlparse_token, sqlparse.sql.Identifier):
        return False
    if sqlparse_token.value.lower() in (
        "count",
        "avg",
        "min",
        "max",
        "sum",
        "distinct",
    ):
        return False
    return True


def _find_table_name_for_column(column_name, tables, table_schemas):
    """Returns table name for column or None if cannot find."""
    table_names_to_columns = {
        table.table_name: table.column_names for table in table_schemas
    }
    for table in tables:
        if table.table_name not in table_names_to_columns:
            raise ParseError("Table name %s not found in schema." % table.table_name)
        if column_name in table_names_to_columns[table.table_name]:
            return table.table_name
    return None


def _get_sql_column(column_token, tables, aliases_to_table_names, table_schemas):
    """Returns SqlColumn tuple for the given column_token string."""
    # Check if token has alias.
    column_token = column_token.lower()

    if "." in column_token:
        splits = column_token.split(".")
        if len(splits) != 2:
            raise ParseError("Unexpected token: %s" % column_token)
        if splits[0] in aliases_to_table_names:
            table_name = aliases_to_table_names[splits[0]].lower()
            return SqlColumn(column_name=splits[1], table_name=table_name)
        else:
            # Assume splits[0] is a table name.
            return SqlColumn(column_name=splits[1], table_name=splits[0].lower())
    else:
        # No prefix, check if there is only one table in statement.
        if len(tables) == 1:
            table_name = tables[0].table_name.lower()
            return SqlColumn(column_name=column_token, table_name=table_name)
        else:
            # Need to use schema information to disambiguate.
            if table_schemas is None:
                raise ParseError(
                    "Column name %s requires disambiguating between "
                    "tables %s using schema, but no schema provided."
                    % (column_token, tables)
                )
            table_name = _find_table_name_for_column(
                column_token, tables, table_schemas
            )
            if not table_name:
                raise UnsupportedSqlError(
                    "Unexpected tables %s given column token %s"
                    % (tables, column_token)
                )
            return SqlColumn(column_name=column_token, table_name=table_name.lower())


def _add_column_spans(
    sqlparse_token, tables, aliases_to_table_names, table_schemas, sql_spans
):
    """Returns True if sqlparse_token represents a column."""
    # Check whether this token is suffixed with a sorting modifier.
    column_token = sqlparse_token.value.lower()
    for suffix in (" asc", " desc"):
        if column_token.lower().endswith(suffix):
            column_token = column_token[: -len(suffix)].lower()
            sql_column = _get_sql_column(
                column_token, tables, aliases_to_table_names, table_schemas
            )
            sql_spans.append(SqlSpan(column=sql_column))
            sql_spans.append(SqlSpan(sql_token=suffix.strip()))
            return
    # Otherwise handle normal case of no suffix.
    sql_column = _get_sql_column(
        column_token, tables, aliases_to_table_names, table_schemas
    )
    sql_spans.append(SqlSpan(column=sql_column))


def _get_sql_table(token_value):
    """Returns SqlTable tuple for the given token_value."""
    token_value = token_value.lower()
    if " as " in token_value:
        # Table is aliased.
        splits = token_value.split(" as ")
        if len(splits) != 2:
            raise ParseError("Error parsing token: %s" % token_value)
        return SqlTable(table_name=splits[0], alias=splits[1])
    else:
        # Table is not aliased.
        return SqlTable(table_name=token_value, alias=None)


def _is_value_literal(sqlparse_token):
    """Returns true if sqlparse_token represents a literal to copy from query."""
    return sqlparse_token.ttype in (
        sqlparse.tokens.Literal.String.Single,
        sqlparse.tokens.Literal.Number.Integer,
        sqlparse.tokens.Literal.String.Symbol,
        sqlparse.tokens.Literal.Number.Float,
    )


def _populate_spans_for_token(
    grouped_tokens, tables, aliases_to_table_names, table_schemas, sql_spans
):
    """Add SqlSpan tuples to sql_spans given grouped_tokens and schema."""
    previous_token = None
    in_from_clause_context = False
    for token in grouped_tokens:
        is_table_mention = False
        if isinstance(token, list):
            nested_spans = []
            _populate_spans_for_statement(token, nested_spans, table_schemas)
            sql_spans.append(SqlSpan(nested_statement=nested_spans))
            previous_token = None
            continue

        if _is_underspecified_table_mention(previous_token, token):
            for token_value in token.value.split(" "):
                sql_table = _get_sql_table(token_value)
                sql_spans.append(SqlSpan(table_name=sql_table.table_name))
        elif _is_table_mention(previous_token, token, in_from_clause_context):
            sql_table = _get_sql_table(token.value)
            sql_spans.append(SqlSpan(table_name=sql_table.table_name))
            is_table_mention = True
        elif _is_column_mention(token):
            _add_column_spans(
                token, tables, aliases_to_table_names, table_schemas, sql_spans
            )
        elif token.is_group:
            _populate_spans_for_token(
                token, tables, aliases_to_table_names, table_schemas, sql_spans
            )
        elif _is_value_literal(token):
            sql_spans.append(SqlSpan(value_literal=token.value))
        else:
            # Otherwise, assume token is a SQL token.
            # TODO(petershaw): Consider not splitting (*).
            sql_spans.append(SqlSpan(sql_token=token.value))

        if is_table_mention or in_from_clause_context and token.value == ",":
            in_from_clause_context = True
        else:
            in_from_clause_context = False

        previous_token = token


def _populate_spans_for_statement(grouped_tokens, sql_spans, table_schemas):
    """Populates sql_spans for list of tokens representing a SQL statement."""
    # List of SqlTable tuples.
    tables = _get_tables(grouped_tokens)
    # TODO(petershaw): Consider unaliased tables for handling ambiguous `*` case.
    aliases_to_table_names = {table.alias: table.table_name for table in tables}
    _populate_spans_for_token(
        grouped_tokens, tables, aliases_to_table_names, table_schemas, sql_spans
    )


def _find_statement_end_idx(tokens, start_idx):
    """Return end index of nested statement starting at start_idx."""
    open_parens = 0
    for idx in range(start_idx, len(tokens)):
        token = tokens[idx]
        if token.value == "(":
            open_parens += 1
        elif token.value == ")":
            if open_parens:
                open_parens -= 1
            else:
                return idx
        elif token.value.lower() in ("exclude", "intersect", "union"):
            return idx
    return len(tokens)


def _regroup_tokens(tokens):
    """Returns recursive list of sqlparse.Token for nested statements."""
    grouped_tokens = []
    current_idx = 0
    while current_idx < len(tokens):
        token = tokens[current_idx]
        if current_idx > 0 and token.value.lower() == "select":
            end_idx = _find_statement_end_idx(tokens, current_idx)
            nested_tokens = tokens[current_idx : end_idx + 1]
            grouped_tokens.append(_regroup_tokens(nested_tokens))
            current_idx = end_idx + 1
        else:
            grouped_tokens.append(token)
            current_idx += 1
    return grouped_tokens


def _get_flattened_non_whitespace_tokens(sqlparse_token):
    """Returns list of sqlparse Tokens."""
    flat_tokens = []
    for token in sqlparse_token:
        if token.ttype == sqlparse.tokens.Whitespace:
            continue
        elif isinstance(token, sqlparse.sql.Identifier):
            flat_tokens.append(token)
        elif token.is_group:
            flat_tokens.extend(_get_flattened_non_whitespace_tokens(token))
        else:
            flat_tokens.append(token)
    return flat_tokens


def _check_spans_against_schema(sql_spans, table_schemas):
    """Raises exception if sql_spans contains bad table or column name."""
    table_names_to_columns = {
        table.table_name: table.column_names for table in table_schemas
    }
    for sql_span in sql_spans:
        if sql_span.table_name:
            if sql_span.table_name not in table_names_to_columns:
                raise ParseError(
                    "Table name %s not in schema %s"
                    % (sql_span.table_name, table_schemas)
                )
        elif sql_span.column:
            if sql_span.column.table_name not in table_names_to_columns:
                raise ParseError(
                    "Table name %s not in schema %s"
                    % (sql_span.column.table_name, table_schemas)
                )
            else:
                column_names = table_names_to_columns[sql_span.column.table_name]
                if sql_span.column.column_name not in column_names:
                    raise ParseError(
                        "Column name %s not in schema %s for table %s for spans %s"
                        % (
                            sql_span.column.column_name,
                            table_schemas,
                            sql_span.column.table_name,
                            sql_spans_to_string(sql_spans, sep=","),
                        )
                    )


def _replace_from_placeholder(tokens):
    """Replace 'FROM' with placeholder symbol in tokens."""
    for token in tokens:
        if token.value.lower() == "from":
            token.value = FROM_CLAUSE_PLACEHOLDER


# TODO(alanesuhr): Test with lowercase=False (or make that the default)
def sql_to_sql_spans(sql_string, table_schemas=None, lowercase=True):
    """Parse sql_string to a list of SqlSpan tuples.

  Args:
    sql_string: The SQL query to convert.
    table_schemas: List of TableSchema tuples.
    lowercase: Whether to lowercase the SQL query.

  Returns:
    List of SqlSpan tuples.

  Raises:
    ParseError: If the SQL query can't be parsed for unexpected reason.
    UnsupportedSqlError: SQL query is not supported by Abstract SQL.
  """
    if sql_string in BAD_SQL:
        raise UnsupportedSqlError("Query matched list of malformed queries.")
    # Get the root sqlparse.sql.Token for the expression.
    if lowercase:
        sql_string = sql_string.lower()
    sql_string = sql_string.strip()
    sql_string = sql_string.rstrip(";")
    # sqlparse expects value literals to have single quotes.
    sql_string = sql_string.replace('"', "'")
    # Change FROM clause placeholder to avoid breaking sqlparse.
    has_from_placeholder = FROM_CLAUSE_PLACEHOLDER in sql_string
    if has_from_placeholder:
        sql_string = sql_string.replace(FROM_CLAUSE_PLACEHOLDER, "from")
    # Parse using sqlparse.
    sqlparse_output = sqlparse.parse(sql_string)
    if not sqlparse_output:
        raise ParseError("sqlparse.parse failed for %s" % sql_string)
    sqlparse_token = sqlparse_output[0]
    flat_tokens = _get_flattened_non_whitespace_tokens(sqlparse_token)
    if has_from_placeholder:
        _replace_from_placeholder(flat_tokens)
    grouped_tokens = _regroup_tokens(flat_tokens)
    sql_spans = []
    _populate_spans_for_statement(grouped_tokens, sql_spans, table_schemas)
    if table_schemas:
        _check_spans_against_schema(sql_spans, table_schemas)
    # Add back placeholder for FROM clause.
    return sql_spans


def sql_spans_to_string(sql_spans, sep=" "):
    """Converts list of SqlSpan tuples to string."""
    strings = []
    for span in sql_spans:
        if span.sql_token:
            strings.append(span.sql_token)
        elif span.column:
            strings.append("%s.%s" % (span.column.table_name, span.column.column_name))
        elif span.table_name:
            strings.append(span.table_name)
        elif span.value_literal:
            strings.append(span.value_literal)
        elif span.nested_statement:
            strings.append(sql_spans_to_string(span.nested_statement))
    return sep.join(strings)


def _get_table_names_from_columns(sql_spans: Sequence[SqlSpan]) -> Set[str]:
    """
    Return set of table names of SqlColumn tuples in sql_spans.

    Finds all columns not in a FROM clause
    """
    table_names_from_columns: Set[str] = set()
    in_from_clause = False
    for span in sql_spans:
        if in_from_clause:
            if span.sql_token and span.sql_token.lower() not in (
                "join",
                "on",
                "=",
                "inner join",
            ):
                in_from_clause = False
        else:
            if span.sql_token and span.sql_token.lower() == "from":
                in_from_clause = True
        if not in_from_clause and span.column:
            table_names_from_columns.add(span.column.table_name)
    return table_names_from_columns


def _get_tables_without_column_copies(sql_spans: Sequence[SqlSpan]) -> List[str]:
    """
    Returns a list of table names with no corresponding column copies.

    These are table names only referenced in a FROM clause (not in WHERE, ORDER BY, SELECT, etc.)
    """
    table_names_from_columns = _get_table_names_from_columns(sql_spans)
    orphaned_tables = []
    for span in sql_spans:
        if (
            span.table_name
            and span.table_name not in table_names_from_columns
            and span.table_name not in orphaned_tables
        ):
            orphaned_tables.append(span.table_name)
    return orphaned_tables


def replace_from_clause(sql_spans):
    """
    Replace fully-specified FROM clause(s) with under-specified FROM clause(s).

    Arguments:
        sql_spans: List of SqlSpan tuples with fully-specified FROM clause.

    Returns:
        List of SqlSpan tuples with under-specified FROM clause.
    """
    replaced_spans = []
    orphaned_tables = _get_tables_without_column_copies(sql_spans)
    # TODO(petershaw): Consider throwing exception if we never find from clause
    # or find more than 1.
    in_from_clause = False
    for span in sql_spans:
        if in_from_clause:
            if span.sql_token and span.sql_token.lower() not in (
                "join",
                "on",
                "=",
                "inner join",
            ):
                in_from_clause = False
        else:
            if span.sql_token and span.sql_token.lower() == "from":
                in_from_clause = True
                # Add placeholder and orphaned tables.
                replaced_spans.append(SqlSpan(sql_token=FROM_CLAUSE_PLACEHOLDER))
                for table_name in orphaned_tables:
                    replaced_spans.append(SqlSpan(table_name=table_name))
        if not in_from_clause:
            if span.nested_statement:
                # Recursively remove FROM clause from nested expression.
                replaced_spans.append(
                    SqlSpan(nested_statement=replace_from_clause(span.nested_statement))
                )
            else:
                replaced_spans.append(span)
    return replaced_spans


T = TypeVar("T")


def _construct_path(start: T, end: T, parents: Mapping[T, T]) -> List[T]:
    path = [end]
    current = end
    while current in parents:
        path.append(parents[current])
        current = parents[current]

    return list(reversed(path))


def _bfs(start: T, goals: Set[T], adjacency: Mapping[T, Set[T]]) -> List[T]:
    """
    Perform breadth-first search starting at start and ending at anything in goals.
    """

    seen: Set[T] = set()
    parents: Dict[T, T] = {}

    # 2. initialize the queue
    queue = Deque([start])
    seen.add(start)

    while queue:
        node = queue.pop()
        assert node in seen

        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            parents[neighbor] = node
            queue.append(neighbor)

            if neighbor in goals:
                return _construct_path(start, neighbor, parents)

    raise NoPathError(
        "No path from %s to goals %s via edges %s" % (start, goals, adjacency)
    )


def _shortest_path(
    unvisited_tables: Sequence[str],
    visited_tables: Sequence[str],
    relations: Mapping[Tuple[str, str], Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Finds the shortest path between any unvisited table and a visited table.

    Performs a BFS starting at each unvisited table and ending when a visited table is reached and keeps track of the shortest path.
    """

    # set up graph
    adjacency: Dict[str, Set[str]] = {}
    for a, b in relations:
        if a not in adjacency:
            adjacency[a] = set()
        adjacency[a].add(b)

        if b not in adjacency:
            adjacency[b] = set()
        adjacency[b].add(a)

    best_path = None

    for start in unvisited_tables:
        path = _bfs(start, set(visited_tables), adjacency)

        if not best_path or len(path) < len(best_path):
            best_path = path

    if best_path is None:
        # In practice this can only be raised if there are no unvisited tables, and the for loop is never entered.
        raise NoPathError(
            "No path found between unvisited tables %s and visited tables %s with relations %s"
            % (unvisited_tables, visited_tables, relations)
        )

    return [(a, b) for a, b in zip(best_path, best_path[1:])]


def _get_fk_relations_helper(
    unvisited_tables,
    visited_tables,
    fk_relations_map: Mapping[Tuple[str, str], Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Returns a list of foreign key relations used to join two tables. First, single edge paths are joined, then if there are still unvisited tables, a graph is constructed.

    Modifies unvisited_tables and visited_tables.
    """
    for table_to_visit in unvisited_tables:
        for table in visited_tables:
            if (table, table_to_visit) in fk_relations_map:
                fk_relation = fk_relations_map[(table, table_to_visit)]
                unvisited_tables.remove(table_to_visit)
                visited_tables.append(table_to_visit)
                return [fk_relation]

    # If we reach this point, there are no single edge paths between visited and unvisited. Instead, we must construct a graph using all tables referenced in fk_relations_map and find the shortest path between unvisited and visited tables.
    path = _shortest_path(unvisited_tables, visited_tables, fk_relations_map)
    col_names = []

    for t1, t2 in path:
        if t1 in unvisited_tables:
            unvisited_tables.remove(t1)
        if t1 not in visited_tables:
            visited_tables.append(t1)

        if t2 in unvisited_tables:
            unvisited_tables.remove(t2)
        if t2 not in visited_tables:
            visited_tables.append(t2)

        col_names.append(fk_relations_map[(t1, t2)])

    return col_names


def _get_fk_relations_linking_tables(
    table_names: Sequence[str], known_fk_relations: Sequence[ForeignKeyRelation]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns (List of visited table names, List of (col name, col name)).
    """

    if not table_names:
        raise UnsupportedSqlError("The SQL query has not referenced any tables!")

    # Build map of (child table, parent table) to (child column, parent column).
    fk_relations_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for relation in known_fk_relations:
        # TODO(petershaw): Consider adding warning if overwriting key.
        # This can lead to ambiguous conversions back to fully-specified SQL.
        fk_relations_map[(relation.child_table, relation.parent_table)] = (
            relation.child_column,
            relation.parent_column,
        )
        # Also add the reverse.
        fk_relations_map[(relation.parent_table, relation.child_table)] = (
            relation.parent_column,
            relation.child_column,
        )

    # Greedily connect all tables using available foreign key relations.
    # Artbirary choose the first table_name to start.
    visited_tables = [table_names[0]]
    unvisited_tables = table_names[1:]
    fk_relations = []
    while unvisited_tables:
        try:
            relations = _get_fk_relations_helper(
                unvisited_tables, visited_tables, fk_relations_map
            )
            fk_relations.extend(relations)
        except NoPathError:
            raise UnsupportedSqlError(
                "Couldn't find path between tables %s given relations %s."
                % (table_names, known_fk_relations)
            )
    # Length of fk_relations will be 1 shorter than visited tables.
    # samuelstevens: added assert to match original authors' comment above
    assert len(visited_tables) - 1 == len(
        fk_relations
    ), f"Internal consistency error: {len(visited_tables)} is not 1 + {len(fk_relations)}"
    return visited_tables, fk_relations


def _get_from_clause_for_tables(
    table_names: List[str], known_fk_relations: Sequence[ForeignKeyRelation]
):
    """Returns list of SqlSpan tuples for FROM clause."""
    visited_tables, fk_relations = _get_fk_relations_linking_tables(
        table_names, known_fk_relations
    )
    sql_spans = []
    sql_spans.append(SqlSpan(table_name=visited_tables[0]))
    for i in range(len(visited_tables) - 1):
        table_a = visited_tables[i]
        table_b = visited_tables[i + 1]
        column_a, column_b = fk_relations[i]
        # join table_b on table_a.column_a = table_b.column_b
        sql_spans.append(SqlSpan(sql_token="join"))
        sql_spans.append(SqlSpan(table_name=table_b))
        sql_spans.append(SqlSpan(sql_token="on"))
        sql_spans.append(
            SqlSpan(column=SqlColumn(column_name=column_a, table_name=table_a))
        )
        sql_spans.append(SqlSpan(sql_token="="))
        sql_spans.append(
            SqlSpan(column=SqlColumn(column_name=column_b, table_name=table_b))
        )
    return sql_spans


def _get_all_table_names(sql_spans: Sequence[SqlSpan]) -> List[str]:
    """Returns set of all table names in sql_spans."""
    # Use a list to preserve some ordering.
    table_names = []
    for sql_span in sql_spans:
        if sql_span.table_name:
            if sql_span.table_name not in table_names:
                table_names.append(sql_span.table_name)
        elif sql_span.column:
            if sql_span.column.table_name not in table_names:
                table_names.append(sql_span.column.table_name)
    return table_names


def _has_nested_from_clause(sql_spans):
    """Returns True if the outer sql statement has nested from clause."""
    previous_span_was_placeholder = False
    for sql_span in sql_spans:
        if (
            previous_span_was_placeholder
            and sql_span.sql_token
            and sql_span.sql_token == "("
        ):
            return True
        if sql_span.sql_token and sql_span.sql_token == FROM_CLAUSE_PLACEHOLDER:
            previous_span_was_placeholder = True
        else:
            previous_span_was_placeholder = False
    return False


def restore_from_clause(sql_spans, fk_relations):
    """
    Restores fully-specified FROM clause.

    Args:
        sql_spans: List of SqlSpan tuples with underspecified FROM clause(s).
        fk_relations: List of ForeignKeyRelation tuples for given schema.

    Returns:
        List of SqlSpan tuples with fully specified FROM clause(s).
    Raises:
        UnsupportedSqlError: If cannot find path between mentioned tables
    """
    # Sort the list to avoid non-determinism.
    table_names = sorted(_get_all_table_names(sql_spans))
    has_nested_from_clause = _has_nested_from_clause(sql_spans)
    restored_sql_spans = []
    in_from_clause = False
    for sql_span in sql_spans:
        if sql_span.nested_statement:
            restored_sql_spans.append(
                SqlSpan(
                    nested_statement=restore_from_clause(
                        sql_span.nested_statement, fk_relations
                    )
                )
            )
            continue

        if in_from_clause:
            if not sql_span.table_name:
                in_from_clause = False
        else:
            if sql_span.sql_token and sql_span.sql_token == FROM_CLAUSE_PLACEHOLDER:
                restored_sql_spans.append(SqlSpan(sql_token="from"))
                if not has_nested_from_clause:
                    restored_sql_spans.extend(
                        _get_from_clause_for_tables(table_names, fk_relations)
                    )
                    in_from_clause = True
                continue

        if not in_from_clause:
            restored_sql_spans.append(sql_span)
    return restored_sql_spans
