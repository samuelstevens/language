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
"""Utilities for processing the SQL output."""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional

import sqlparse
from prelude import flattened

from language.xsp.data_preprocessing import sqlparse_keyword_utils
from language.xsp.data_preprocessing.language_utils import Wordpiece
from language.xsp.data_preprocessing.schema_utils import DatabaseTable, TableColumn

sqlparse_keyword_utils.remove_bad_sqlparse_keywords()


def tokenize(sql: str) -> List[str]:
    """Tokenizes a SQL query into a list of SQL tokens."""
    parsed = sqlparse.parse(sql)[0]

    # (samuelstevens) Rejoin functions with their parentheses: [count, (, 1, )] -> [count(, 1, )]
    def flatten_except_functions(chunk) -> List[str]:
        if isinstance(chunk, sqlparse.sql.Function):
            real_tokens = [
                tok for tok in chunk.tokens if tok.ttype != sqlparse.tokens.Whitespace
            ]
            assert len(real_tokens) == 2
            ident, paren = real_tokens

            open_paren, *raw_args, close_paren = paren.tokens

            assert open_paren.value == "("
            assert close_paren.value == ")"

            args = flattened([flatten_except_functions(tok) for tok in raw_args])

            func_name = ident.value.strip()

            return [func_name + "(", *args, ")"]

        if isinstance(chunk, sqlparse.sql.TokenList):
            return flattened([flatten_except_functions(tok) for tok in chunk])

        return [chunk.value.strip()] if chunk.value.strip() else []

    # want to flatten all tokens EXCEPT functions
    return flatten_except_functions(parsed)


def anonymize_aliases(sql: str) -> List[str]:
    """Renames aliases to a consistent format (e.g., using T#)."""

    sql_tokens = []
    tokens = tokenize(sql)

    # First, split all TABLE.COLUMN examples into three tokens.
    for token in tokens:
        token = token.replace('"', "'")
        if (
            token != "."
            and token.count(".") == 1
            and not token.replace(".", "", 1).isnumeric()
        ):
            table, column = token.split(".")
            sql_tokens.extend([table, ".", column])
        else:
            sql_tokens.append(token)

    # Create an alias dictionary that maps from table names to column names
    alias_dict: Dict[str, str] = {}
    for token in sql_tokens:
        if "alias" in token and token not in alias_dict:
            alias_dict[token] = "T" + str(len(alias_dict) + 1)

    # Reconstruct the SQL query, this time replacing old alias names with the new
    # assigned alias names.
    new_tokens = []
    for token in sql_tokens:
        if token in alias_dict:
            new_tokens.append(alias_dict[token])
        else:
            new_tokens.append(token)

    return new_tokens


def preprocess_sql(sql: str) -> str:
    """Preprocesses a SQL query into a clean string form."""
    return " ".join(anonymize_aliases(sql)).replace(" . ", ".")


class SchemaEntityCopy:
    """A copy action from the schema."""

    copied_table: Optional[DatabaseTable]
    copied_column: Optional[TableColumn]

    def __init__(
        self, copied_table: DatabaseTable = None, copied_column: TableColumn = None
    ):
        self.copied_table = copied_table
        self.copied_column = copied_column

    def to_json(self):
        if self.copied_table:
            return {"copied_table": self.copied_table.to_json()}
        if self.copied_column:
            return {"copied_column": self.copied_column.to_json()}

    @staticmethod
    def from_json(dictionary) -> "SchemaEntityCopy":
        if "copied_table" in dictionary:
            copied_table = DatabaseTable.from_json(dictionary["copied_table"])
            return SchemaEntityCopy(copied_table=copied_table)

        if "copied_column" in dictionary:
            copied_column = TableColumn.from_json(dictionary["copied_column"])
            return SchemaEntityCopy(copied_column=copied_column)

        raise ValueError(
            f"Dictionary {dictionary} does not contain keys 'copied_table' or 'copied_column'"
        )

    def __repr__(self):
        return (
            f"(copied_table: {self.copied_table}, copied_column: {self.copied_column})"
        )


class SQLAction(object):
    """Describes a single generation action for a SQL query."""

    def __init__(self, symbol=None, entity_copy=None, utterance_copy=None):
        # Make sure only one of the things are set.
        assert len([obj for obj in [symbol, entity_copy, utterance_copy] if obj]) <= 1

        self.symbol = symbol
        self.entity_copy = entity_copy
        self.utterance_copy = utterance_copy

    def to_json(self):
        if self.symbol:
            return {"symbol": self.symbol}
        if self.entity_copy:
            return {"entity_copy": self.entity_copy.to_json()}
        if self.utterance_copy:
            return {"utterance_copy": self.utterance_copy.to_json()}

    def from_json(self, dictionary) -> "SQLAction":
        """Converts from a JSON representation to a SQL action."""
        # Should only have one key -- any of the above keys.
        assert len(dictionary) == 1
        if "symbol" in dictionary:
            self.symbol = dictionary["symbol"]
            return self

        if "entity_copy" in dictionary:
            self.entity_copy = SchemaEntityCopy().from_json(dictionary["entity_copy"])
            return self

        if "utterance_copy" in dictionary:
            self.utterance_copy = Wordpiece.from_json(dictionary["utterance_copy"])
            return self

        raise ValueError(
            f"Dictionary with keys {dictionary.keys()} does not contain a good key!"
        )

    def __repr__(self):
        return f"SQLAction(symbol='{self.symbol}', entity_copy='{self.entity_copy}', utterance_copy='{self.utterance_copy}')"


class SQLQuery(object):
    """Contains information about a SQL query grounded in an utterance/schema."""

    actions: List[SQLAction]

    def __init__(self, actions: List[SQLAction]):
        self.actions = actions

    def to_json(self):
        return {"actions": [action.to_json() for action in self.actions]}

    @staticmethod
    def from_json(dictionary: Dict[str, Any]) -> "SQLQuery":
        actions = []
        for action in dictionary["actions"]:
            actions.append(SQLAction().from_json(action))
        return SQLQuery(actions)

    def __repr__(self):
        return f"(actions: {self.actions})"

    @staticmethod
    def empty() -> "SQLQuery":
        return SQLQuery([])
