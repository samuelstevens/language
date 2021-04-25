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
"""Contains utilities for processing database schemas."""

from __future__ import absolute_import, division, print_function

import dataclasses
import json
from typing import Any, Dict, List, NewType, Set

from typing_extensions import TypedDict

from language.xsp.data_preprocessing.language_utils import Wordpiece, get_wordpieces

ACCEPTABLE_COL_TYPES = {"text", "number", "others", "time", "boolean"}

Column = TypedDict(
    "Column",
    {"field name": str, "is primary key": bool, "is foreign key": bool, "type": str},
)

TableName = NewType("TableName", str)
Schema = Dict[TableName, List[Column]]


@dataclasses.dataclass
class TableColumn:
    """Contains information about column in a database table."""

    column_type: str
    original_column_name: str
    column_name_wordpieces: List[Wordpiece]
    table_name: str
    is_foreign_key: bool
    matches_to_utterance: Any

    def to_json(self) -> Dict[str, Any]:
        assert self.column_type in ACCEPTABLE_COL_TYPES, (
            "Column type not " "recognized: %r; name: %r"
        ) % (self.column_type, self.original_column_name)
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(dictionary) -> "TableColumn":
        """Sets the properties of the column from a dictionary representation."""
        original_column_name = dictionary["original_column_name"]
        column_name_wordpieces = [
            Wordpiece.from_json(wordpiece)
            for wordpiece in dictionary["column_name_wordpieces"]
        ]
        column_type = dictionary["column_type"]
        table_name = dictionary["table_name"]
        is_foreign_key = dictionary["is_foreign_key"]
        matches_to_utterance = dictionary["matches_to_utterance"]
        assert column_type in ACCEPTABLE_COL_TYPES, (
            "Column type not " "recognized: %r; name: %r"
        ) % (column_type, original_column_name)

        return TableColumn(
            column_type,
            original_column_name,
            column_name_wordpieces,
            table_name,
            is_foreign_key,
            matches_to_utterance,
        )


@dataclasses.dataclass
class DatabaseTable:
    """Contains information about a table in a database."""

    original_table_name: str
    table_name_wordpieces: List[Wordpiece]
    table_columns: List[TableColumn]
    matches_to_utterance: Any

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(dictionary) -> "DatabaseTable":
        """Converts from a JSON dictionary to a DatabaseTable object."""
        original_table_name = dictionary["original_table_name"]
        table_name_wordpieces = [
            Wordpiece.from_json(wordpiece)
            for wordpiece in dictionary["table_name_wordpieces"]
        ]
        table_columns = [
            TableColumn.from_json(column) for column in dictionary["table_columns"]
        ]
        matches_to_utterance = dictionary["matches_to_utterance"]

        return DatabaseTable(
            original_table_name,
            table_name_wordpieces,
            table_columns,
            matches_to_utterance,
        )

    def __str__(self):
        return json.dumps(self.to_json())


def column_is_primary_key(column):
    """Returns whether a column object is marked as a primary key."""
    primary_key = column["is primary key"]
    if isinstance(primary_key, str):
        primary_key = primary_key.lower()
        if primary_key in {"y", "n", "yes", "no", "-"}:
            primary_key = primary_key in {"y", "yes"}
        else:
            raise ValueError("primary key should be a boolean: " + primary_key)
    return primary_key


def column_is_foreign_key(column):
    """Returns whether a column object is marked as a foreign key."""
    foreign_key = column["is foreign key"]
    if isinstance(foreign_key, str):
        foreign_key = foreign_key.lower()

        if foreign_key in {"y", "n", "yes", "no", "-"}:
            foreign_key = foreign_key in {"y", "yes"}
        else:
            raise ValueError(
                "Foreign key should be a boolean: "
                + foreign_key
                + ". Context: "
                + str(column)
            )

    return foreign_key


def process_columns(
    columns, tokenizer, table_name, aligned_schema_entities
) -> List[TableColumn]:
    """Processes a column in a table to a TableColumn object."""
    column_obj_list = list()
    for column in columns:
        original_column_name = column["field name"]
        column_name_wordpieces = get_wordpieces(
            original_column_name.replace("_", " "), tokenizer
        )[0]
        col_type = column["type"].lower()
        if (
            "int" in col_type
            or "float" in col_type
            or "double" in col_type
            or "decimal" in col_type
        ):
            col_type = "number"
        if "varchar" in col_type or "longtext" in col_type:
            col_type = "text"
        column_type = col_type
        table_name = table_name

        matches_to_utterance = (
            original_column_name.lower().replace("_", " ") in aligned_schema_entities
        )

        is_foreign_key = column_is_foreign_key(column)

        column_obj_list.append(
            TableColumn(
                column_type,
                original_column_name,
                column_name_wordpieces,
                table_name,
                is_foreign_key,
                matches_to_utterance,
            )
        )
    return column_obj_list


def process_table(
    table_name, columns, tokenizer, aligned_schema_entities
) -> DatabaseTable:
    """Processes a schema table into a DatabaseTable object."""
    original_table_name = table_name

    matches_to_utterance = (
        original_table_name.lower().replace("_", " ") in aligned_schema_entities
    )

    # Name wordpieces. Remove underscores then tokenize.
    table_name_wordpieces = get_wordpieces(table_name.replace("_", " "), tokenizer)[0]

    table_columns = process_columns(
        columns, tokenizer, original_table_name, aligned_schema_entities
    )

    return DatabaseTable(
        original_table_name, table_name_wordpieces, table_columns, matches_to_utterance
    )


def process_tables(schema, tokenizer, aligned_schema_entities):
    """Processes each table in a schema."""
    return [
        process_table(table_name, columns, tokenizer, aligned_schema_entities)
        for table_name, columns in schema.items()
    ]


def get_schema_entities(schema: Schema) -> Set[str]:
    """
    Gets the schema entities (column and table names) for a schema.
    """
    names = set()
    for table_name, cols in schema.items():
        names.add(table_name.lower().replace("_", " "))
        for col in cols:
            names.add(col["field name"].lower().replace("_", " "))
    return names
