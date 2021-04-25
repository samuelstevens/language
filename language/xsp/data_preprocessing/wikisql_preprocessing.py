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
"""Utilities for loading the WikiSQL dataset."""

from __future__ import absolute_import, division, print_function

import json
from typing import Any, Optional, Tuple, Union, cast

import sqlparse

from language.xsp.data_preprocessing import abstract_sql, abstract_sql_converters
from language.xsp.data_preprocessing.nl_to_sql_example import (
    NLToSQLExample,
    populate_utterance,
)
from language.xsp.data_preprocessing.sql_parsing import ParseError, populate_sql
from language.xsp.data_preprocessing.sql_utils import preprocess_sql


def normalize_sql(sql, replace_period=True):
    """Normalizes WikiSQL SQL queries."""
    sql = sql.replace("_/_", "_OR_")
    sql = sql.replace("/", "_OR_")
    sql = sql.replace("?", "")

    if replace_period:
        sql = sql.replace(".", "")
        sql = sql.replace("(", "")
        sql = sql.replace(")", "")
    sql = sql.replace("%", "")
    return sql


def normalize_entities(entity_name):
    """Normalizes database entities (table and column names)."""
    entity_name = normalize_sql(entity_name)
    entity_name = entity_name.replace(" ", "_").upper()
    return entity_name


def convert_wikisql(
    input_example: Union[Tuple[str, str], Tuple[str, str, Any]],
    schema,
    tokenizer,
    generate_sql: bool,
    anonymize_values: bool,
    use_abstract_sql: bool,
    tables_schema=None,
    allow_value_generation=False,
) -> Optional[NLToSQLExample]:
    """
    Converts a WikiSQL example into a NLToSQLExample.
    """

    if len(input_example) == 2:
        # https://github.com/python/mypy/issues/1178
        nl_str, sql_str = cast(Tuple[str, str], input_example)
    else:
        nl_str, sql_str, _ = cast(Tuple[str, str, Any], input_example)

    example = NLToSQLExample.empty(nl_str)

    try:
        try:
            populate_utterance(example, schema, tokenizer)
        except ValueError as e:
            print("Couldn't populate utterance in wikisql example:")
            print(e)
            return None

        # WikiSQL databases have a single table.
        assert len(schema) == 1

        # Some preprocessing of the WikiSQL SQL queries.
        sql = input_example[1].rstrip("; ")
        sql = sql.replace("TABLE", list(schema.keys())[0])
        sql = sql.replace("_FIELD", "")
        string_split_sql = sql.split(" ")
        if string_split_sql[1].lower() in {"count", "min", "max", "avg", "sum"}:
            # Add parentheses around the column that's an argument of any of these
            # aggregate functions (because gold annotations don't have it).
            sql = " ".join(
                string_split_sql[0:2]
                + ["(", string_split_sql[2], ")"]
                + string_split_sql[3:]
            )

        sql = normalize_sql(sql, replace_period=False)

        try:
            sql = preprocess_sql(sql)
        except UnicodeDecodeError:
            print("Couldn't preprocess sql in wikisql example:")
            return None

        sql = sql.lower()
        parsed_sql = sqlparse.parse(sql)[0]

        successful_copy = True
        if generate_sql:
            try:
                if use_abstract_sql:
                    successful_copy = abstract_sql_converters.populate_abstract_sql(
                        example, sql, tables_schema, anonymize_values
                    )
                else:
                    successful_copy = populate_sql(
                        parsed_sql, example, anonymize_values
                    )
            except (
                ParseError,
                ValueError,
                AssertionError,
                KeyError,
                IndexError,
                abstract_sql.ParseError,
                abstract_sql.UnsupportedSqlError,
            ):
                return None

        if not successful_copy and not allow_value_generation:
            return None

        if not example.gold_sql_query.actions:
            return None
        elif example.gold_sql_query.actions[-1].symbol == "=":
            return None

    except UnicodeEncodeError as e:
        print(e)
        return None

    return example


def load_wikisql_tables(filepath):
    """Loads the WikiSQL tables from a path and reformats as the format."""
    dbs = dict()
    with open(filepath) as infile:
        tables = [json.loads(line) for line in infile if line]

    for table in tables:
        db_dict = dict()
        table_name = (
            table["section_title"]
            if "section_title" in table and table["section_title"]
            else (table["name"] if "name" in table else table["page_title"])
        )

        table_name = normalize_entities(table_name)

        db_dict[table_name] = list()
        for column_name, column_type in zip(table["header"], table["types"]):
            if column_type == "real":
                column_type = "number"
            assert column_type in {"text", "number"}, column_type
            column_name = normalize_entities(column_name)

            db_dict[table_name].append(
                {
                    "field name": column_name,
                    "is primary key": False,
                    "is foreign key": False,
                    "type": column_type,
                }
            )
        if table["id"] not in dbs:
            dbs[table["id"]] = db_dict

    return dbs
