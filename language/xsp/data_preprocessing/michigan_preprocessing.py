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
"""Loads the Michigan datasets from a file."""

from __future__ import absolute_import, division, print_function

import csv
import json
from typing import Any, FrozenSet, List, Optional, Tuple, Union, cast, overload

import sqlparse
import tensorflow.compat.v1.gfile as gfile
from typing_extensions import Literal

import language.xsp.data_preprocessing.sql_parsing as sql_parsing
from bert.tokenization import FullTokenizer
from language.xsp.data_preprocessing import abstract_sql_converters
from language.xsp.data_preprocessing.abstract_sql import TableSchema
from language.xsp.data_preprocessing.nl_to_sql_example import (
    NLToSQLExample,
    populate_utterance,
)
from language.xsp.data_preprocessing.schema_utils import Column, Schema, TableName
from language.xsp.data_preprocessing.sql_utils import preprocess_sql


@overload
def get_nl_sql_pairs(
    filepath: str, splits: FrozenSet[str], with_dbs: Literal[True]
) -> List[Tuple[str, str, Any]]:
    ...


@overload
def get_nl_sql_pairs(
    filepath: str, splits: FrozenSet[str], with_dbs: Literal[False]
) -> List[Tuple[str, str]]:
    ...


@overload
def get_nl_sql_pairs(filepath: str, splits: FrozenSet[str]) -> List[Tuple[str, str]]:
    ...


def get_nl_sql_pairs(
    filepath: str, splits: FrozenSet[str], with_dbs: bool = False
) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Any]]]:
    """Gets pairs of natural language and corresponding gold SQL for Michigan."""
    with open(filepath) as infile:
        data = json.load(infile)

    pairs = []  # type: ignore

    tag = "[" + filepath.split("/")[-1].split(".")[0] + "]"
    print("Getting examples with tag " + tag)

    # The UMichigan data is split by anonymized queries, where values are
    # anonymized but table/column names are not. However, our experiments are
    # performed on the original splits of the data.
    for query in data:
        # Take the first SQL query only. From their Github documentation:
        # "Note - we only use the first query, but retain the variants for
        #  completeness"
        anonymized_sql = query["sql"][0]

        variable_count = max(
            len(example["variables"]) for example in query["sentences"]
        )

        # It's also associated with a number of natural language examples, which
        # also contain anonymous tokens. Save the de-anonymized utterance and query.
        for example in query["sentences"]:
            if example["question-split"] not in splits:
                continue

            # If we don't have placeholders for all the variables, skip this example.
            if len(example["variables"]) < variable_count:
                continue

            assert len(example["variables"]) == variable_count

            nl: str = example["text"]
            sql: str = anonymized_sql

            # Go through the anonymized values and replace them in both the natural
            # language and the SQL.
            #
            # It's very important to sort these in descending order. If one is a
            # substring of the other, it shouldn't be replaced first lest it ruin the
            # replacement of the superstring.
            #
            # (samuelstevens) Furthermore, not all example in query['sentences'] has the same number of variables.
            # If that's the case, only use the examples with all variables.
            for variable_name, value in sorted(
                example["variables"].items(), key=lambda x: len(x[0]), reverse=True
            ):
                if not value:
                    # TODO(alanesuhr) While the Michigan repo says to use a - here, the
                    # thing that works is using a % and replacing = with LIKE.

                    # It's possible that I should remove such clauses from the SQL, as
                    # long as they lead to the same table result. They don't align well
                    # to the natural language at least.

                    # See: https://github.com/jkkummerfeld/text2sql-data/tree/master/data
                    value = "%"

                nl = nl.replace(variable_name, value)
                sql = sql.replace(variable_name, value)

            # In the case that we replaced an empty anonymized value with %, make it
            # compilable new allowing equality with any string.
            sql = sql.replace('= "%"', 'LIKE "%"')

            if with_dbs:
                pairs.append((nl, sql, example["table-id"]))  # type: ignore
            else:
                pairs.append((nl, sql))  # type: ignore

    return pairs  # type: ignore


def read_schema(schema_csv: str) -> Schema:
    """Loads a database schema from a CSV representation."""
    tables: Schema = {}

    with gfile.Open(schema_csv) as infile:
        for ordered_col in csv.DictReader(
            infile,
            quotechar='"',
            delimiter=",",
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        ):
            column = {
                key.lower().strip(): value for key, value in ordered_col.items() if key
            }

            table_name = column["table name"]
            if table_name != "-":
                if table_name not in tables:
                    tables[TableName(table_name)] = list()
                column.pop("table name")
                tables[TableName(table_name)].append(cast(Column, column))
    return tables


def convert_michigan(
    nl_query: str,
    sql_str: str,
    schema: Schema,
    tokenizer: FullTokenizer,
    generate_sql: bool,
    anonymize_values: bool,
    abstract_sql: bool,
    table_schemas: Optional[List[TableSchema]],
    allow_value_generation: bool,
) -> Optional[NLToSQLExample]:
    """
    Converts a Michigan example to the standard format.

    Args:
        nl_query: natural language query
        sql: SQL query
        schema: JSON object for SPIDER schema in converted format.
        wordpiece_tokenizer: language.bert.tokenization.FullTokenizer instance.
        generate_sql: If True, will populate SQL.
        anonymize_values: If True, anonymizes values in SQL.
        abstract_sql: If True, use under-specified FROM clause.
        table_schemas: required if abstract_sql, list of TableSchema tuples.
        allow_value_generation: Allow value generation.

    Returns:
        NLToSQLExample instance.
    """
    example = NLToSQLExample.empty(nl_query)
    populate_utterance(example, schema, tokenizer)

    # gold_sql_query =

    # Set the output
    successful_copy = True
    if generate_sql:
        if abstract_sql:
            assert table_schemas
            successful_copy = abstract_sql_converters.populate_abstract_sql(
                example, sql_str, table_schemas, anonymize_values
            )
        else:
            sql_query: sqlparse.sql.Statement = sqlparse.parse(
                preprocess_sql(sql_str.rstrip("; ").lower())
            )[0]
            try:
                successful_copy = sql_parsing.populate_sql(
                    sql_query, example, anonymize_values
                )
            except sql_parsing.ParseError as e:
                print(e)
                successful_copy = False

    # If the example contained an unsuccessful copy action, and values should not
    # be generated, then return an empty example.
    if not successful_copy and not allow_value_generation:
        return None

    return example

    if generate_sql:
        raise ValueError(
            "Generating annotated SQL is not yet supported for Michigan datasets. "
            "Tried to annotate: " + sql_query
        )
