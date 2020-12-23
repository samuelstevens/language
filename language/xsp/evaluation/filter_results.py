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
"""Filters the results of running evaluation to include only clean examples.

It filters the following:

- Examples where the resulting table of the gold query is the empty table
- Examples where the resulting table of the gold query is [0], if the gold
  query returns a count
- Examples where strings or numerical values in the gold query are not present
  in the input utterance
- Examples where multiple columns are selected in the resulting table

Usage:
    The argument is the *_eval.txt file that is generated by
    official_evaluation.py.

"""
from __future__ import absolute_import, division, print_function

import sys

with open(sys.argv[1]) as infile:
    examples = infile.read().split("\n\n")

num_exec_correct = 0
num_filtered = 0
for example in examples[:-1]:

    # Filter out examples with empty gold tables.
    if "Gold table was EMPTY!" in example:
        continue

    # Filter out examples with a result of [0] and that require a count.
    if example.endswith("Gold table:\n\t[0]") and (
        "gold query:\n\tselect count" in example.lower()
        or "gold query:\n\tselect distinct count" in example.lower()
    ):
        continue

    # Filter out examples that require copying values that can't be copied.
    prev_value = ""
    example_lines = example.split("\n")
    last_quote = ""
    gold_query_idx = example_lines.index("Gold query:") + 1
    utterance = example_lines[1]
    copiable = True
    in_equality = False
    numerical_value = ""
    handled_prefix = False
    too_many_selects = False
    gold_query = example_lines[gold_query_idx].strip()

    for i, char in enumerate(gold_query):
        # Check that it's only selecting a single table at the top
        if (
            not handled_prefix
            and i - 4 >= 0
            and gold_query[i - 4 : i].lower() == "from"
        ):
            handled_prefix = True
            if gold_query[:i].count(",") > 0:
                too_many_selects = True

        if char == last_quote:
            last_quote = ""

            prev_value = prev_value.replace("%", "")

            if prev_value not in utterance:
                copiable = False

            prev_value = ""

        elif last_quote:
            prev_value += char
        elif char in {'"', "'"}:
            last_quote = char

        if char in {"=", ">", "<"}:
            in_equality = True

        if in_equality:
            if char.isdigit() or char == ".":
                if numerical_value or (not prev_value and gold_query[i - 1] == " "):
                    numerical_value += char

            if char == " " and numerical_value:
                in_equality = False

                if numerical_value not in utterance and numerical_value not in {
                    "0",
                    "1",
                }:
                    # Allow generation of 0, 1 for compositionality purposes.
                    copiable = False
                numerical_value = ""

    if not copiable or too_many_selects:
        continue

    num_filtered += 1

    if "Execution was correct? True" in example:
        num_exec_correct += 1

    print(example + "\n")

print(
    "Performance on subset: "
    + "{0:.2f}".format(100.0 * num_exec_correct / num_filtered),
    num_exec_correct,
    num_filtered,
)
print("Filtered from %d to %d examples" % (len(examples) - 1, num_filtered))
