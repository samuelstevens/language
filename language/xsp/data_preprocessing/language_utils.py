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
"""Utilities for processing and storing the natural langauge input."""

from __future__ import absolute_import, division, print_function

import dataclasses
from typing import Collection, List, Optional, Set, Tuple


@dataclasses.dataclass
class Wordpiece:
    """Contains wordpiece information as a substring of an utterance."""

    wordpiece: str
    tokenized_index: int
    span_start_index: int
    span_end_index: int
    matches_to_schema: bool

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(dictionary) -> "Wordpiece":
        return Wordpiece(**dictionary)


def get_wordpieces(
    sequence: str, tokenizer, schema_entities: Optional[Collection[str]] = None
) -> Tuple[List[Wordpiece], Set[str]]:
    """Sets the wordpieces for a NLToSQLExample."""
    # First, it finds exact-string alignment between schema entities and the utterance.
    aligned_entities: Set[str] = set()
    aligned_chars = [False for _ in range(len(sequence))]
    if schema_entities:
        for schema_entity in sorted(schema_entities, key=len, reverse=True):
            if schema_entity in sequence.lower():
                aligned_entities.add(schema_entity)
                start_idx = sequence.lower().index(schema_entity)

                for i in range(start_idx, start_idx + len(schema_entity)):
                    aligned_chars[i] = True

    # Get the spans for the wordpieces
    wordpieces: List[str] = tokenizer.tokenize(sequence)

    original_seq_index = 0
    wordpieces_with_spans = list()
    for i, wordpiece in enumerate(wordpieces):
        search_wordpiece = wordpiece
        if wordpiece.startswith("#"):
            search_wordpiece = wordpiece[2:]

        # It will be a substring of the lowered original sequence.
        found = True
        while (
            sequence.lower()[
                original_seq_index : original_seq_index + len(search_wordpiece)
            ]
            != search_wordpiece
        ):
            original_seq_index += 1

            if original_seq_index + len(search_wordpiece) > len(sequence):
                found = False
                break

        span_start = original_seq_index
        span_end = original_seq_index + len(search_wordpiece)  # Not inclusive!

        if not found:
            raise ValueError(
                "Span not found! \nWordpiece: " + wordpiece + "\nSequence: " + sequence
            )

        if sequence.lower()[span_start:span_end] != search_wordpiece:
            raise ValueError(
                "Found span did not match!\nWordpiece: "
                + wordpiece
                + "\nSpan: "
                + sequence.lower()[span_start:span_end]
            )

        # See if the span start/end align at all with the aligned chars
        aligned = False
        for j in range(span_start, span_end):
            if aligned_chars[j]:
                aligned = True
                break

        wordpieces_with_spans.append(
            Wordpiece(wordpiece, i, span_start, span_end, aligned)
        )

    return wordpieces_with_spans, aligned_entities
