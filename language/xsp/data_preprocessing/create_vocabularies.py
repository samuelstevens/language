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
"""Creates output vocabulary for NLToSQLExamples."""
import json
import os
from typing import Set

import tensorflow.compat.v1.gfile as gfile
from absl import app, flags

from language.xsp.data_preprocessing.nl_to_sql_example import NLToSQLExample

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "The directory containing the input JSON files.")

flags.DEFINE_list("input_filenames", None, "Which files to extract vocabulary from.")

flags.DEFINE_string("output_path", "", "Location to save the output vocabulary.")


def get_symbol(line: str) -> Set[str]:
    gold_query = NLToSQLExample.from_json(json.loads(line)).gold_sql_query
    return {token.symbol for token in gold_query.actions if token.symbol}


def years() -> Set[str]:
    return {str(year) for year in range(1990, 2021)}


def main(unused_argv):
    # Load the examples
    vocabulary = set()
    valid_filenames = [filename for filename in FLAGS.input_filenames if filename]
    for filename in valid_filenames:
        with open(os.path.join(FLAGS.data_dir, filename)) as infile:
            for line in infile:
                if not line:
                    continue

                symbols = get_symbol(line)
                new_symbols = [symbol for symbol in symbols if symbol not in vocabulary]
                if new_symbols:
                    print(new_symbols)
                for symbol in symbols:
                    vocabulary.add(symbol)

    print("Writing vocabulary of size %d to %s" % (len(vocabulary), FLAGS.output_path))
    with gfile.Open(FLAGS.output_path, "w") as ofile:
        ofile.write("\n".join(list(vocabulary)))


if __name__ == "__main__":
    app.run(main)
