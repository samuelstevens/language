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
"""
Official evaluation script for natural language to SQL datasets.

Arguments:
  predictions_filepath (str): Path to a predictions file (in JSON format).
  output_filepath (str): Path to the file where the result of execution is
    saved.
  cache_filepath (str): Path to a JSON file containing a mapping from gold SQL
    queries to cached resulting tables.  Should be ran locally. All filepaths
    above should refer to the local filesystem.
"""
from __future__ import absolute_import, division, print_function

import argparse
import enum
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import timeout_decorator
from prelude import SumType, never
from tqdm import tqdm

from language.xsp.model.inference import ExecutionInstructions, Prediction

# Maximum allowable timeout for executing predicted and gold queries.
TIMEOUT = 60

# Maximum number of candidates we should consider
MAX_CANDIDATE = 20

# These are substrings of exceptions from sqlite3 that indicate certain classes
# of schema and syntax errors.
SCHEMA_INCOHERENCE_STRINGS = {
    "no such table",
    "no such column",
    "ambiguous column name",
}
SYNTAX_INCORRECTNESS_STRINGS = {
    "bad syntax",
    "unrecognized token",
    "incomplete input",
    "misuse of aggregate",
    "left and right",
    "wrong number of arguments",
    "sub-select returns",
    "1st order by term does not match any column",
    "no such function",
    "clause is required before",
    "incorrect number of bindings",
    "datatype mismatch",
    "syntax error",
    "not allowed in the group by",
    "row value misused",
    "group by term out of range",
}


class ExecutionError(SumType):
    schema = enum.auto()
    syntax = enum.auto()
    timeout = enum.auto()
    unknown = enum.auto()


class ExecutionResult(NamedTuple):
    utterance: str
    predicted_queries: List[str]
    gold_query: str

    best_prediction: str

    column_f1: float
    table_f1: float

    gold_results: Any
    pred_results: Any

    equivalent: bool
    position: int

    error_cause: Optional[ExecutionError]
    error_message: Optional[str]

    gold_error_cause: Optional[ExecutionError]
    gold_error_message: Optional[str]


class Metrics(NamedTuple):
    column_f1: float
    table_f1: float
    execution_f1: float

    precision: float
    recall: float

    schema_errors: float
    syntax_errors: float
    conversion_errors: float
    timeouts: float
    gold_error: float

    num_empty_gold: int
    num_empty_pred: int

    string_same: float
    exec_results_same: float


class ExceptionStrParseError(Exception):
    pass


T = TypeVar("T")


def identity(x: T) -> T:
    return x


def _exception_str_to_cause(exception_str: str) -> ExecutionError:
    error_cause = None

    for substring in SCHEMA_INCOHERENCE_STRINGS:
        if substring in exception_str.lower():
            error_cause = ExecutionError.schema
            break

    if not error_cause:
        for substring in SYNTAX_INCORRECTNESS_STRINGS:
            if substring in exception_str.lower():
                error_cause = ExecutionError.syntax
                break

    if not error_cause and "timeout" in exception_str:
        error_cause = ExecutionError.timeout

    if not error_cause:
        raise ExceptionStrParseError(exception_str)

    return error_cause


def normalize_sql_str(string):
    """Normalizes the format of a SQL string for string comparison."""
    string = string.lower()
    while "  " in string:
        string = string.replace("  ", " ")
    string = string.strip()
    string = string.replace("( ", "(").replace(" )", ")")
    string = string.replace(" ;", ";")
    string = string.replace('"', "'")

    if ";" not in string:
        string += ";"
    return string


def string_acc(s1, s2):
    """Computes string accuracy between two SQL queries."""
    return normalize_sql_str(s1) == normalize_sql_str(s2)


def result_table_to_string(table):
    """Converts a resulting SQL table to a human-readable string."""
    string_val = (
        "\t" + "\n\t".join([str(row) for row in table[: min(len(table), 5)]]) + "\n"
    )
    if len(table) > 5:
        string_val += "... and %d more rows.\n" % (len(table) - 5)
    return string_val


def try_executing_query(
    prediction: str, cursor, case_sensitive=True, verbose=False
) -> Tuple[Any, Any, Any]:
    """Attempts to execute a SQL query against a database given a cursor."""
    exception_str = None

    prediction_str = prediction[:]
    prediction_str = prediction_str.replace(";", "").strip()

    st = time.time()
    try:
        if not case_sensitive:
            new_prediction = ""
            last_quote = ""
            for char in prediction:
                new_prediction += char
                if char in {'"', "'"} and not last_quote:
                    last_quote = char
                elif char == last_quote:
                    last_quote = ""
                    new_prediction += " COLLATE NOCASE"
            prediction = new_prediction

            if verbose:
                print("Executing case-insensitive query:")
                print(new_prediction)
        pred_results = timeout_execute(cursor, prediction)
    except timeout_decorator.timeout_decorator.TimeoutError:
        print("Timed out!")
        if verbose:
            print(prediction)
        pred_results = []
        exception_str = "timeout"
    except (
        sqlite3.Warning,
        sqlite3.Error,
        sqlite3.DatabaseError,
        sqlite3.IntegrityError,
        sqlite3.ProgrammingError,
        sqlite3.OperationalError,
        sqlite3.NotSupportedError,
    ) as e:
        exception_str = str(e).lower()
        pred_results = []
    execution_time = time.time() - st

    return pred_results, exception_str, execution_time


@timeout_decorator.timeout(seconds=TIMEOUT, use_signals=False)
def timeout_execute(cursor, prediction):
    cursor.execute(prediction)
    pred_results = cursor.fetchall()
    pred_results = [list(result) for result in pred_results]
    return pred_results


def find_used_entities_in_string(query, columns, tables):
    """Heuristically finds schema entities included in a SQL query."""
    used_columns = set()
    used_tables = set()

    nopunct_query = query.replace(".", " ").replace("(", " ").replace(")", " ")

    for token in nopunct_query.split(" "):
        if token.lower() in columns:
            used_columns.add(token.lower())
        if token.lower() in tables:
            used_tables.add(token.lower())
    return used_columns, used_tables


def compute_f1(precision, recall):
    if precision + recall > 0.0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0


def compute_set_f1(pred_set, gold_set):
    """Computes F1 of items given two sets of items."""
    prec = 1.0
    if pred_set:
        prec = float(len(pred_set & gold_set)) / len(pred_set)

    rec = 1.0
    if gold_set:
        rec = float(len(pred_set & gold_set)) / len(gold_set)
    return compute_f1(prec, rec)


def col_tab_f1(schema, gold_query, predicted_query):
    """Computes the F1 of tables and columns mentioned in the two queries."""

    # Get the schema entities.
    db_columns = set()
    db_tables = set()
    for name, cols in schema.items():
        for col in cols:
            db_columns.add(col["field name"].lower())
        db_tables.add(name.lower())

    # Heuristically find the entities used in the gold and predicted queries.
    pred_columns, pred_tables = find_used_entities_in_string(
        predicted_query, db_columns, db_tables
    )
    gold_columns, gold_tables = find_used_entities_in_string(
        gold_query, db_columns, db_tables
    )

    # Compute and return column and table F1.
    return (
        compute_set_f1(pred_columns, gold_columns),
        compute_set_f1(pred_tables, gold_tables),
    )


def execute_prediction(
    prediction: Prediction, empty_table_cursor, cursor, case_sensitive, verbose
):
    """
    Executes a single example's prediction(s).

    If more than one prediction is available, the most likely executable
    prediction is used as the "official" prediction.

    Args:
        prediction: A dictionary containing information for a single example's
    prediction.
        empty_table_cursor: The cursor to a database containing no records, to be
    used only to determine whether a query is executable in the database.
        cursor: The sqlite3 database cursor to execute queries on.
        case_sensitive: Boolean indicating whether the execution should be case
    sensitive with respect to string values.
        verbose: Whether to print details about what queries are being executed.

    Returns:
        Tuple containing the highest-ranked executable query, the resulting table,
    and any exception string associated with executing this query.
  """

    # Go through predictions in order of probability and test their executability
    # until you get an executable prediction. If you don't find one, just
    # "predict" the most probable one.
    paired_preds_and_scores = zip(prediction["predictions"], prediction["scores"])
    sorted_by_scores = sorted(paired_preds_and_scores, key=lambda x: x[1], reverse=True)

    best_prediction = None
    pred_results = None
    exception_str = None
    execution_time = 0

    if len(sorted_by_scores) > MAX_CANDIDATE:
        sorted_by_scores = sorted_by_scores[:MAX_CANDIDATE]

    for i, (pred, _) in enumerate(sorted_by_scores):
        # Try predicting
        if verbose:
            print("Trying to execute query:\n\t" + pred)
            print("... on empty database")
        temp_exception_str = try_executing_query(
            pred, empty_table_cursor, case_sensitive, verbose
        )[1]

        if temp_exception_str:
            if i == 0:
                # By default, set the prediction to the first (highest-scoring)
                # one.
                best_prediction = pred

                # Get the actual results
                if verbose:
                    print("... on actual database")
                pred_results, exception_str, execution_time = try_executing_query(
                    pred, cursor, case_sensitive, verbose
                )
            if exception_str == "timeout":
                # Technically, this query didn't have a syntax problem, so
                # continue and set this as the best prediction.
                best_prediction = pred

                if verbose:
                    print("... on actual database")
                pred_results, exception_str, execution_time = try_executing_query(
                    pred, cursor, case_sensitive, verbose
                )
                break
        else:
            best_prediction = pred
            exception_str = None

            if verbose:
                print("No exception... on actual database")
            pred_results, _, execution_time = try_executing_query(
                pred, cursor, case_sensitive, verbose
            )
            break

    return best_prediction, pred_results, exception_str, execution_time


def write_results(results: List[ExecutionResult], ofile) -> None:
    metrics = aggregate_metrics(results)

    for i, result in enumerate(results):
        ofile.write("Example #" + str(i) + "\n")

        ofile.write(result.utterance.strip() + "\n")
        ofile.write("Predicted query:\n")
        if result.best_prediction:
            ofile.write("\t" + result.best_prediction.strip() + "\n")
        else:
            ofile.write(f"ERROR: Cannot write prediction {result.best_prediction}\n")

        if result.error_message:
            ofile.write(result.error_message + "\n")

        ofile.write("Gold query:\n")
        ofile.write("\t" + result.gold_query.strip() + "\n")

        # Add some debugging information about the tables, and compute the precisions.
        if result.pred_results:
            if not result.equivalent:
                ofile.write("Predicted table:\n")
                ofile.write(result_table_to_string(result.pred_results))
        elif result.best_prediction is None or not result.equivalent:
            ofile.write("Predicted table was EMPTY!\n")

        if result.gold_results:
            ofile.write("Gold table:\n")
            ofile.write(result_table_to_string(result.gold_results))
        else:
            ofile.write("Gold table was EMPTY!\n")

        ofile.write(f"Column F1: {result.column_f1}\n")
        ofile.write(f"Column F1: {result.table_f1}\n")

        ofile.write("Execution was correct? " + str(result.equivalent) + "\n")

    ofile.write("String accuracy: " + "{0:.2f}".format(metrics.string_same) + "\n")
    ofile.write("Accuracy: " + "{0:.2f}".format(metrics.exec_results_same) + "\n")
    ofile.write(
        "Precision: "
        + "{0:.2f}".format(100.0 * metrics.precision)
        + " ; "
        + str(metrics.num_empty_pred)
        + " nonempty predicted tables"
        + "\n"
    )
    ofile.write(
        "Recall: "
        + "{0:.2f}".format(100.0 * metrics.recall)
        + " ; "
        + str(metrics.num_empty_gold)
        + " nonempty gold tables"
        + "\n"
    )
    ofile.write(
        "Execution F1: " + "{0:.2f}".format(100.0 * metrics.execution_f1) + "\n"
    )

    ofile.write(f"Timeout: {metrics.timeouts:.2f}%\n")
    ofile.write(f"Gold did not execute: {metrics.gold_error:.2f}%\n")

    ofile.write(
        "Average column F1: " + "{0:.2f}%".format(100.0 * metrics.column_f1) + "\n"
    )
    ofile.write("Average table F1: " + "{0:.2f}%".format(metrics.table_f1) + "\n")

    ofile.write(f"Schema errors: {metrics.schema_errors:.2f}%\n")
    ofile.write(f"Syntax errors: {metrics.syntax_errors:.2f}%\n")
    ofile.write(f"Conversion errors: {metrics.conversion_errors:.2f}%\n")

    ofile.write("\n")
    ofile.flush()


def aggregate_metrics(results: List[ExecutionResult]) -> Metrics:
    assert len(results) > 0, "Must have at least one result!"
    exec_results_same = []
    string_same = []

    precisions = []
    recalls: List[int] = []

    column_f1s = []
    table_f1s = []

    conversion_errors = 0.0

    schema_errors = 0.0
    syntax_errors = 0.0
    timeouts = 0.0

    gold_error = 0.0

    for result in results:

        if result.best_prediction:
            string_same.append(string_acc(result.gold_query, result.best_prediction))
            column_f1s.append(result.column_f1)
            table_f1s.append(result.table_f1)
        else:
            string_same.append(0.0)
            column_f1s.append(0.0)
            table_f1s.append(0.0)

            conversion_errors += 1

        if result.pred_results:
            precisions.append(int(result.equivalent))
        if result.gold_results:
            recalls.append(int(result.equivalent))

        if result.error_cause:
            if result.error_cause is ExecutionError.schema:
                schema_errors += 1
            elif result.error_cause is ExecutionError.syntax:
                syntax_errors += 1
            elif result.error_cause is ExecutionError.timeout:
                timeouts += 1
            elif result.error_cause is ExecutionError.unknown:
                pass
            else:
                never(result.error_cause)

        if result.gold_error_cause:
            gold_error += 1

        exec_results_same.append(int(result.equivalent))

    precision = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls) if recalls else 0.0

    return Metrics(
        execution_f1=compute_f1(precision, recall),
        num_empty_pred=len(precisions),
        num_empty_gold=len(recalls),
        timeouts=timeouts / len(results) * 100,
        gold_error=gold_error / len(results) * 100,
        schema_errors=schema_errors / len(results) * 100,
        syntax_errors=syntax_errors / len(results) * 100,
        conversion_errors=conversion_errors / len(results) * 100,
        string_same=np.mean(string_same) * 100,
        exec_results_same=np.mean(exec_results_same) * 100,
        column_f1=np.mean(column_f1s),
        table_f1=np.mean(table_f1s),
        precision=precision,
        recall=recall,
    )


def execute_predictions(
    instructions: List[ExecutionInstructions],
    cache_dict: Dict[str, Any],
    case_sensitive: bool,
    verbose: bool,
    update_cache: bool,
) -> Tuple[List[ExecutionResult], Dict[str, Any]]:
    """
    Executes predicted/gold queries

    Writes results to ofile.

    Args:
        instructions: A list of dictionaries defining the executions to make, containing predictions made by a model.
        cache_dict: A dictionary mapping from gold queries to the resulting tables.
        case_sensitive: A Boolean indicating whether execution of queries should be
        case sensitive with respect to strings.
        verbose: Whether to print detailed information about evaluation (e.g., for debugging).
        update_cache: Whether to execute and cache gold queries.
    Returns:
        A list of execution results and the cache dict.
    """
    assert cache_dict is not None, "Must provide a cache dict, even if empty"

    results = []

    iterator = tqdm
    if verbose:
        # Don't use TQDM if verbose: it might mess up the verbose messages
        iterator = identity  # type: ignore

    for i, instruction in iterator(enumerate(instructions)):
        error_cause = None
        error_message = None
        gold_error_cause = None
        gold_error_message = None

        # Attempt to connect to the database for executing.
        try:
            conn = sqlite3.connect(instruction["database_path"])
            conn.text_factory = lambda x: str(x, encoding="utf-8", errors="ignore")
        except sqlite3.OperationalError as e:
            print(instruction["database_path"])
            raise e

        try:
            empty_conn = sqlite3.connect(instruction["empty_database_path"])
            empty_conn.text_factory = lambda x: str(
                x, encoding="utf-8", errors="ignore"
            )
        except sqlite3.OperationalError as e:
            print(instruction["empty_database_path"])
            raise e

        empty_cursor = empty_conn.cursor()
        cursor = conn.cursor()

        if verbose:
            print(
                "Finding the highest-rated prediction for utterance:\n\t"
                + instruction["prediction"]["utterance"].strip()
            )

        (
            best_prediction,
            pred_results,
            exception_str,
            execution_time,
        ) = execute_prediction(
            instruction["prediction"], empty_cursor, cursor, case_sensitive, verbose
        )

        # If it didn't execute correctly, check why.
        if exception_str:
            try:
                error_message = exception_str
                if verbose:
                    print(exception_str)
                error_cause = _exception_str_to_cause(exception_str)
                if error_cause is ExecutionError.timeout:
                    error_message = "Execution (predicted) took too long"
            except ExceptionStrParseError:
                # If the error type hasn't been identified, exit and report it.
                print("SQL error:")
                print(exception_str)
                print("For SQL string:")
                print(best_prediction)
                error_case = ExecutionError.unknown

            # Predicted table should be empty for all of these cases.
            pred_results = []

        # Compare to gold and update metrics
        gold_query = instruction["gold"]

        # Get the gold results
        if gold_query not in cache_dict:
            if instruction["prediction"]["utterance"].strip() not in cache_dict:
                if update_cache:
                    if verbose:
                        print("Trying to execute the gold query:\n\t" + gold_query)
                    (
                        gold_results,
                        gold_exception_str,
                        execution_time,
                    ) = try_executing_query(gold_query, cursor, case_sensitive, verbose)
                    if gold_exception_str:
                        gold_error_message = gold_exception_str

                        try:
                            gold_error_cause = _exception_str_to_cause(
                                gold_exception_str
                            )
                        except ExceptionStrParseError:
                            pass

                        if verbose:
                            print(
                                "Error executing gold query:\n\t"
                                + gold_query
                                + "\n\n\t"
                                + gold_exception_str
                            )

                    cache_dict[gold_query] = gold_results
                else:
                    print(gold_query)
                    print(instruction["prediction"]["utterance"].strip())
                    raise ValueError("Cache miss!")

        gold_results = cache_dict[gold_query]

        col_f1, tab_f1 = 0, 0

        if best_prediction:
            col_f1, tab_f1 = col_tab_f1(
                instruction["schema"], gold_query, best_prediction
            )

            if "order by" in gold_query:
                results_equivalent = pred_results == gold_results
            else:
                pred_set = set()
                gold_set = set()
                for pred in pred_results:
                    if isinstance(pred, list):
                        pred_set.add(" ".join([str(item) for item in pred]))
                    else:
                        pred_set.add(pred)
                for gold in gold_results:
                    if isinstance(gold, list):
                        gold_set.add(" ".join([str(item) for item in gold]))
                    else:
                        gold_set.add(gold)

                results_equivalent = pred_set == gold_set

        else:
            # Only consider correct if the gold table was empty.
            results_equivalent = gold_results == []

        conn.close()
        empty_conn.close()

        result = ExecutionResult(
            predicted_queries=instruction["prediction"]["predictions"],
            best_prediction=best_prediction,
            pred_results=pred_results,
            gold_results=gold_results,
            equivalent=results_equivalent,
            utterance=instruction["prediction"]["utterance"],
            position=i,
            column_f1=col_f1,
            table_f1=tab_f1,
            gold_query=instruction["gold"],
            error_cause=error_cause,
            error_message=error_message,
            gold_error_cause=gold_error_cause,
            gold_error_message=gold_error_message,
        )
        results.append(result)

    return results, cache_dict


def main(
    predictions_filepath: str,
    output_filepath: str,
    cache_filepath: str,
    verbose: bool,
    update_cache: bool,
):
    assert predictions_filepath.endswith(
        ".json"
    ), f"Expected .json file, got {predictions_filepath}"
    with open(predictions_filepath) as infile:
        # Load the predictions filepath.
        predictions = json.load(infile)
    print("Loaded %d predictions." % len(predictions))

    # Load or create the cache dictionary mapping from gold queries to resulting
    # tables.
    cache_dict = None

    print("cache path: " + cache_filepath)

    basefilename = os.path.basename(predictions_filepath).lower()

    cache_dict = {}
    if os.path.exists(cache_filepath):
        print("Loading cache from %s" % cache_filepath)
        with open(cache_filepath) as infile:
            cache_dict = json.load(infile)
        print("Loaded %d cached queries" % len(cache_dict))

    results, cache_dict = execute_predictions(
        predictions, cache_dict, "scholar" not in basefilename, verbose, update_cache,
    )

    # Create the text file that results will be written to.
    with open(output_filepath, "w") as ofile:
        write_results(results, ofile)

    if "spider" not in basefilename:
        try:
            cache_str = json.dumps(cache_dict)
            with open(cache_filepath, "w") as ofile:
                ofile.write(cache_str)
        except UnicodeDecodeError as e:
            print("Could not save the cache dict. Exception:")
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_filepath",
        type=str,
        help="Where the predictions JSON file is located.",
    )
    parser.add_argument(
        "--output_filepath", type=str, help="Where to write the results."
    )
    parser.add_argument(
        "--cache_filepath", type=str, help="A cache of the gold tables.", default=""
    )
    parser.add_argument(
        "--verbose", type=bool, help="If set to True, evaluation will be verbose."
    )
    parser.add_argument(
        "--update_cache",
        type=bool,
        help="If set to True, will execute and cache gold queries.",
    )
    args = parser.parse_args()

    main(
        args.predictions_filepath,
        args.output_filepath,
        args.cache_filepath,
        args.verbose,
        args.update_cache,
    )
