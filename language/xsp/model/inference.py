import os
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence

import apache_beam
import sqlparse
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from typing_extensions import TypedDict

from language.xsp.data_preprocessing.michigan_preprocessing import (
    get_nl_sql_pairs,
    read_schema,
)
from language.xsp.data_preprocessing.spider_preprocessing import (
    load_spider_examples,
    load_spider_tables,
    preprocess_sql,
)
from language.xsp.model import input_pipeline, model_builder
from language.xsp.model.model_config import ModelConfig

QUOTES = {"'", '"'}


class Config(NamedTuple):
    dataset_name: str
    splits: List[str]
    output_vocab_filepath: str
    clean_output_vocab_filepath: Optional[str]
    beam_size: int
    using_abstract_sql: bool
    database_directory: str
    empty_database_directory: str
    original_data_directory: str
    model_config: ModelConfig


class Prediction(TypedDict):
    utterance: str
    predictions: List[str]
    scores: List[float]


class ExecutionInstructions(TypedDict):
    prediction: Prediction
    gold: Any
    database_path: str
    empty_database_path: str
    schema: Any


def _action_id_to_table_name_map(segment_ids, copy_strings):
    """Returns a map of action_ids to table names for columns."""
    current_segment_id = 0
    table_name = None
    # The first copy_string for a new segment_id is the table name.
    # Following copy_string are for columns belonging to that table.
    # TODO(petershaw): This is really hacky! We should provide a better channel
    # for passing this information to the output during inference.
    action_id_to_table_name_map = {}
    for action_id, (segment_id, copy_string) in enumerate(
        zip(segment_ids, copy_strings)
    ):
        if segment_id > current_segment_id:
            current_segment_id = segment_id
            table_name = copy_string
        elif table_name:
            action_id_to_table_name_map[action_id] = table_name
    return action_id_to_table_name_map


def clean_predicted_sequence(
    action_ids,
    action_types,
    scores,
    vocab,
    copy_strings,
    segment_ids,
    restore_preds_from_asql,
    clean_vocab=None,
):
    """Cleans a set of predicted SQL queries."""
    copy_strings = [tok.decode("utf-8") for tok in copy_strings]
    action_id_to_table_name_map = None
    if restore_preds_from_asql:
        action_id_to_table_name_map = _action_id_to_table_name_map(
            segment_ids, copy_strings
        )
    string_seq = []
    for action_type, action_id in zip(action_types, action_ids):
        if action_type == 1:
            # Generate symbol from output vocabulary.
            pred_idx = action_id
            if pred_idx == 1:
                # END symbol.
                break
            # Indices into vocab are offset by 3.
            symbol = vocab[pred_idx - 3]
            if clean_vocab:
                if symbol in clean_vocab:
                    string_seq.append(symbol)
            else:
                string_seq.append(symbol)
        else:
            # Copy symbol from input.
            symbol = copy_strings[action_id]
            if restore_preds_from_asql and action_id in action_id_to_table_name_map:
                # For abstract SQL, need to fully qualify column names by prepending
                # the table name.
                table_name = action_id_to_table_name_map[action_id]
                symbol = "%s.%s" % (table_name, symbol)
            string_seq.append(symbol)

    sql = ""
    in_quote = False
    for i, token in enumerate(string_seq):
        if not in_quote and token not in QUOTES:
            sql += " "
        if token in QUOTES:
            in_quote = not in_quote
        sql += token
        if (
            not in_quote
            or token not in QUOTES
            and i < len(string_seq) - 1
            and string_seq[i + 1] not in QUOTES
        ):
            sql += " "

    sql = sql.replace("  ", " ")
    sql = sql.replace(" ##", "")
    sql = sqlparse.format(sql, reident=True, keyword_case="upper")
    sql = sql.replace("\n", " ")
    sql = sql.replace("( ", "(")
    sql = sql.replace(" )", ")")
    sql = sql.replace(" . ", ".")
    sql = sql.replace(" %", "%")
    sql = sql.replace("% ", "%")

    for func in ("count", "min", "max", "avg", "sum"):
        sql = sql.replace("%s (" % func.upper(), "%s(" % func)

    for i in range(1, 11):
        sql = sql.replace("t%d" % i, "T%d" % i)

    sql = sql.strip()

    return sql, float(scores)


def setup_graph(
    model_config: ModelConfig,
    output_vocab_filepath: str,
    clean_output_vocab_filepath: Optional[str],
    beam_size: int,
):
    """Sets up the Tenorflow graph for inference."""
    # Set up the model for inference
    # model_config = load_config(os.path.join(config_filepath))
    placeholder, features, labels = input_pipeline.create_placeholder_inputs(
        model_config.model_parameters.use_segment_ids,
        model_config.model_parameters.use_foreign_key_features,
        model_config.model_parameters.use_alignment_features,
    )

    model_fn = model_builder.build_model_fn(
        model_config,
        output_vocab_filepath,
        clean_output_vocab_filepath,
        beam_size=beam_size,
    )
    mode = tf.estimator.ModeKeys.PREDICT
    predictions = model_fn(features, labels, mode).predictions

    saver = tf.train.Saver()

    return saver, placeholder, predictions


def _get_copy_strings(tf_example):
    copy_strings = []
    for token in tf_example.feature_lists.feature_list["copy_strings"].feature:
        if len(token.bytes_list.value) != 1:
            raise ValueError("Invalid copy_strings in example: %s" % tf_example)
        copy_strings.append(token.bytes_list.value[0])
    if not copy_strings:
        raise ValueError("Missing copy_strings in example: %s" % tf_example)
    return copy_strings


def _get_segment_ids(tf_example):
    segment_ids = []
    for token in tf_example.feature_lists.feature_list["segment_ids"].feature:
        if len(token.int64_list.value) != 1:
            raise ValueError("Invalid segment_ids in example: %s" % tf_example)
        segment_ids.append(token.int64_list.value[0])
    if not segment_ids:
        raise ValueError("Missing segment_ids in example: %s" % tf_example)
    return segment_ids


def get_prediction(
    placeholder,
    tf_example,
    sess,
    outputs,
    vocab,
    beam_size,
    restore_preds_from_asql=False,
    clean_vocab=None,
):
    """Gets predicted outputs for a specific input to the model."""
    copy_strings = _get_copy_strings(tf_example)
    segment_ids = _get_segment_ids(tf_example)

    feed_dict = {placeholder: tf_example.SerializeToString()}
    output_vals = sess.run(outputs, feed_dict=feed_dict)

    predictions = []
    scores = []
    for index in range(beam_size):
        prediction, score = clean_predicted_sequence(
            output_vals["predicted_action_ids"][index],
            output_vals["predicted_action_types"][index],
            output_vals["scores"][index],
            vocab,
            copy_strings,
            segment_ids,
            restore_preds_from_asql=restore_preds_from_asql,
            clean_vocab=clean_vocab,
        )
        predictions.append(prediction)
        scores.append(score)
    return predictions, scores


class RunInferenceDoFn(apache_beam.DoFn):
    """DoFn for running inference on an example given model parameters."""

    class _GraphState(object):
        """This class caches the tf session/graph across process instances."""

        def __init__(
            self,
            checkpoint,
            model_config: ModelConfig,
            output_vocab_filepath: str,
            clean_output_vocab_filepath: Optional[str],
            beam_size: int,
        ):
            # Set up the graph and load the checkpoint
            saver, placeholder, outputs = setup_graph(
                model_config,
                output_vocab_filepath,
                clean_output_vocab_filepath,
                beam_size,
            )

            self.placeholder = placeholder
            self.sess = tf.Session()
            self.saver = saver
            self.outputs = outputs

            print("Restoring checkpoint: {}".format(checkpoint))
            self.saver.restore(self.sess, checkpoint)

    _vocab: List[str]
    _clean_vocab: Optional[List[str]]
    _graph_state: _GraphState

    def __init__(self, checkpoint, config: Config):
        with tf.gfile.Open(config.output_vocab_filepath) as infile:
            self._vocab = [line.strip() for line in infile]

        if config.clean_output_vocab_filepath:
            with tf.gfile.Open(config.clean_output_vocab_filepath) as infile:
                self._clean_vocab = [line.strip() for line in infile]
        else:
            self._clean_vocab = None

        self._graph_state = self._GraphState(
            checkpoint,
            config.model_config,
            config.output_vocab_filepath,
            config.clean_output_vocab_filepath,
            config.beam_size,
        )

    def non_parallel_process(
        self, example, beam_size: int, using_abstract_sql: bool
    ) -> Prediction:
        # Runs inference for the example.
        predicted_sequences, scores = get_prediction(
            self._graph_state.placeholder,
            example,
            self._graph_state.sess,
            self._graph_state.outputs,
            self._vocab,
            beam_size,
            using_abstract_sql,
            self._clean_vocab,
        )

        return {
            "utterance": dict(example.context.feature)["key"]
            .bytes_list.value[0]
            .decode("utf-8"),
            "predictions": predicted_sequences,
            "scores": scores,
        }

    def process(
        self, example, beam_size: int, using_abstract_sql: bool
    ) -> Iterator[Prediction]:
        if isinstance(example, str):
            raise ValueError("Example is a str! %r" % example)
        yield self.non_parallel_process(example, beam_size, using_abstract_sql)


def load_tf_examples(input_path: str) -> List[tf.train.SequenceExample]:
    """
    Loads tf.train.Examples from the TFRecord file format.

    Args:
        input_path: The TFRecord file with the input examples.

    Returns: a list of examples
    """

    assert input_path.endswith(
        ".tfrecords"
    ), f"input_path should be a .tfrecords file; got {input_path}."

    # Load and process the TFRecords. First, inference is ran on these records
    # without looking at the gold query.
    examples = []
    record_iterator = tf.python_io.tf_record_iterator(path=input_path)
    for record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(record)
        examples.append(example)

    return examples


def inference(
    examples: List[tf.train.Example], checkpoint: str, config: Config,
) -> List[Prediction]:
    """
    Runs inference locally.

    Args:
        examples: the tf.train.Examples loaded from disk (use load_tf_examples)
        checkpoint: Filepath to the model save checkpoint.
        config: Config for inference.
    Returns:
        A list of model predictions and corresponding scores.
    """

    fn = RunInferenceDoFn(checkpoint, config)

    # array[:None] is the same as array[:] (doesn't change size)
    return [
        fn.non_parallel_process(example, config.beam_size, config.using_abstract_sql)
        for example in tqdm(examples)
    ]


def load_schema_obj(dataset_name: str, data_dir: str) -> Dict[Any, Any]:
    if dataset_name.lower() == "spider":
        return load_spider_tables(os.path.join(data_dir, "tables.json"))
    elif dataset_name.lower() == "wikisql":
        raise ValueError("WikiSQL inference is not supported yet")
    else:
        schema_csv = os.path.join(data_dir, dataset_name + "_schema.csv",)
        return read_schema(schema_csv)


def match_and_save(
    config: Config, predictions: Sequence[Prediction], schema_obj: Dict[Any, Any]
) -> List[ExecutionInstructions]:
    """
    Loads an original dataset and matches with a predictions file.
    """

    prediction_dict: Dict[str, Any] = {}
    for prediction in predictions:
        prediction_dict[prediction["utterance"]] = prediction

    # Load the data for this particular dataset (for look up)
    # `examples` is a list of dictionaries for each example, containing a TFRecord
    #  object, nl, sql, and a db_id (if running inference on Spider).
    matched_examples: List[ExecutionInstructions] = []
    if config.dataset_name.lower() == "spider":
        assert len(config.splits) == 1
        split = config.splits[0]

        for example in load_spider_examples(
            os.path.join(config.original_data_directory, split + ".json")
        ):
            # Looks up the example's schema.
            schema = schema_obj[example["db_id"]]

            # Returns a dictionary containing relevant prediction information.
            database_filepath = os.path.join(
                "spider_databases", example["db_id"] + ".sqlite"
            )
            key = " ".join(example["question_toks"])

            try:
                prediction = prediction_dict[key]
            except KeyError:
                continue

            matched_examples.append(
                {
                    "prediction": {
                        "utterance": key,
                        "predictions": prediction["predictions"],
                        "scores": prediction["scores"],
                    },
                    "gold": example["query"],
                    "database_path": os.path.join(
                        config.database_directory, database_filepath
                    ),
                    "empty_database_path": os.path.join(
                        config.empty_database_directory, database_filepath
                    ),
                    "schema": schema,
                }
            )

    elif config.dataset_name.lower() == "wikisql":
        raise ValueError("Inference on WikiSQL not supported.")
    else:
        dataset_path: str = os.path.join(
            config.original_data_directory, config.dataset_name + ".json"
        )
        for nl, sql in get_nl_sql_pairs(dataset_path, frozenset(config.splits)):
            # TODO(samuelstevens): What is the point of encoding then decoding? Simplify.
            key = nl.encode("utf8").decode("utf-8")

            # Returns a dictionary containing relevant prediction information.
            database_filepath = config.dataset_name + ".db"

            prediction = prediction_dict[key]

            matched_examples.append(
                {
                    "prediction": {
                        "utterance": key,
                        "predictions": prediction["predictions"],
                        "scores": prediction["scores"],
                    },
                    "gold": preprocess_sql(sql),
                    "database_path": os.path.join(
                        config.database_directory, database_filepath
                    ),
                    "empty_database_path": os.path.join(
                        config.empty_database_directory, database_filepath
                    ),
                    "schema": schema_obj,
                }
            )

    assert len(matched_examples) == len(
        predictions
    ), f"Only matched {len(matched_examples)} of {len(predictions)} examples."

    return matched_examples

    # with tf.gfile.Open(output_path, "w") as ofile:
    #     ofile.write(json.dumps(matched_examples))
