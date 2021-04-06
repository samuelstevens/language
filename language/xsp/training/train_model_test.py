import os
import tempfile

import keepsake
from absl.testing import absltest

from language.xsp.training import train_model


class CheckpointManagementTest(absltest.TestCase):
    def test_nothing_to_delete(self) -> None:
        experiment = keepsake.init(params={})
        experiment.checkpoint(
            step=100,
            metrics={"eval_execution_f1": 0.5},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        experiment.checkpoint(
            step=200,
            metrics={"eval_execution_f1": 0.4},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        self.assertEqual(train_model.checkpoints_to_delete(experiment), [])
        experiment.delete()

    def test_remove_old_checkpoint(self) -> None:
        experiment = keepsake.init(params={})
        experiment.checkpoint(
            step=100,
            metrics={"eval_execution_f1": 0.5},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        experiment.checkpoint(
            step=200,
            metrics={"eval_execution_f1": 0.6},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        self.assertEqual(train_model.checkpoints_to_delete(experiment), [100])
        experiment.delete()

    def test_remove_middle_checkpoint(self) -> None:
        experiment = keepsake.init(params={})
        experiment.checkpoint(
            step=100,
            metrics={"eval_execution_f1": 0.7},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        experiment.checkpoint(
            step=200,
            metrics={"eval_execution_f1": 0.6},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        experiment.checkpoint(
            step=300,
            metrics={"eval_execution_f1": 0.3},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        self.assertEqual(train_model.checkpoints_to_delete(experiment), [200])
        experiment.delete()

    def test_only_one_checkpoint(self) -> None:
        experiment = keepsake.init(params={})
        experiment.checkpoint(
            step=100,
            metrics={"eval_execution_f1": 0.7},
            primary_metric=("eval_execution_f1", "maximize"),
        )
        self.assertEqual(train_model.checkpoints_to_delete(experiment), [])
        experiment.delete()

    def test_real_example(self) -> None:
        experiment = keepsake.init()
        checkpoints = [
            (10000, 0.42, 1.34),
            (20000, 0.56, 0.17),
            (30000, 0.59363, 0.10),
            (40000, 0.58, 0.076),
            (50000, 0.61, 0.06),
            (60000, 0.61, 0.04),
            (70000, 0.61, 0.04),
            (80000, 0.61, 0.03),
            (90000, 0.62, 0.02),
            (100000, 0.61, 0.02),
        ]
        for step, eval_score, train_score in checkpoints:
            experiment.checkpoint(
                step=step,
                metrics={"eval": eval_score, "train": train_score},
                primary_metric=("eval", "maximize"),
            )

        self.assertEqual(
            train_model.checkpoints_to_delete(experiment),
            [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000],
        )

    def test_delete_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_model.delete_checkpoint(tmpdir, 0)

    def test_delete_existing_checkpoint(self) -> None:
        step = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{train_model.checkpoint_path(tmpdir, step)}.meta", "w") as file:
                file.write("meta")
            with open(
                f"{train_model.checkpoint_path(tmpdir, step)}.index", "w"
            ) as file:
                file.write("index")
            with open(
                f"{train_model.checkpoint_path(tmpdir, step)}.data-0-of-1", "w"
            ) as file:
                file.write("data")

            train_model.delete_checkpoint(tmpdir, step)

            self.assertEmpty(os.listdir(tmpdir))


if __name__ == "__main__":
    absltest.main()
