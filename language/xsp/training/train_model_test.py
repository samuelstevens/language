import os
import tempfile
from typing import List

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
        experiment.delete()

    def test_delete_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_model.delete_checkpoint(tmpdir, 0)

    def _make_checkpoint_helper(self, tmpdir: str, step: int) -> List[str]:
        meta_file = f"{train_model.checkpoint_path(tmpdir, step)}.meta"
        with open(meta_file, "w") as file:
            file.write("meta")
        index_file = f"{train_model.checkpoint_path(tmpdir, step)}.index"
        with open(index_file, "w") as file:
            file.write("index")
        data_file = f"{train_model.checkpoint_path(tmpdir, step)}.data-0-of-1"
        with open(data_file, "w") as file:
            file.write("data")

        return list(map(os.path.basename, [meta_file, index_file, data_file]))

    def test_delete_existing_checkpoint(self) -> None:
        step = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_checkpoint_helper(tmpdir, step)

            train_model.delete_checkpoint(tmpdir, step)

            self.assertEmpty(os.listdir(tmpdir))

    def test_dont_delete_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_checkpoint_helper(tmpdir, 100)
            files = self._make_checkpoint_helper(tmpdir, 1000)

            train_model.delete_checkpoint(tmpdir, 100)

            self.assertEqual(set(os.listdir(tmpdir)), set(files))

    def test_dont_delete_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_checkpoint_helper(tmpdir, 100)
            files = self._make_checkpoint_helper(tmpdir, 200)
            files += self._make_checkpoint_helper(tmpdir, 300)

            train_model.delete_checkpoint(tmpdir, 100)

            self.assertEqual(set(os.listdir(tmpdir)), set(files))

    def test_integration(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_checkpoint_helper(tmpdir, 10000)
            self._make_checkpoint_helper(tmpdir, 20000)
            self._make_checkpoint_helper(tmpdir, 30000)
            self._make_checkpoint_helper(tmpdir, 40000)
            self._make_checkpoint_helper(tmpdir, 50000)
            self._make_checkpoint_helper(tmpdir, 60000)
            self._make_checkpoint_helper(tmpdir, 70000)
            self._make_checkpoint_helper(tmpdir, 80000)
            files = self._make_checkpoint_helper(tmpdir, 90000)
            files += self._make_checkpoint_helper(tmpdir, 100000)

            for step in train_model.checkpoints_to_delete(experiment):
                train_model.delete_checkpoint(tmpdir, step)

            self.assertEqual(set(os.listdir(tmpdir)), set(files))

        experiment.delete()


if __name__ == "__main__":
    absltest.main()
