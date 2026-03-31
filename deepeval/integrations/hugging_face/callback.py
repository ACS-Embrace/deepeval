from typing import Union, List, Dict, Optional
from .utils import get_column_order, generate_test_cases
from .rich_manager import RichManager

from deepeval.metrics import BaseMetric
from deepeval.evaluate.execute import execute_test_cases
from deepeval.dataset import EvaluationDataset

try:
    from transformers import (
        TrainerCallback,
        ProgressCallback,
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
    )

    class DeepEvalHuggingFaceCallback(TrainerCallback):
        """
        Custom callback for deep evaluation during model training.

        Args:
            metrics (List[BaseMetric]): List of evaluation metrics.
            evaluation_dataset (EvaluationDataset): Dataset for evaluation.
            tokenizer_args (Dict): Arguments for the tokenizer.
            aggregation_method (str): Method for aggregating metric scores.
            trainer (Trainer): Model trainer.
        """

        def __init__(
            self,
            trainer: Trainer,
            evaluation_dataset: EvaluationDataset = None,
            metrics: List[BaseMetric] = None,
            tokenizer_args: Dict = None,
            aggregation_method: str = "avg",
            show_table: bool = False,
            generator_args: Dict = None,
            rich_manager: Optional[RichManager] = None,
        ) -> None:
            super().__init__()

            self.show_table = show_table
            self.metrics = metrics
            self.evaluation_dataset = evaluation_dataset
            self.tokenizer_args = tokenizer_args
            self.generator_args = generator_args
            self.aggregation_method = aggregation_method
            self.trainer = trainer

            self.task_descriptions = {
                "generating": "[blue][STATUS] [white]Generating output from model (might take up few minutes)",
                "training": "[blue][STATUS] [white]Training in Progress",
                "evaluate": "[blue][STATUS] [white]Evaluating test-cases (might take up few minutes)",
                "training_end": "[blue][STATUS] [white]Training Ended",
            }

            self.train_bar_started = False
            self.epoch_counter = 0
            self.deepeval_metric_history = []
            self._pending_scores = None

            total_train_epochs = self.trainer.args.num_train_epochs
            if rich_manager is not None:
                self.rich_manager = rich_manager
            else:
                self.rich_manager = RichManager(show_table, total_train_epochs)
            self.trainer.remove_callback(ProgressCallback)

        def _calculate_metric_scores(self) -> Dict[str, List[float]]:
            """
            Calculate final evaluation scores based on metrics and test cases.

            Returns:
                Dict[str, List[float]]: Metric scores for each test case.
            """
            valid_test_cases = [
                tc for tc in self.evaluation_dataset.test_cases
                if tc.actual_output and tc.actual_output.strip()
            ]
            skipped = len(self.evaluation_dataset.test_cases) - len(valid_test_cases)
            if skipped:
                print(
                    f"[DeepEval] Warning: {skipped} test case(s) skipped "
                    f"due to empty actual_output."
                )
            if not valid_test_cases:
                return {}

            # Evaluate one test case at a time so a single evaluation LLM
            # failure (e.g. invalid JSON response) doesn't discard the results
            # of every other test case.
            scores = {}
            eval_failures = 0
            for tc in valid_test_cases:
                try:
                    test_results = execute_test_cases(
                        test_cases=[tc],
                        metrics=self.metrics,
                    )
                    for test_result in test_results:
                        for metric in test_result.metrics_data:
                            if metric.score is None:
                                continue
                            scores.setdefault(str(metric.name), []).append(metric.score)
                except Exception as e:
                    eval_failures += 1
                    print(f"[DeepEval] Warning: test case evaluation failed and was skipped: {e}")

            if eval_failures:
                print(
                    f"[DeepEval] Warning: {eval_failures} test case(s) failed "
                    f"during evaluation and were excluded from scores."
                )
            return self._aggregate_scores(scores) if scores else {}

        def _aggregate_scores(
            self, scores: Dict[str, List[float]]
        ) -> Dict[str, float]:
            """
            Aggregate metric scores using the specified method.

            Args:
                aggregation_method (str): Method for aggregating scores.
                scores (Dict[str, List[float]]): Metric scores for each test case.

            Returns:
                Dict[str, float]: Aggregated metric scores.
            """
            aggregation_functions = {
                "avg": lambda x: sum(x) / len(x),
                "max": max,
                "min": min,
            }
            if self.aggregation_method not in aggregation_functions:
                raise ValueError(
                    "Incorrect 'aggregation_method', only accepts ['avg', 'min, 'max']"
                )
            return {
                key: aggregation_functions[self.aggregation_method](value)
                for key, value in scores.items()
            }

        def on_epoch_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the beginning of each training epoch.
            """
            try:
                self.epoch_counter += 1
            except Exception as e:
                print(f"[DeepEval] Warning: on_epoch_begin failed and was skipped: {e}")

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the end of each training epoch.
            Generates test cases and evaluates them so that on_log only
            needs to merge the pre-computed scores with training metrics.
            """
            try:
                control.should_log = True

                if not self.show_table:
                    return

                self.rich_manager.change_spinner_text(
                    self.task_descriptions["generating"]
                )
                test_cases = generate_test_cases(
                    self.trainer.model,
                    self.trainer.tokenizer,
                    self.tokenizer_args,
                    self.evaluation_dataset,
                    self.generator_args,
                )
                self.evaluation_dataset.test_cases = test_cases

                self.rich_manager.change_spinner_text(
                    self.task_descriptions["evaluate"]
                )
                self._pending_scores = self._calculate_metric_scores()
            except Exception as e:
                print(f"[DeepEval] Warning: on_epoch_end failed and was skipped: {e}")

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered after logging the last logs.
            Merges pre-computed deepeval scores with the trainer's logged
            training metrics and updates the display table.
            """
            try:
                if (
                    self.show_table
                    and self._pending_scores is not None
                    and len(self.deepeval_metric_history) + 1 <= state.epoch
                ):
                    self.rich_manager.advance_progress()

                    scores = dict(self._pending_scores)
                    self._pending_scores = None
                    scores.update(state.log_history[-1])
                    self.deepeval_metric_history.append(scores)

                    self.rich_manager.change_spinner_text(
                        self.task_descriptions["training"]
                    )
                    columns = self._generate_table()
                    self.rich_manager.update(columns)
            except Exception as e:
                print(f"[DeepEval] Warning: on_log failed and was skipped: {e}")

        def _generate_table(self):
            """
            Generates table, along with progress bars

            Returns:
                rich.Columns: contains table and 2 progress bars
            """
            column, table = self.rich_manager.create_column()
            all_keys = {}
            for row in self.deepeval_metric_history:
                all_keys.update(row)
            order = get_column_order(all_keys)

            if self.show_table:
                for key in order:
                    table.add_column(key)

                for row in self.deepeval_metric_history:
                    table.add_row(*[str(row.get(value, "N/A")) for value in order])

            return column

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the end of model training.
            """
            try:
                self.rich_manager.change_spinner_text(
                    self.task_descriptions["training_end"]
                )
                self.rich_manager.stop()
            except Exception as e:
                print(f"[DeepEval] Warning: on_train_end failed and was skipped: {e}")

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the beginning of model training.
            """
            try:
                self.rich_manager.start()
                self.rich_manager.change_spinner_text(
                    self.task_descriptions["training"]
                )
            except Exception as e:
                print(f"[DeepEval] Warning: on_train_begin failed and was skipped: {e}")

except ImportError:

    class DeepEvalHuggingFaceCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'transformers' library is required to use the DeepEvalHuggingFaceCallback."
            )
