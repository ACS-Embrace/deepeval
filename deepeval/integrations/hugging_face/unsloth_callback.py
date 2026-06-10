"""
Unsloth-compatible DeepEval callbacks.

Wraps model inference in Unsloth's FastLanguageModel.for_inference() /
FastLanguageModel.for_training() guards, which are required for correct
LoRA/PEFT behaviour during generation.

Single-callback usage
---------------------
    callback = DeepEvalUnslothCallback(trainer=trainer, ...)
    trainer.add_callback(callback)

Multi-callback usage (shared model, multiple eval datasets / metric sets)
--------------------------------------------------------------------------
HuggingFace invokes callbacks sequentially in one thread, so a traditional
blocking barrier is not needed.  UnslothBarrier uses reference-counting
instead: the first callback to arrive activates inference mode, and the
last one to finish restores training mode.  State resets automatically
for the next epoch (cyclic).

Pass a shared RichManager so that all callbacks render into a single live
display rather than each spawning their own.  The first callback passed the
RichManager should own it (default); the rest receive it as-is.

    from deepeval.integrations.hugging_face.rich_manager import RichManager

    shared_rm = RichManager(show_table=True, total_train_epochs=trainer.args.num_train_epochs)
    barrier = UnslothBarrier(n_callbacks=2)
    trainer.add_callback(DeepEvalUnslothCallback(..., barrier=barrier, rich_manager=shared_rm))
    trainer.add_callback(DeepEvalUnslothCallback(..., barrier=barrier, rich_manager=shared_rm))
"""

import warnings
from typing import Dict, List, Optional

from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BaseMetric
from deepeval.integrations.hugging_face.callback import DeepEvalHuggingFaceCallback
from deepeval.integrations.hugging_face.rich_manager import RichManager
from deepeval.integrations.hugging_face.utils import generate_test_cases

try:
    from unsloth import FastLanguageModel as _FastLanguageModel
except ImportError:
    _FastLanguageModel = None


class UnslothBarrier:
    """
    Coordinates a single for_inference / for_training toggle across
    multiple DeepEvalUnslothCallback instances that share the same model.

    Because HuggingFace calls callbacks sequentially (not concurrently),
    this is a reference-counting guard rather than a blocking barrier:

    - The *first* callback to call enter_inference() calls
      FastLanguageModel.for_inference(model).
    - Subsequent callbacks skip the switch (model is already in inference mode).
    - The *last* callback to call exit_inference() calls
      FastLanguageModel.for_training(model) and resets all counters so
      the barrier is ready for the next epoch (cyclic behaviour).

    If any callback raises during its inference step, exit_inference()
    is still called from a finally block, so the model is always
    restored to training mode.

    Args:
        n_callbacks: Total number of DeepEvalUnslothCallback instances
                     sharing this barrier.  Must be >= 1.
    """

    def __init__(self, n_callbacks: int) -> None:
        if n_callbacks < 1:
            raise ValueError("n_callbacks must be >= 1")
        self.n_callbacks = n_callbacks
        self._enter_count: int = 0
        self._exit_count: int = 0
        self._inference_active: bool = False

    def enter_inference(self, model) -> None:
        """
        Must be called by each callback before running inference.
        Only the first caller per epoch activates for_inference().
        """
        self._enter_count += 1
        if self._enter_count == 1:
            if _FastLanguageModel is None:
                raise ImportError(
                    "unsloth is not installed. Install it with: pip install unsloth"
                )
            _FastLanguageModel.for_inference(model)
            self._inference_active = True

    def exit_inference(self, model) -> None:
        """
        Must be called by each callback after inference, even on failure.
        Only the last caller per epoch restores for_training() and resets
        the barrier for the next epoch.
        """
        self._exit_count += 1
        if self._exit_count >= self.n_callbacks:
            if self._inference_active:
                if _FastLanguageModel is None:
                    warnings.warn(
                        "[DeepEval] Cannot restore training mode: "
                        "unsloth is not installed."
                    )
                else:
                    _FastLanguageModel.for_training(model)
            # Reset for the next epoch (cyclic).
            self._enter_count = 0
            self._exit_count = 0
            self._inference_active = False

    @property
    def in_inference(self) -> bool:
        """True while the model is currently in inference mode."""
        return self._inference_active


class DeepEvalUnslothCallback(DeepEvalHuggingFaceCallback):
    """
    DeepEval callback with Unsloth for_inference / for_training guards.

    Inherits all behaviour from DeepEvalHuggingFaceCallback and overrides
    on_epoch_end to switch the model into inference mode before generating
    outputs and back to training mode afterwards — always, even if the
    evaluation raises.

    Args:
        trainer:            HuggingFace Trainer instance.
        evaluation_dataset: Dataset containing goldens to evaluate against.
        metrics:            List of DeepEval BaseMetric instances.
        tokenizer_args:     Forwarded to tokenizer(...).
        aggregation_method: "avg" (default), "min", or "max".
        show_table:         Display the Rich metrics table if True.
        generator_args:     Forwarded to model.generate(...).
        barrier:            Optional shared UnslothBarrier.  Pass the same
                            instance to every callback that shares a model
                            so for_inference is toggled exactly once per
                            epoch regardless of how many callbacks run.
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
        barrier: Optional[UnslothBarrier] = None,
        rich_manager: Optional[RichManager] = None,
        timeout_s: int = 0,
        eval_every_steps: Optional[int] = None,
    ) -> None:
        super().__init__(
            trainer=trainer,
            evaluation_dataset=evaluation_dataset,
            metrics=metrics,
            tokenizer_args=tokenizer_args,
            aggregation_method=aggregation_method,
            show_table=show_table,
            generator_args=generator_args,
            rich_manager=rich_manager,
        )
        self._barrier = barrier
        self._owns_inference: bool = False  # only used when barrier is None
        self._timeout_s: int = timeout_s
        self._eval_every_steps: Optional[int] = eval_every_steps
        self._last_step_eval: int = -1
        self.last_test_case_results: list = []

        if _FastLanguageModel is None and barrier is None:
            warnings.warn(
                "[DeepEval] unsloth is not installed. "
                "DeepEvalUnslothCallback will run without "
                "for_inference / for_training mode switching."
            )

    def _activate_inference(self, model) -> None:
        """Switch model to inference mode."""
        if self._barrier is not None:
            self._barrier.enter_inference(model)
        elif _FastLanguageModel is not None:
            _FastLanguageModel.for_inference(model)
            self._owns_inference = True

    def _deactivate_inference(self, model) -> None:
        """
        Restore model to training mode.
        Safe to call even if _activate_inference was never reached —
        always called from a finally block.
        """
        if self._barrier is not None:
            self._barrier.exit_inference(model)
        elif self._owns_inference and _FastLanguageModel is not None:
            try:
                _FastLanguageModel.for_training(model)
            finally:
                self._owns_inference = False

    @property
    def _should_evaluate(self) -> bool:
        """
        Controls whether on_epoch_end runs inference + evaluation.
        Defaults to show_table so the base callback is unchanged.
        Subclasses (e.g. the W&B callback) can override to True so
        evaluation always runs regardless of the show_table setting.
        """
        return self.show_table

    def _run_inference_evaluation(self, state: TrainerState) -> Optional[Dict]:
        """
        Shared helper: switch to inference mode, generate outputs, evaluate,
        restore training mode.  Returns the aggregated scores dict or None.
        """
        model = self.trainer.model
        self._owns_inference = False
        scores = None
        try:
            self._activate_inference(model)

            self.rich_manager.change_spinner_text(
                self.task_descriptions["generating"]
            )
            test_cases = generate_test_cases(
                model,
                self.trainer.tokenizer,
                self.tokenizer_args,
                self.evaluation_dataset,
                self.generator_args,
            )
            self.evaluation_dataset.test_cases = test_cases

            self.rich_manager.change_spinner_text(
                self.task_descriptions["evaluate"]
            )
             # Prevent indefinite hang on slow/dropped OpenRouter connections
            if self._timeout_s > 0:
                try:
                    import litellm
                    litellm.request_timeout = self._timeout_s
                except Exception:
                    pass

            scores = self._calculate_metric_scores()

            # Build per-sample results for downstream savers
            self.last_test_case_results = [
                {
                    "input":    tc.input,
                    "expected": tc.expected_output,
                    "actual":   tc.actual_output,
                    "score":    tc.metrics_data[0].score if tc.metrics_data else None,
                    "passed":   tc.metrics_data[0].success if tc.metrics_data else None,
                }
                for tc in (self.evaluation_dataset.test_cases or [])
                if tc.actual_output
            ]
        finally:
            try:
                self._deactivate_inference(model)
            except Exception as restore_e:
                print(
                    f"[DeepEval] Warning: failed to restore training mode: {restore_e}"
                )
        return scores

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Run a baseline evaluation before any training steps so we have an
        epoch-0 reference point in both the Rich table and W&B.
        """
        super().on_train_begin(args, state, control, **kwargs)

        if not self._should_evaluate:
            return

        try:
            baseline_scores = self._run_inference_evaluation(state)
            if baseline_scores:
                self.rich_manager.contribute_epoch_data(0, baseline_scores)
                if self.show_table:
                    columns = self._generate_table()
                    self.rich_manager.update(columns)
        except Exception as e:
            print(
                f"[DeepEval] Warning: baseline evaluation failed and was skipped: {e}"
            )

        self.rich_manager.change_spinner_text(self.task_descriptions["training"])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Wraps the parent's generate + evaluate block in Unsloth mode guards.

        Execution order
        ---------------
        1. Set control.should_log = True (mirrors parent behaviour).
        2. Early-return if _should_evaluate is False (no inference needed).
        3. Reset _pending_scores to None so a failed epoch never re-logs
           stale scores from a previous epoch.
        4. Call _activate_inference() — may be a no-op if this callback
           is not the first to arrive at the barrier this epoch.
        5. Generate test-case outputs via the model.
        6. Run DeepEval metrics.
        7. In the finally block, call _deactivate_inference() regardless
           of success or failure so the model always returns to training.
        """
        try:
            control.should_log = True

            if not self._should_evaluate:
                return

            # Reset so a failed epoch never re-logs stale scores from
            # a previous epoch.
            self._pending_scores = None

            try:
                self._pending_scores = self._run_inference_evaluation(state)
            except Exception as inner_e:
                print(
                    f"[DeepEval] Warning: on_epoch_end inference/evaluation "
                    f"failed and was skipped: {inner_e}"
                )

        except Exception as e:
            print(
                f"[DeepEval] Warning: on_epoch_end failed and was skipped: {e}"
            )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._eval_every_steps is None:
            return control
        step = state.global_step
        if step == 0 or step == self._last_step_eval:
            return control
        if step % self._eval_every_steps != 0:
            return control
        self._last_step_eval = step
        try:
            scores = self._run_inference_evaluation(state)
            if scores:
                self._log_step_scores(scores, step)
        except Exception as e:
            print(f"[DeepEval] Warning: on_step_end eval failed: {e}")
        return control

    def _log_step_scores(self, scores: Dict, step: int) -> None:
        """Log step-based scores. Subclasses override to add W&B logging."""
        pass


class DeepEvalUnslothWandbCallback(DeepEvalUnslothCallback):
    """
    Extends DeepEvalUnslothCallback with Weights & Biases metric logging.

    After each epoch's evaluation completes, the DeepEval metric scores
    stored in _pending_scores are logged to the active wandb run under
    the "deepeval/" namespace (e.g. "deepeval/Answer Relevancy").
    Standard training metrics (loss, lr, etc.) are left to the HuggingFace
    WandbCallback to handle — this only adds the deepeval layer on top.

    Args:
        trainer:            HuggingFace Trainer instance.
        evaluation_dataset: Dataset containing goldens to evaluate against.
        metrics:            List of DeepEval BaseMetric instances.
        tokenizer_args:     Forwarded to tokenizer(...).
        aggregation_method: "avg" (default), "min", or "max".
        show_table:         Display the Rich metrics table if True.
        generator_args:     Forwarded to model.generate(...).
        barrier:            Optional shared UnslothBarrier.
        wandb_prefix:       Prefix applied to every logged key.
                            Defaults to "deepeval/".
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
        barrier: Optional[UnslothBarrier] = None,
        rich_manager: Optional[RichManager] = None,
        wandb_prefix: str = "deepeval/",
        timeout_s: int = 0,
        eval_every_steps: Optional[int] = None,
    ) -> None:
        super().__init__(
            trainer=trainer,
            evaluation_dataset=evaluation_dataset,
            metrics=metrics,
            tokenizer_args=tokenizer_args,
            aggregation_method=aggregation_method,
            show_table=show_table,
            generator_args=generator_args,
            barrier=barrier,
            rich_manager=rich_manager,
            timeout_s=timeout_s,
            eval_every_steps=eval_every_steps,
        )
        self._wandb_prefix = wandb_prefix

        try:
            import wandb as _wandb  # noqa: F401
        except ImportError:
            warnings.warn(
                "[DeepEval] wandb is not installed. "
                "DeepEvalUnslothWandbCallback will not log metrics. "
                "Install it with: pip install wandb"
            )

    @property
    def _should_evaluate(self) -> bool:
        """Always evaluate so wandb metrics are logged regardless of show_table."""
        return True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Run baseline evaluation (via parent) then log the results to W&B at
        step=0.  It is safe to use an explicit step here because the Trainer
        hasn't started logging yet, so there is no monotonicity race.
        """
        super().on_train_begin(args, state, control, **kwargs)

        baseline = self.rich_manager.get_epoch_data(0)
        if not baseline:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            log_payload = {
                f"{self._wandb_prefix}{k}": v for k, v in baseline.items()
            }
            wandb.log(log_payload, step=0)
        except Exception as e:
            print(
                f"[DeepEval] Warning: baseline wandb logging failed and was skipped: {e}"
            )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Calls the parent on_epoch_end (inference guard + evaluation),
        then logs whatever landed in _pending_scores to wandb.
        """
        super().on_epoch_end(args, state, control, **kwargs)

        # Read but do NOT clear _pending_scores here — on_log still needs it
        # to update the Rich table.  The reset at the top of the next epoch's
        # on_epoch_end prevents stale replay.
        if not self._pending_scores:
            return

        try:
            import wandb

            if wandb.run is None:
                warnings.warn(
                    "[DeepEval] No active wandb run found. "
                    "Call wandb.init() before training to enable logging."
                )
                return

            log_payload = {
                f"{self._wandb_prefix}{k}": v
                for k, v in self._pending_scores.items()
            }
            # Do NOT pass an explicit step.  Evaluation is slow — by the time
            # wandb.log() is called, the Trainer's WandbCallback has already
            # advanced the step counter past state.global_step, causing W&B to
            # silently drop any log with a step <= its current step.
            # Letting W&B auto-increment avoids the race at the cost of a small
            # step offset (the deepeval point appears at the step when eval
            # finished, not when it started).
            wandb.log(log_payload)

        except Exception as e:
            print(
                f"[DeepEval] Warning: wandb logging failed and was skipped: {e}"
            )

    def _log_step_scores(self, scores: Dict, step: int) -> None:
        try:
            import wandb
            if wandb.run is None:
                return
            wandb.log(
                {f"{self._wandb_prefix}{k}": v for k, v in scores.items()},
                step=step,
            )
        except Exception as e:
            print(f"[DeepEval] Warning: step W&B logging failed: {e}")
