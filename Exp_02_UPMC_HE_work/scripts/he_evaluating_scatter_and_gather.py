# import gc
import threading

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.controller_spec import (
    ClientTask,
    OperatorMethod,
    Task,
    TaskOperatorKey,
)
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.security.logging import secure_format_exception


class HEEvaluatingScatterAndGather(ScatterAndGather):
    """NVFLARE 2.7 ScatterAndGather with post-aggregation local evaluation.

    The parent ScatterAndGather workflow performs the training round using the
    configured HE aggregator and HE shareable generator. This subclass retains
    that flow and adds an ``evaluate_global`` broadcast after each successful
    aggregation/persist operation.

    Evaluation data is sent using the configured shareable generator. In an HE
    job, the client-side HEModelDecryptor must therefore also be attached to the
    ``evaluate_global`` task. The evaluation acknowledgement is metrics-only and
    must not be passed through the train-task HE encryptor.
    """

    def __init__(
        self,
        *args,
        eval_targets=None,
        eval_timeout=3600,
        eval_task_name="evaluate_global",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.eval_targets = list(
            eval_targets or ["site-1", "external-site-1"]
        )
        self.eval_timeout = int(eval_timeout)
        self.eval_task_name = str(eval_task_name)

        if len(self.eval_targets) != 2:
            raise ValueError(
                "Exactly two evaluation sites are required: "
                "site-1 and external-site-1."
            )

        if len(set(self.eval_targets)) != len(self.eval_targets):
            raise ValueError("Evaluation site names must be unique.")

        if self.eval_timeout <= 0:
            raise ValueError("eval_timeout must be positive.")

        if not self.eval_task_name:
            raise ValueError("eval_task_name must be non-empty.")

        self._evaluation_responses = set()
        self._evaluation_errors = {}
        self._evaluation_lock = threading.Lock()

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """Run NVFLARE 2.7 ScatterAndGather, then evaluate each global model.

        This method follows the NVFLARE 2.7 ScatterAndGather.control_flow
        sequence. The only additional operation is the global evaluation
        broadcast after the aggregated ModelLearnable has been persisted.
        """
        try:
            self.log_info(
                fl_ctx,
                "Beginning HE-enabled ScatterAndGather training phase.",
            )
            self._phase = AppConstants.PHASE_TRAIN
            fl_ctx.set_prop(
                AppConstants.PHASE,
                self._phase,
                private=True,
                sticky=False,
            )
            fl_ctx.set_prop(
                AppConstants.NUM_ROUNDS,
                self._num_rounds,
                private=True,
                sticky=False,
            )
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            if self._current_round is None:
                self._current_round = self._start_round

            while self._current_round < self._start_round + self._num_rounds:
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(
                    fl_ctx,
                    f"Round {self._current_round} started.",
                )
                fl_ctx.set_prop(
                    AppConstants.GLOBAL_MODEL,
                    self._global_weights,
                    private=True,
                    sticky=True,
                )
                fl_ctx.set_prop(
                    AppConstants.CURRENT_ROUND,
                    self._current_round,
                    private=True,
                    sticky=True,
                )
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Distribute the current global model for local training.
                data_shareable = (
                    self.shareable_gen.learnable_to_shareable(
                        self._global_weights,
                        fl_ctx,
                    )
                )
                data_shareable.set_header(
                    AppConstants.CURRENT_ROUND,
                    self._current_round,
                )
                data_shareable.set_header(
                    AppConstants.NUM_ROUNDS,
                    self._num_rounds,
                )
                data_shareable.add_cookie(
                    AppConstants.CONTRIBUTION_ROUND,
                    self._current_round,
                )

                operator = {
                    TaskOperatorKey.OP_ID: self.train_task_name,
                    TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
                    TaskOperatorKey.TIMEOUT: self._train_timeout,
                    TaskOperatorKey.AGGREGATOR: self.aggregator_id,
                }

                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    operator=operator,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )
                # Match NVFLARE ScatterAndGather client tracking.
                self._current_failed_clients = set()
                self._current_num_targets = len(self._engine.get_clients())

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=(
                        self._wait_time_after_min_received
                    ),
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, "Start HE/FedAvg aggregation.")
                self.fire_event(
                    AppEventType.BEFORE_AGGREGATION,
                    fl_ctx,
                )
                aggr_result = self.aggregator.aggregate(fl_ctx)
                self.fire_event_with_data(
                    AppEventType.AFTER_AGGREGATION,
                    fl_ctx,
                    AppConstants.AGGREGATION_RESULT,
                    aggr_result,
                )
                self.log_info(fl_ctx, "End HE/FedAvg aggregation.")

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(
                    AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE,
                    fl_ctx,
                )
                self._global_weights = (
                    self.shareable_gen.shareable_to_learnable(
                        aggr_result,
                        fl_ctx,
                    )
                )

                if not isinstance(self._global_weights, ModelLearnable):
                    raise RuntimeError(
                        "HE shareable generator did not return a "
                        "ModelLearnable global model."
                    )

                fl_ctx.set_prop(
                    AppConstants.GLOBAL_MODEL,
                    self._global_weights,
                    private=True,
                    sticky=True,
                )
                fl_ctx.sync_sticky()
                self.fire_event(
                    AppEventType.AFTER_SHAREABLE_TO_LEARNABLE,
                    fl_ctx,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                # Preserve the standard ScatterAndGather persistence behavior.
                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (
                            self._current_round + 1
                        ) % self._persist_every_n_rounds == 0
                    ) or (
                        self._current_round
                        == self._start_round + self._num_rounds - 1
                    ):
                        self.log_info(
                            fl_ctx,
                            "Start persist model on server.",
                        )
                        self.fire_event(
                            AppEventType.BEFORE_LEARNABLE_PERSIST,
                            fl_ctx,
                        )
                        self.persistor.save(
                            self._global_weights,
                            fl_ctx,
                        )
                        self.fire_event(
                            AppEventType.AFTER_LEARNABLE_PERSIST,
                            fl_ctx,
                        )
                        self.log_info(
                            fl_ctx,
                            "End persist model on server.",
                        )

                # New behavior: evaluate the newly aggregated global model
                # locally at UPMC and Moffitt before starting the next round.
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.evaluate_global_model(
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(
                    fl_ctx,
                    f"Round {self._current_round} finished.",
                )

                self._current_round += 1

                if (
                    self._snapshot_every_n_rounds != 0
                    and self._current_round % self._snapshot_every_n_rounds == 0
                ):
                    self._engine.persist_components(
                        fl_ctx,
                        completed=False,
                    )

                # gc.collect()
                # Reset accumulated updates before the next training round.
                self.aggregator.reset(fl_ctx)

                # Respect NVFLARE's configured memory cleanup interval.
                self._maybe_cleanup_memory()

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(
                fl_ctx,
                "Finished HE-enabled ScatterAndGather training.",
            )

        except Exception as e:
            error_msg = (
                "Exception in HEEvaluatingScatterAndGather control_flow: "
                f"{secure_format_exception(e)}"
            )
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def evaluate_global_model(
        self,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> None:
        """Broadcast the current aggregated global model for local evaluation.

        Only a completion acknowledgement is accepted from each site. Metrics
        and patient-level predictions remain at the institution, as in the
        existing Exp_02_UPMC client implementation.
        """
        self._evaluation_responses = set()
        self._evaluation_errors = {}

        self.log_info(
            fl_ctx,
            "Dispatching aggregated global model for local held-out "
            f"evaluation: round {self._current_round}.",
        )

        # Use the HE shareable generator so the task payload follows the same
        # representation as the federated model exchange. The client-side
        # HEModelDecryptor must include eval_task_name in its task list.
        eval_shareable = self.shareable_gen.learnable_to_shareable(
            self._global_weights,
            fl_ctx,
        )
        eval_shareable.set_header(
            AppConstants.CURRENT_ROUND,
            self._current_round,
        )
        eval_shareable.set_header(
            AppConstants.NUM_ROUNDS,
            self._num_rounds,
        )

        eval_task = Task(
            name=self.eval_task_name,
            data=eval_shareable,
            timeout=self.eval_timeout,
            result_received_cb=self._process_global_eval_result,
        )

        self.broadcast_and_wait(
            task=eval_task,
            fl_ctx=fl_ctx,
            targets=self.eval_targets,
            min_responses=len(self.eval_targets),
            wait_time_after_min_received=0,
            abort_signal=abort_signal,
        )

        if abort_signal.triggered:
            self.log_info(
                fl_ctx,
                "Abort signal received while waiting for global evaluation.",
            )
            return

        received = set(self._evaluation_responses)
        expected = set(self.eval_targets)

        if self._evaluation_errors:
            details = "; ".join(
                f"{site}: {message}"
                for site, message in sorted(
                    self._evaluation_errors.items()
                )
            )
            raise RuntimeError(
                "Global evaluation failed: " + details
            )

        if received != expected:
            missing = sorted(expected - received)
            raise RuntimeError(
                "Global evaluation did not complete at both sites. "
                f"Missing responses: {missing}"
            )

        self.log_info(
            fl_ctx,
            "Global evaluation completed at both sites for "
            f"round {self._current_round}.",
        )

    def _process_global_eval_result(
        self,
        client_task: ClientTask,
        fl_ctx: FLContext,
    ) -> None:
        """Validate a metrics-only evaluation acknowledgement."""
        client_name = client_task.client.name
        result = client_task.result

        try:
            if client_name not in self.eval_targets:
                raise RuntimeError(
                    f"Unexpected evaluation response from {client_name!r}."
                )

            with self._evaluation_lock:
                if client_name in self._evaluation_responses:
                    raise RuntimeError(
                        f"Duplicate evaluation response from {client_name}."
                    )

            if result is None:
                raise RuntimeError(
                    "Received an empty response to evaluate_global."
                )

            return_code = result.get_return_code()
            if return_code != ReturnCode.OK:
                raise RuntimeError(
                    f"Client returned NVFLARE status {return_code!r}."
                )

            dxo = from_shareable(result)

            if dxo.data_kind != DataKind.METRICS:
                raise RuntimeError(
                    "Expected a metrics-only acknowledgement; received "
                    f"data kind {dxo.data_kind!r}."
                )

            metrics = dxo.data
            if not isinstance(metrics, dict):
                raise RuntimeError(
                    "Evaluation acknowledgement metrics must be a dictionary."
                )

            # Exp_02_UPMC's evaluation client intentionally returns only this
            # acknowledgement. Do not transmit predictions or patient-level
            # evaluation results to the server.
            unexpected_metrics = set(metrics) - {"evaluation_complete"}
            if unexpected_metrics:
                raise RuntimeError(
                    "Evaluation response included unexpected fields: "
                    f"{sorted(unexpected_metrics)}. Only the completion "
                    "acknowledgement may leave the institution."
                )

            if float(metrics.get("evaluation_complete", 0.0)) != 1.0:
                raise RuntimeError(
                    "Client did not confirm successful global evaluation."
                )

            with self._evaluation_lock:
                self._evaluation_responses.add(client_name)

            self.log_info(
                fl_ctx,
                f"Received local evaluation completion from {client_name} "
                f"for round {self._current_round}.",
            )

        except Exception as e:
            with self._evaluation_lock:
                self._evaluation_errors[client_name] = str(e)

            self.log_error(
                fl_ctx,
                f"Invalid evaluate_global response from {client_name}: {e}",
            )

        finally:
            # Release the result payload promptly.
            client_task.result = None
