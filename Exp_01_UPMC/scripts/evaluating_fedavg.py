import time

from nvflare.app_common.workflows.fedavg import FedAvg


class EvaluatingFedAvg(FedAvg):
    """
    Standard NVFlare FedAvg with one additional operation:

    After each successfully persisted global aggregation, dispatch the
    aggregated global model to both sites for local held-out test evaluation.

    Test predictions and patient-level data remain at each institution.
    Evaluation does not affect aggregation, model selection, or training.
    """

    def __init__(
        self,
        *args,
        eval_targets=None,
        eval_timeout=3600,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.eval_targets = list(
            eval_targets or ["site-1", "external-site-1"]
        )
        self.eval_timeout = int(eval_timeout)

        if len(self.eval_targets) != 2:
            raise ValueError(
                "Exactly two evaluation sites are required: "
                "site-1 and external-site-1."
            )

        if len(set(self.eval_targets)) != len(self.eval_targets):
            raise ValueError("Evaluation site names must be unique.")

        if self.eval_timeout <= 0:
            raise ValueError("eval_timeout must be positive.")

        self._evaluation_responses = {}

    def save_model(self, model):
        """
        Preserve standard FedAvg persistence, then evaluate the persisted
        aggregated global model at both institutions.
        """
        super().save_model(model)

        self.evaluate_global_model(model)

    def _collect_evaluation_result(self, result):
        """Collect completion acknowledgements without patient-level data."""
        if result is None:
            raise RuntimeError(
                "Received an empty response to evaluate_global."
            )

        meta = result.meta or {}
        site_name = meta.get("client_name")

        if site_name not in self.eval_targets:
            raise RuntimeError(
                "Unexpected evaluation response from client "
                f"{site_name!r}."
            )

        if site_name in self._evaluation_responses:
            raise RuntimeError(
                f"Duplicate evaluation response from {site_name}."
            )

        # The server must never receive patient-level predictions.
        if result.params is not None:
            raise RuntimeError(
                f"{site_name} returned parameters during global evaluation. "
                "Expected an acknowledgement only."
            )

        self._evaluation_responses[site_name] = True

        self.info(
            f"Received local evaluation completion from {site_name} "
            f"for round {self.current_round}."
        )

    def evaluate_global_model(self, model):
        """Dispatch the aggregated model and wait for both site responses."""
        self._evaluation_responses = {}

        self.info(
            f"Dispatching aggregated global model for local held-out "
            f"evaluation: round {self.current_round}."
        )

        self.send_model(
            task_name="evaluate_global",
            data=model,
            targets=self.eval_targets,
            min_responses=len(self.eval_targets),
            timeout=self.eval_timeout,
            callback=self._collect_evaluation_result,
        )

        while True:
            standing = self.get_num_standing_tasks()

            if standing is None:
                raise RuntimeError(
                    "NVFlare did not provide a standing-task count; "
                    "cannot safely confirm completion of global evaluation."
                )

            if standing == 0:
                break

            if self.abort_signal.triggered:
                raise RuntimeError(
                    "Abort signal received while waiting for global "
                    "evaluation responses."
                )

            time.sleep(self._task_check_period)

        received = set(self._evaluation_responses)
        expected = set(self.eval_targets)

        if received != expected:
            missing = sorted(expected - received)
            raise RuntimeError(
                "Global evaluation did not complete at both sites. "
                f"Missing responses: {missing}"
            )

        self.info(
            f"Global evaluation completed at both sites for "
            f"round {self.current_round}."
        )
