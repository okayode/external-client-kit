"""Job-local TenSEAL context loading for NVFLARE HE components."""

from pathlib import Path

import tenseal as ts

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey

from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.app_opt.he.model_serialize_filter import HEModelSerializeFilter
from nvflare.app_opt.he.model_shareable_generator import HEModelShareableGenerator


def load_job_local_context(context_file, fl_ctx):
    """Load a TenSEAL context from the current site's job custom directory."""

    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    # job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
    job_id = fl_ctx.get_job_id()

    if workspace is None:
        raise RuntimeError("NVFLARE workspace object is unavailable.")

    if not job_id:
        raise RuntimeError("NVFLARE current job ID is unavailable.")

    filename = Path(context_file)

    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError(
            f"Invalid job-local HE context filename: {context_file}"
        )

    custom_dir = Path(workspace.get_app_custom_dir(job_id)).resolve()
    context_path = (custom_dir / filename).resolve()

    if custom_dir not in context_path.parents:
        raise ValueError("HE context path escapes the job custom directory.")

    if not context_path.is_file():
        raise FileNotFoundError(
            f"Job-local HE context not found: {context_path}"
        )

    context = ts.context_from(context_path.read_bytes())

    return context


def _handle_context_event(component, event_type, fl_ctx):
    if event_type == EventType.START_RUN:
        component.tenseal_context = load_job_local_context(
            component.tenseal_context_file,
            fl_ctx,
        )
    elif event_type == EventType.END_RUN:
        component.tenseal_context = None


class JobLocalHEModelEncryptor(HEModelEncryptor):
    """NVFLARE HE encryptor using a job-local TenSEAL context."""

    def handle_event(self, event_type, fl_ctx):
        _handle_context_event(self, event_type, fl_ctx)

        if event_type == EventType.START_RUN:
            if self.encrypt_layers is None:
                self.encrypt_layers = [True]
                self.logger.info("Encrypting all layers")

class JobLocalHEModelDecryptor(HEModelDecryptor):
    """NVFLARE HE decryptor using a job-local TenSEAL context."""

    def handle_event(self, event_type, fl_ctx):
        _handle_context_event(self, event_type, fl_ctx)

    def decryption(self, params: dict, encrypted_layers: dict, fl_ctx):
        """Decrypt encrypted model variables while preserving unencrypted buffers."""

        if encrypted_layers is None:
            raise ValueError("encrypted_layers is None!")

        # Identify model variables missing from the encryption metadata.
        missing_layers = [
            name for name in params
            if name not in encrypted_layers
        ]

        # Only allow missing BatchNorm tracking counters.
        unexpected_layers = [
            name for name in missing_layers
            if not name.endswith(".num_batches_tracked")
        ]

        if unexpected_layers:
            raise KeyError(
                "Unexpected model variables missing from HE encryption "
                f"metadata: {unexpected_layers}"
            )

        # Mark BatchNorm tracking counters as unencrypted.
        normalized_layers = dict(encrypted_layers)

        for name in missing_layers:
            normalized_layers[name] = False

        # Preserve NVFLARE's original HE decryption implementation.
        return super().decryption(
            params=params,
            encrypted_layers=normalized_layers,
            fl_ctx=fl_ctx,
        )


# class JobLocalHEModelDecryptor(HEModelDecryptor):
#     """NVFLARE HE decryptor using a job-local TenSEAL context."""

#     def handle_event(self, event_type, fl_ctx):
#         _handle_context_event(self, event_type, fl_ctx)


class JobLocalHEModelSerializeFilter(HEModelSerializeFilter):
    """NVFLARE HE persistor filter using a job-local TenSEAL context."""

    def handle_event(self, event_type, fl_ctx):
        _handle_context_event(self, event_type, fl_ctx)


class JobLocalHEModelShareableGenerator(HEModelShareableGenerator):
    """NVFLARE HE shareable generator using a job-local TenSEAL context."""

    def handle_event(self, event_type, fl_ctx):
        _handle_context_event(self, event_type, fl_ctx)
