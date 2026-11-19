import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tenseal as ts

from nvflare.client.config import TransferType
from nvflare.job_config.api import FilterType
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.in_process_client_api_executor import (
    PTInProcessClientAPIExecutor,
)

from nvflare.app_opt.he.intime_accumulate_model_aggregator import (
    HEInTimeAccumulateWeightedAggregator,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = EXPERIMENT_DIR / "app"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"

CONTROLLER_SOURCE = (
    SCRIPTS_DIR / "he_evaluating_scatter_and_gather.py"
)
HE_COMPONENTS_SOURCE = APP_DIR / "he_components.py"

DEFAULT_MODEL = (
    Path.home()
    / "model_weights"
    / "nlst_federated"
    / "upmc_initial_model.pt"
)

DEFAULT_OUTPUT = (
    Path.home()
    / "nlst_fed_jobs"
    / "Exp_02_UPMC_HE_site_specific_v1"
)

JOB_NAME = "nlst_upmc_moffitt_exp02_he"

SERVER_SITE = "server"
CLIENT_SITES = ("site-1", "external-site-1")

REQUIRED_FILES = (
    "client.py",
    "data.py",
    "model.py",
    "train_utils.py",
)

ENTRYPOINT = '''\
from pathlib import Path
import runpy
import sys

app_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(app_dir))
runpy.run_path(str(app_dir / "client.py"), run_name="__main__")
'''


def load_initial_artifact(path):
    artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(artifact, dict):
        raise ValueError("Initial model artifact must be a dictionary.")

    if "model_state_dict" not in artifact:
        raise ValueError(
            "Initial artifact must contain model_state_dict."
        )

    if artifact.get("num_classes") != 2:
        raise ValueError("Expected a 2-class initial model.")

    if not isinstance(artifact["model_state_dict"], dict):
        raise TypeError("model_state_dict must be a dictionary.")

    return artifact


def validate_he_contexts(server_path, client_path):
    """Validate key separation and client-server HE compatibility."""

    server_context = ts.context_from(server_path.read_bytes())
    client_context = ts.context_from(client_path.read_bytes())

    if server_context.has_secret_key():
        raise RuntimeError(
            "Server context contains a secret key. "
            "Refusing to export it."
        )

    if client_context.has_secret_key() is False:
        raise RuntimeError(
            "Client context does not contain a secret key."
        )

    # Client encrypts model-update values.
    test_values = [1.0, 2.0, 3.0]

    encrypted = ts.ckks_vector(client_context, test_values)

    # Server deserializes and performs encrypted computation.
    server_vector = ts.ckks_vector_from(
        server_context,
        encrypted.serialize(),
    )

    server_result = server_vector + server_vector

    # Client receives and decrypts the server result.
    client_result = ts.ckks_vector_from(
        client_context,
        server_result.serialize(),
    )

    decrypted = client_result.decrypt()

    np.testing.assert_allclose(
        decrypted,
        [2.0, 4.0, 6.0],
        rtol=1e-4,
        atol=1e-4,
    )

    print("HE context validation: PASSED")
    print("  Server context: no secret key")
    print("  Client context: secret key present")
    print("  Client encryption: PASSED")
    print("  Server encrypted computation: PASSED")
    print("  Client decryption: PASSED")



def save_initial_checkpoint(artifact, model, server_app):
    custom_dir = server_app / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = custom_dir / "upmc_initial_model.pt"

    state_dict = {
        key: value.detach().cpu()
        for key, value in artifact["model_state_dict"].items()
    }

    torch.save(state_dict, checkpoint_path)

    saved_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(saved_state, strict=True)

    print(f"Initial checkpoint: {checkpoint_path}")
    print("Initial checkpoint validation: PASSED")


def validate_export(output, artifact, eval_timeout):
    job_dir = output / JOB_NAME

    server_app = job_dir / "app_server"
    site_apps = {
        site: job_dir / f"app_{site}"
        for site in CLIENT_SITES
    }

    # Check server application.
    server_custom = server_app / "custom"

    required_server_files = {
        "he_evaluating_scatter_and_gather.py",
        "he_components.py",
        "server_context.tenseal",
        "upmc_initial_model.pt",
    }

    server_files = {
        path.name
        for path in server_custom.rglob("*")
        if path.is_file()
    }

    missing = required_server_files - server_files

    if missing:
        raise RuntimeError(
            f"Server app is missing files: {sorted(missing)}"
        )

    # Verify server context remains public-only after export.
    exported_server_context = ts.context_from(
        (server_custom / "server_context.tenseal").read_bytes()
    )

    if exported_server_context.has_secret_key():
        raise RuntimeError(
            "Exported server context unexpectedly contains a secret key."
        )

    # Check both site-specific client applications.
    client_context_bytes = []

    for site, app in site_apps.items():
        custom_dir = app / "custom"

        required_client_files = set(REQUIRED_FILES) | {
            "nlst_client_entry.py",
            "he_components.py",
            "client_context.tenseal",
        }

        exported_files = {
            path.name
            for path in app.rglob("*")
            if path.is_file()
        }

        missing = required_client_files - exported_files

        if missing:
            raise RuntimeError(
                f"{site} app is missing files: {sorted(missing)}"
            )

        context_bytes = (
            custom_dir / "client_context.tenseal"
        ).read_bytes()

        client_context_bytes.append(context_bytes)

        context = ts.context_from(context_bytes)

        if not context.has_secret_key():
            raise RuntimeError(
                f"{site} client context does not contain a secret key."
            )

    if client_context_bytes[0] != client_context_bytes[1]:
        raise RuntimeError(
            "The two client applications do not have identical "
            "client contexts."
        )

    # Validate the registered server workflow.
    server_config_path = (
        server_app / "config" / "config_fed_server.json"
    )

    server_config = json.loads(
        server_config_path.read_text(encoding="utf-8")
    )

    workflows = [
        workflow
        for workflow in server_config.get("workflows", [])
        if "HEEvaluatingScatterAndGather" in workflow.get("path", "")
    ]

    if len(workflows) != 1:
        raise RuntimeError(
            "Expected exactly one HEEvaluatingScatterAndGather "
            f"workflow; found {len(workflows)}."
        )

    workflow_args = workflows[0].get("args", {})

    if workflow_args.get("eval_timeout", 3600) != eval_timeout:
        raise RuntimeError(
            "Exported evaluation timeout does not match requested value."
        )

    if workflow_args.get("eval_targets") != list(CLIENT_SITES):
        raise RuntimeError("Unexpected global evaluation targets.")

    components = server_config.get("components", [])

    persistors = [
        component
        for component in components
        if component.get("id") == "persistor"
    ]

    if len(persistors) != 1:
        raise RuntimeError(
            "Expected exactly one server model persistor."
        )

    persistor_args = persistors[0].get("args", {})

    if persistor_args.get("filter_id") != "model_serialize_filter":
        raise RuntimeError(
            "Server persistor is not connected to the HE serialization filter."
        )

    if (
        persistor_args.get("source_ckpt_file_full_name")
        != "upmc_initial_model.pt"
    ):
        raise RuntimeError("Initial checkpoint path is incorrect.")

    print()
    print("HE JOB EXPORT VALIDATION: PASSED")
    print("  Server application: present")
    print("  Server context: public-only")
    print("  Moffitt application: present")
    print("  UPMC application: present")
    print("  Client contexts: secret key present")
    print("  Client contexts: identical")
    print("  HE evaluation controller: registered once")
    print("  HE persistor filter: registered")
    print("  Initial checkpoint: present")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the site-specific Exp_02_UPMC HE job. "
            "Exports only; does not modify startup kits or submit jobs."
        )
    )

    parser.add_argument(
        "--initial-model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--server-context",
        type=Path,
        required=True,
        help="Public-only server_context.tenseal",
    )

    parser.add_argument(
        "--client-context",
        type=Path,
        required=True,
        help="Client context containing the secret key",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eval-timeout", type=int, default=3600)

    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1.")

    if args.eval_timeout <= 0:
        parser.error("--eval-timeout must be positive.")

    initial_model_path = args.initial_model.expanduser().resolve()
    server_context_path = args.server_context.expanduser().resolve()
    client_context_path = args.client_context.expanduser().resolve()

    for path in (
        initial_model_path,
        server_context_path,
        client_context_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for name in REQUIRED_FILES:
        if not (APP_DIR / name).is_file():
            raise FileNotFoundError(APP_DIR / name)

    if not CONTROLLER_SOURCE.is_file():
        raise FileNotFoundError(CONTROLLER_SOURCE)

    if not HE_COMPONENTS_SOURCE.is_file():
        raise FileNotFoundError(HE_COMPONENTS_SOURCE)

    output = args.output.expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. "
            "Choose a new output directory; existing exports "
            "will not be overwritten."
        )

    # Validate the HE keys before constructing the job.
    validate_he_contexts(
        server_context_path,
        client_context_path,
    )

    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR))

    from model import SybilNLSTClassifier
    from he_evaluating_scatter_and_gather import (
        HEEvaluatingScatterAndGather,
    )

    artifact = load_initial_artifact(initial_model_path)

    model = SybilNLSTClassifier(dropout=0.5)
    model.load_state_dict(
        artifact["model_state_dict"],
        strict=True,
    )
    model.cpu()

    # Create a persistor that uses the HE serialization filter.
    persistor = PTFileModelPersistor(
        model=model,
        source_ckpt_file_full_name="upmc_initial_model.pt",
        filter_id="model_serialize_filter",
    )

    job = BaseFedJob(
        name=JOB_NAME,
        min_clients=2,
    )

    # Register model, persistor, and locator.
    model_ids = job.to_server(
        PTModel(
            model=model,
            persistor=persistor,
        )
    )

    persistor_id = model_ids["persistor_id"]

    # Server-side HE components.
    job.to_server(
        JobLocalHEModelSerializeFilter(
            tenseal_context_file="server_context.tenseal",
        ),
        id="model_serialize_filter",
    )

    shareable_generator_id = job.to_server(
        JobLocalHEModelShareableGenerator(
            tenseal_context_file="server_context.tenseal",
        ),
        id="shareable_generator",
    )

    aggregator_id = job.to_server(
        HEInTimeAccumulateWeightedAggregator(
            expected_data_kind=__import__(
                "nvflare.apis.dxo",
                fromlist=["DataKind"],
            ).DataKind.WEIGHT_DIFF,
            weigh_by_local_iter=False,
        ),
        id="aggregator",
    )

    # Custom HE-aware controller with post-aggregation evaluation.
    controller = HEEvaluatingScatterAndGather(
        min_clients=2,
        num_rounds=args.rounds,
        persistor_id=persistor_id,
        aggregator_id=aggregator_id,
        shareable_generator_id=shareable_generator_id,
        eval_targets=list(CLIENT_SITES),
        eval_timeout=args.eval_timeout,
    )

    job.to_server(controller)

    # Stage server-only files and the public context.
    job.add_file_to_server(
        str(CONTROLLER_SOURCE),
        app_folder_type="custom",
    )

    job.add_file_to_server(
        str(HE_COMPONENTS_SOURCE),
        app_folder_type="custom",
    )

    job.add_file_to_server(
        str(server_context_path),
        app_folder_type="custom",
    )

    # Build each client application separately.
    with tempfile.TemporaryDirectory(
        prefix="nlst_exp02_he_"
    ) as temp_dir:

        staging = Path(temp_dir)

        for name in REQUIRED_FILES:
            shutil.copy2(
                APP_DIR / name,
                staging / name,
            )

        entrypoint = staging / "nlst_client_entry.py"
        entrypoint.write_text(
            ENTRYPOINT,
            encoding="utf-8",
        )

        for site in CLIENT_SITES:

            # Client source files.
            for name in (*REQUIRED_FILES, "nlst_client_entry.py"):
                job.to(
                    str(staging / name),
                    site,
                )

            # Client-side HE component module and secret context.
            job.add_file_to(
                str(HE_COMPONENTS_SOURCE),
                site,
                app_folder_type="custom",
            )

            job.add_file_to(
                str(client_context_path),
                site,
                app_folder_type="custom",
            )

            # Client training executor.
            executor = PTInProcessClientAPIExecutor(
                task_script_path="nlst_client_entry.py",
                params_transfer_type=TransferType.FULL,
            )

            job.to(executor, site)

            # Decrypt incoming train and global-evaluation models.
            job.to(
                JobLocalHEModelDecryptor(
                    tenseal_context_file="client_context.tenseal",
                ),
                site,
                tasks=["train", "evaluate_global"],
                filter_type=FilterType.TASK_DATA,
            )

            # Encrypt outgoing training updates only.
            # Evaluation acknowledgments remain unencrypted.
            job.to(
                JobLocalHEModelEncryptor(
                    tenseal_context_file="client_context.tenseal",
                    weigh_by_local_iter=True,
                ),
                site,
                tasks=["train"],
                filter_type=FilterType.TASK_RESULT,
            )

        # Create parent directory and export only.
        output.parent.mkdir(parents=True, exist_ok=True)

        job.export_job(str(output))

    # Stage and strictly validate the initial model checkpoint.
    server_app = output / JOB_NAME / "app_server"

    save_initial_checkpoint(
        artifact=artifact,
        model=model,
        server_app=server_app,
    )

    validate_export(
        output=output,
        artifact=artifact,
        eval_timeout=args.eval_timeout,
    )

    print()
    print("========== EXP_02_UPMC HE EXPORT ==========")
    print(f"Export directory: {output}")
    print(f"Job name: {JOB_NAME}")
    print("Clients: site-1, external-site-1")
    print(f"Training rounds: {args.rounds}")
    print(f"Evaluation timeout: {args.eval_timeout}")
    print(f"Persistor ID: {persistor_id}")
    print("Server context: public-only")
    print("Client contexts: secret-key contexts")
    print()
    print("JOB EXPORTED AND VALIDATED.")
    print("No startup files were modified.")
    print("No job was submitted or executed.")


if __name__ == "__main__":
    # Imported here to keep the component imports explicit.
    from he_components import (
        JobLocalHEModelDecryptor,
        JobLocalHEModelEncryptor,
        JobLocalHEModelSerializeFilter,
        JobLocalHEModelShareableGenerator,
    )

    main()
