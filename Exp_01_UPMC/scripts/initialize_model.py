import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from model import SybilNLSTClassifier, load_sybil_encoder, save_initial_state


def main():
    parser = argparse.ArgumentParser(
        description="Initialize the Sybil-pretrained NLST classifier."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "SYBIL_CHECKPOINT",
                str(
                    Path.home()
                    / "model_weights"
                    / "sybil"
                    / "sybil_checkpoints"
                    / "28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt"
                ),
            )
        ),
        help="Pretrained Sybil checkpoint; override with SYBIL_CHECKPOINT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "NLST_INITIAL_MODEL",
                str(
                    Path.home()
                    / "model_weights"
                    / "nlst_federated"
                    / "upmc_initial_model.pt"
                ),
            )
        ),
        help="Output model path; override with NLST_INITIAL_MODEL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Make the newly initialized classification head reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Sybil checkpoint: {args.checkpoint}")
    print(f"Output artifact: {args.output}")

    model = SybilNLSTClassifier(dropout=0.5)

    loaded_tensors = load_sybil_encoder(
        model=model,
        checkpoint_path=args.checkpoint,
    )

    if loaded_tensors <= 0:
        raise RuntimeError("No pretrained encoder tensors were loaded.")

    save_initial_state(
        model=model,
        output_path=args.output,
        loaded_encoder_tensors=loaded_tensors,
    )

    # Reload the saved artifact to validate its structure.
    artifact = torch.load(
        args.output,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "loaded_encoder_tensors",
        "architecture",
        "num_classes",
    }
    missing = required_keys.difference(artifact)
    if missing:
        raise RuntimeError(f"Initial artifact is missing fields: {missing}")

    expected_state = model.state_dict()
    saved_state = artifact["model_state_dict"]

    if expected_state.keys() != saved_state.keys():
        raise RuntimeError("Saved model state keys do not match the model.")

    for name, tensor in expected_state.items():
        if tensor.shape != saved_state[name].shape:
            raise RuntimeError(f"Saved tensor shape mismatch: {name}")

    print(f"Loaded encoder tensors: {loaded_tensors}")
    print(f"Architecture: {artifact['architecture']}")
    print(f"Number of classes: {artifact['num_classes']}")
    print(f"State-dict tensors: {len(saved_state)}")
    print(f"Artifact size: {args.output.stat().st_size / 1024**2:.2f} MiB")
    print("INITIAL MODEL VALIDATION PASSED")


if __name__ == "__main__":
    main()
