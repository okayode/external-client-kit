from pathlib import Path

import torch
import torch.nn as nn
import torchvision


class SybilR3D18(nn.Module):
    """R3D-18 encoder with the original Sybil image_encoder structure."""

    def __init__(self):
        super().__init__()

        encoder = torchvision.models.video.r3d_18(weights=None)

        self.image_encoder = nn.Sequential(
            *list(encoder.children())[:-2]
        )

    def forward(self, x):
        return self.image_encoder(x)


class SybilNLSTClassifier(nn.Module):
    """
    Binary NLST classifier using a Sybil-pretrained R3D-18 encoder.

    Input:  [B, 3, D, H, W]
    Output: [B, 2] class logits
    """

    def __init__(self, dropout=0.5):
        super().__init__()

        self.backbone = SybilR3D18()

        self.head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        features = self.backbone(x)

        avg = features.mean(dim=(2, 3, 4))
        mx = features.amax(dim=(2, 3, 4))

        features = torch.cat([avg, mx], dim=1)

        return self.head(features)


def load_sybil_encoder(model, checkpoint_path):
    """
    Load the pretrained Sybil encoder into a classifier.

    Uses the checkpoint structure and key mapping established in
    Sybil_1_Resnet18_CNN_NLST.py. The checkpoint must be trusted because
    weights_only=False permits Python object deserialization.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Sybil checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(
            "Expected the Sybil checkpoint to contain 'state_dict'."
        )

    state = checkpoint["state_dict"]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value

    own = model.backbone.image_encoder.state_dict()
    loaded = {}

    for key, value in cleaned.items():
        if not key.startswith("image_encoder."):
            continue

        new_key = key[len("image_encoder."):]

        if (
            new_key in own
            and torch.is_tensor(value)
            and own[new_key].shape == value.shape
        ):
            loaded[new_key] = value

    missing = [key for key in own if key not in loaded]

    print(f"Sybil encoder tensors matched: {len(loaded)} / {len(own)}")

    if missing:
        raise RuntimeError(
            "Sybil encoder checkpoint mismatch. "
            f"Missing or incompatible keys ({len(missing)}): "
            f"{missing[:15]}"
        )

    model.backbone.image_encoder.load_state_dict(
        loaded,
        strict=True,
    )

    print("Sybil R3D-18 encoder loaded successfully.")
    return len(loaded)


def save_initial_state(model, output_path, loaded_encoder_tensors):
    """Save the initial model state for federated server initialization."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loaded_encoder_tensors": int(loaded_encoder_tensors),
            "architecture": "SybilNLSTClassifier-R3D18",
            "num_classes": 2,
        },
        output_path,
    )

    print(f"Saved initial model state: {output_path}")
