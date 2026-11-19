from pathlib import Path
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# DATA CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "NLST_DATA_ROOT",
        str(Path.home() / "NLST_UPMC")
    )
).expanduser().resolve()

TRAIN_CSV = DATA_ROOT / "NLST-T0-pids-train.csv"
TEST_CSV = DATA_ROOT / "NLST-T0-pids-test.csv"

LABEL_COLS = [f"labelT{i}" for i in range(8)]

EXPECTED_SHAPE = (3, 200, 256, 256)

BATCH_SIZE = 2
NUM_WORKERS = 4

SHIFT = (4, 8, 8)


def prepare_dataframe(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"pid", "volT0", *LABEL_COLS}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    df = df.copy()
    df = df.dropna(subset=["pid", "volT0"])

    if df["pid"].duplicated().any():
        duplicates = (
            df.loc[df["pid"].duplicated(), "pid"]
            .tolist()
        )

        raise ValueError(
            f"Duplicate PIDs in {csv_path}: {duplicates[:10]}"
        )

    labels = (
        df[LABEL_COLS]
        .fillna(0)
        .astype(int)
        .max(axis=1)
    )

    if not labels.isin([0, 1]).all():
        raise ValueError(
            "Derived labels must be binary (0 or 1)."
        )

    df["label"] = labels.astype(int)

    def resolve_volume_path(value):

        path = Path(str(value)).expanduser()

        if not path.is_absolute():
            path = DATA_ROOT / path

        return str(path.resolve())

    df["volume_path"] = df["volT0"].map(
        resolve_volume_path
    )

    missing_files = [
        p
        for p in df["volume_path"]
        if not Path(p).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} volume files are missing. "
            f"Examples: {missing_files[:3]}"
        )

    return df.reset_index(drop=True)


def make_train_validation_split(
    train_df,
    val_fraction=0.15,
    seed=42
):

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_df["label"],
    )

    return (
        train_part.reset_index(drop=True),
        val_part.reset_index(drop=True),
    )


def load_volume(path):

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(obj, dict):

        if "volume" not in obj:
            raise ValueError(
                f"No 'volume' entry in {path}"
            )

        x = torch.as_tensor(obj["volume"]).float()

    else:
        x = torch.as_tensor(obj).float()

    if x.ndim == 5:

        if x.shape[0] != 1:
            raise ValueError(
                f"Unexpected batch dimension in {path}: "
                f"{tuple(x.shape)}"
            )

        x = x[0]

    if tuple(x.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected [3, 200, 256, 256], "
            f"got {tuple(x.shape)} for {path}"
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"Non-finite values in volume: {path}"
        )

    return x


def random_shift(x):

    pad = [
        value
        for shift in reversed(SHIFT)
        for value in (shift, shift)
    ]

    xp = F.pad(x, pad)

    offsets = [
        random.randint(0, 2 * shift)
        for shift in SHIFT
    ]

    depth, height, width = x.shape[1:]

    return xp[
        :,
        offsets[0]:offsets[0] + depth,
        offsets[1]:offsets[1] + height,
        offsets[2]:offsets[2] + width,
    ]


class NLSTVolumeDataset(Dataset):

    def __init__(self, dataframe, augment=False):

        self.df = dataframe.reset_index(
            drop=True
        ).copy()

        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        x = load_volume(row["volume_path"])

        y = int(row["label"])

        if self.augment:

            if random.random() < 0.5:
                x = x.flip(3)

            x = random_shift(x)

            if random.random() < 0.5:

                x = (
                    x * (
                        1.0 + 0.03 * torch.randn(1)
                    )
                    + 0.03 * torch.randn(1)
                )

        return x, torch.tensor(
            y,
            dtype=torch.long
        )


def make_loaders(
    batch_size=BATCH_SIZE,
    val_fraction=0.15,
    seed=42,
    num_workers=NUM_WORKERS,
):

    train_df = prepare_dataframe(TRAIN_CSV)

    test_df = prepare_dataframe(TEST_CSV)

    overlap = (
        set(train_df["pid"])
        & set(test_df["pid"])
    )

    if overlap:
        raise ValueError(
            "Train/test PID overlap detected: "
            f"{list(overlap)[:10]}"
        )

    local_train, local_val = (
        make_train_validation_split(
            train_df,
            val_fraction=val_fraction,
            seed=seed,
        )
    )

    print(
        f"Training examples:   {len(local_train)}"
    )

    print(
        f"Validation examples: {len(local_val)}"
    )

    print(
        f"Held-out test cases: {len(test_df)}"
    )

    for name, frame in (
        ("train", local_train),
        ("validation", local_val),
        ("test", test_df),
    ):

        counts = (
            frame["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"{name} label counts (0/1): {counts}"
        )

    persistent = num_workers > 0

    train_loader = DataLoader(
        NLSTVolumeDataset(
            local_train,
            augment=True
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        NLSTVolumeDataset(
            local_val,
            augment=False
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        NLSTVolumeDataset(
            test_df,
            augment=False
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )