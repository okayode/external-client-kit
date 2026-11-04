# External NVFlare Client Kit

Reusable NVIDIA FLARE 2.7.3 client kit for connecting an external institution to a central federated learning server.

The primary external-institution workflow documented here is the `Exp_01_UPMC/`. The repository also retains the original CIFAR-10 demonstration, which is documented in [Appendix A](#appendix-a-archived-cifar-10-demonstration).

Raw institutional data remain on the participating institution's infrastructure. UPMC uses its own locally stored `NLST_UPMC` dataset.

## 1. Joint Moffitt-UPMC architecture

The coordinated NLST experiment uses two NVFlare clients and the existing AWS EC2 aggregation server:

| Participant | NVFlare identity | Compute and data |
|---|---|---|
| Moffitt | `site-1` | Moffitt compute and local `NLST_Moffitt` data |
| UPMC | `external-site-1` | UPMC compute and local `NLST_UPMC` data |
| Aggregation server | Server | Existing AWS EC2 NVFlare server |

**Important distinction:** UPMC starts and keeps its NVFlare client running. The coordinated federated job is exported and submitted to the central NVFlare server by the authorized job operator. Starting `startup/start.sh` does not itself submit or launch a training job.

## 2. Software requirements

The documented validated environment is:

```text
NVIDIA NVFlare 2.7.3
Python 3.11
PyTorch 2.14.0
torchvision 0.29.0
NumPy 2.4.6
```

Activate the UPMC Python environment that contains the required NVFlare version and dependencies. Install the repository dependencies from the kit root when needed:

```bash
cd ~/external-client-kit
python -m pip install -r requirements.txt
```

Do not replace or recreate an existing working NVFlare environment solely for this update. Confirm the active environment before starting the client:

```bash
python --version
python -m pip show nvflare torch torchvision numpy
```

## 3. Get the repository from GitHub or update an existing clone

UPMC should update its existing checkout.

```bash
cd "$HOME/external-client-kit"

# Check the current branch and whether tracked files have local changes.
git branch --show-current
git status --short

# Fetch and fast-forward the existing main branch.
git pull --ff-only origin main
```

The expected shared branch for this workflow is `main`. If the checkout is on another branch, or `git status --short` shows modified tracked files, do not force the update. Preserve or reconcile those changes with the repository maintainer first. **Do not use `git reset --hard` as an update shortcut.**

The UPMC dataset is stored outside the repository. The machine-specific file `Exp_01_UPMC/config/client_config.json` is ignored by Git and should remain local; do not replace it with the tracked example during an ordinary update.

After updating, confirm the checkout:

```bash
git log -1 --oneline
git status --short
```

## 4. Configure UPMC's local NLST dataset

The UPMC NLST data are not part of the Git repository. UPMC must already have its own dataset on storage accessible from the compute node that runs the NVFlare client.

The configured dataset directory must contain the two patient-list CSV files and the volume files referenced by `volT0`:

```text
NLST_UPMC/
├── NLST-T0-pids-train.csv
├── NLST-T0-pids-test.csv
└── ... volume files referenced by the CSVs ...
```

Create the local configuration directory and copy the tracked example **only if a local configuration does not already exist**:

```bash
cd "$HOME/external-client-kit"
mkdir -p Exp_01_UPMC/config

if [ ! -f Exp_01_UPMC/config/client_config.json ]; then
    cp Exp_01_UPMC/app/client_config.upmc.example.json \
       Exp_01_UPMC/config/client_config.json
else
    echo "Existing UPMC local configuration preserved."
fi
```

Edit `Exp_01_UPMC/config/client_config.json`. Set `data_root` to the actual UPMC dataset directory. For example, if the dataset is in the home directory as `~/NLST_UPMC`, use:

```json
{
  "data_root": "~/NLST_UPMC",
  "batch_size": 2,
  "val_fraction": 0.15,
  "seed": 42,
  "num_workers": 4,
  "local_epochs": 1,
  "total_rounds": 12,
  "lr_backbone": 0.000002,
  "lr_head": 0.0005,
  "weight_decay": 0.005,
  "warmup_epochs": 2,
  "max_grad_norm": 5.0,
  "threshold": 0.5,
  "amp": true
}
```

The example above retains the experiment settings in the tracked example configuration. Change `data_root` if UPMC stores the dataset elsewhere. Do not put a Moffitt path in this file.

Validate the JSON and required CSV files. Adjust the `DATA_ROOT` below if the configured location is different:

```bash
python -m json.tool Exp_01_UPMC/config/client_config.json >/dev/null

python - <<'PY'
from pathlib import Path
root = (Path.home() / "NLST_UPMC").resolve()
required = ["NLST-T0-pids-train.csv", "NLST-T0-pids-test.csv"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Missing files under {root}: {missing}")
print(f"UPMC dataset directory checks passed: {root}")
PY
```

If the dataset is not at `~/NLST_UPMC`, update the `root` in the validation snippet to match `data_root` in the JSON file.

The local configuration file is:

```text
Exp_01_UPMC/config/client_config.json
```

The tracked example is:

```text
Exp_01_UPMC/app/client_config.upmc.example.json
```

The local file is excluded by `.gitignore`. Keep it on UPMC's system; do not commit it or change the tracked example to contain an institution-specific path. The UPMC client reads the local configuration through `NLST_CLIENT_CONFIG`; its `data_root` setting is used to set `NLST_DATA_ROOT` before the data loader is imported.

The client validates the CSV files and checks that the volume paths referenced in the CSVs exist when it initializes the data loaders. A successful JSON check alone does not validate all volume files.

## 5. Start the UPMC NVFlare client

Run the client from the repository root, using the UPMC environment with NVFlare 2.7.3 active:

```bash
cd "$HOME/external-client-kit"
./startup/start.sh
```

The startup configuration uses the provisioned NVFlare identity:

```text
external-site-1
```

The UPMC startup script sets `EXTERNAL_CLIENT_KIT_ROOT` and, for the NLST experiment, points `NLST_CLIENT_CONFIG` to the UPMC-local configuration file. The NVFlare client then connects to the configured central server and waits for tasks.

Use UPMC's normal cluster/process-management policy for long-running jobs. Do not start multiple copies of the same NVFlare client identity at the same time. The startup script uses a `pid.fl` file to track its FL process; do not delete that file while the client is still running.

### Network connectivity

The UPMC compute environment must be able to reach the central NVFlare server address and client communication port configured in the supplied startup files. The current deployment uses AWS EC2 port `8002` for the FL client connection. The server's administrative/overseer port is `8003`; UPMC should use the access arrangement authorized by the server administrator rather than changing the startup configuration independently.

## 6. How the joint federated job starts

UPMC does **not** start the coordinated training job by running a separate local training script. The sequence is:

1. UPMC configures its local dataset and starts `./startup/start.sh`.
2. Moffitt starts its `site-1` NVFlare client with its own Moffitt-local dataset configuration.
3. The authorized central job operator confirms that both client identities are connected to the existing AWS server.
4. The central operator submits the coordinated NLST job to the server.
5. NVFlare delivers the task to both clients. UPMC trains using its local `NLST_UPMC` data; Moffitt trains using its local `NLST_Moffitt` data.
6. The clients return model updates for server aggregation. The coordinated workflow also requests global-model evaluation at the participating sites.

The expected client identities for the two-site experiment are:

```text
site-1          Moffitt
external-site-1 UPMC
```

UPMC should leave the client running while the server operator submits and executes the coordinated job.

## 7. Data handling and privacy

UPMC's raw NLST data remain on UPMC-managed storage. Moffitt's raw NLST data remain on Moffitt-managed storage. Neither institution needs to mount the other institution's dataset. The client application loads local volumes and returns model information required by the federated-learning protocol; it does not package the raw CSVs or CT volumes into the shared job.

Patient-level predictions and local evaluation outputs are written locally by the client application. Follow institutional data-use, security, and logging requirements when retaining or sharing those outputs.

## 8. Validation status and expected checks

### Existing CIFAR-10 demonstration

The original external client kit has an existing CIFAR-10 demonstration validation record. Its details are retained in [Appendix A](#appendix-a-archived-cifar-10-demonstration).

### NLST Moffitt-UPMC practice run

The 12-round practice logs showed both `site-1` and `external-site-1` training, receiving global evaluation tasks, sending acknowledgements, and reaching normal run termination. In that practice run, the UPMC data directory was visible on the Moffitt cluster at `/mnt/weka/home/4476224/NLST_UPMC`.

That test verified the two-client NVFlare workflow in the Moffitt cluster environment. It did **not** demonstrate that UPMC had run the client on its own compute infrastructure. Independent UPMC participation is validated only after UPMC pulls the portable client-kit update, configures its own local dataset, connects from its own compute environment, and completes a joint job with Moffitt through the AWS server.

Before the first independent UPMC run, confirm all of the following:

- UPMC's `external-site-1` client connects to the configured AWS server.
- The log reports UPMC's actual local dataset directory, not a Moffitt `/mnt/weka/...` path.
- Both required CSV files and the referenced CT volume files are accessible on UPMC compute.
- The server submits the coordinated two-client job and both clients participate in the training rounds.
- Both clients receive the global evaluation task and complete the run without an unhandled exception.


### Generating federated-learning performance metrics

After an NVFlare NLST run has produced its `log.txt`, generate performance plots and a round-by-round metrics table with `FL_NLST_metrics.py`, located in the `external-client-kit/` repository root. From the repository root, pass the run directory name (for example, the NVFlare run UUID) as the script's command-line argument. The script reads `log.txt` from that directory; no run-specific paths need to be edited or hard-coded in the script.

```bash
cd "$HOME/external-client-kit"

# Replace the example UUID with the directory for the run you want to analyze.
python FL_NLST_metrics.py 1df6f954-7f53-4239-b4b7-95517c27b4ec
```

The script creates the following files inside the specified run directory:

```text
<run-directory>/
├── log.txt
├── validation_metrics.png
├── heldout_test_metrics.png
└── metrics_summary.csv
```

`validation_metrics.png` summarizes validation loss, accuracy, ROC-AUC, PR-AUC, and elapsed time from the start of the round's training task through completion of validation, plotted against federated round. `heldout_test_metrics.png` summarizes held-out test ROC-AUC, PR-AUC, balanced accuracy, the ROC-AUC 95% confidence interval (lower and upper bounds), and elapsed time for global-model test evaluation, also plotted against federated round. `metrics_summary.csv` contains the extracted metrics for each round. Time values are reported in minutes and are calculated from timestamps in the NVFlare log; they represent the logged task intervals, not model-training time alone. If the required Python packages are not already available in the active environment, install them with `python -m pip install matplotlib pandas numpy`.

## 9. Repository and credential security

This repository contains institution-specific NVFlare credentials in `startup/` and is intended to remain private and access-controlled. Grant access only to authorized collaborators. Do not publish the repository or its credentials in a public location.

The `external-site-1` credentials are for the institution authorized to use that identity. For additional external institutions, provision separate NVFlare client identities and credentials rather than reusing `external-site-1`.

---

# Appendix A. Archived CIFAR-10 demonstration

This section preserves the original CIFAR-10 demonstration instructions. The CIFAR-10 application, demonstration data, federated split, scripts, and supporting repository content remain in the repository; they are not replaced by the UPMC NLST application.

## A1. CIFAR-10 repository content

The original demonstration-related paths include:

```text
external-client-kit/
├── app/
│   ├── client.py
│   ├── model.py
│   ├── train_utils.py
│   └── data/
├── config/
├── data/
│   └── cifar10/
├── local/
├── scripts/
│   ├── prepare_data.sh
│   └── prepare_split.sh
├── splits/
│   └── external-site-1.npy
└── src/
```

The repository also retains the NVFlare `startup/` directory, `.gitignore`, `README.md`, and `requirements.txt` described in the main sections above.

## A2. Prepare CIFAR-10 data

The repository already contains the CIFAR-10 demonstration dataset:

```text
data/cifar10/
```

An authorized institution cloning the repository does not need to run `prepare_data.sh` to reproduce the current demonstration.

For a new CIFAR-10 dataset, prepare the data with:

```bash
./scripts/prepare_data.sh
```

An institution may instead use its own locally stored dataset if the client application and data utilities support that format. Raw institutional data remain on the institution's infrastructure.

## A3. Prepare the CIFAR-10 federated data split

The repository already contains the validated demonstration split:

```text
splits/external-site-1.npy
```

An authorized institution cloning the repository does not need to regenerate the split to reproduce the current demonstration.

For a new dataset or different federated partition, generate a site-specific split with:

```bash
./scripts/prepare_split.sh <site_name> <site_index> <num_sites> [alpha] [seed]
```

Example:

```bash
./scripts/prepare_split.sh external-site-1 1 2 0.5 0
```

The institution is responsible for determining how its local data are partitioned for federated learning.

The UPMC NLST application does not use this CIFAR-10 `.npy` split. It uses the train and test patient lists in UPMC's local NLST dataset and creates a local training/validation split from the training CSV using `val_fraction` and `seed`.

## A4. Original CIFAR-10 workflow

```text
Clone client kit
        ↓
Install requirements
        ↓
Use included data and federated split
        ↓
Start NVFlare client
        ↓
Central server sends FL job
        ↓
Client trains using local data
        ↓
Client sends model update
        ↓
Server aggregates client updates
        ↓
Updated global model is returned
```

## A5. Original CIFAR-10 validation record

The original README recorded the following validated configuration:

```text
NVFlare: 2.7.3
External client: external-site-1
Internal client: site-1
Aggregation server: AWS EC2
Server ports: 8002, 8003
FL algorithm: FedAvg
Test: 40 federated rounds
Status: FINISHED:COMPLETED
```

The validated architecture used one internal Moffitt client, one external client, and the AWS EC2 server as the aggregation server.