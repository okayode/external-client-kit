# External NVFlare Client Kit

Reusable NVIDIA FLARE 2.7.3 client kit for connecting an external institution to a central federated learning server.

The repository contains the client application, model, training utilities, CIFAR-10 demonstration data, federated split, NVFlare startup configuration, and client credentials. Raw institutional data remain on the participating institution's infrastructure and are not transmitted to the server.

## 1. Architecture

```text
Repository Structure

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
├── src/
├── startup/
│   ├── client.crt
│   ├── client.key
│   ├── fed_client.json
│   ├── rootCA.pem
│   ├── signature.json
│   ├── start.sh
│   └── sub_start.sh
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. Software Requirements

Validated environment:
```
NVIDIA NVFlare 2.7.3
Python 3.11
PyTorch 2.14.0
torchvision 0.29.0
NumPy 2.4.6
```
Install dependencies with:
```
pip install -r requirements.txt
```

## 3. Prepare Local Data

The repository already contains the CIFAR-10 demonstration dataset:
```
data/cifar10/
```
Therefore, an authorized institution cloning this repository does not need to run prepare_data.sh to reproduce the current demonstration.

For a new CIFAR-10 dataset, the data can be prepared with:
```
./scripts/prepare_data.sh
```
An institution may instead use its own locally stored dataset if the client application and data utilities support that format.

Raw institutional data remain on the institution's infrastructure.

## 4. Prepare the Federated Data Split

The repository already contains the validated demonstration split:
```
splits/external-site-1.npy
```
Therefore, an authorized institution cloning this repository does not need to regenerate the split to reproduce the current demonstration.

For a new dataset or different federated partition, generate a new site-specific split with:
```
./scripts/prepare_split.sh <site_name> <site_index> <num_sites> [alpha] [seed]
```
Example:
```
./scripts/prepare_split.sh external-site-1 1 2 0.5 0
```
The institution is responsible for determining how its local data are partitioned for federated learning.

## 5. NVFlare Startup Configuration

The startup/ directory contains the authenticated NVFlare client configuration:
```
startup/
├── client.crt
├── client.key
├── fed_client.json
├── rootCA.pem
├── signature.json
├── start.sh
└── sub_start.sh
```
These files provide the client identity and configuration required to establish the secure connection with the central NVFlare server.

The current validated client identity is:
```
external-site-1
```
The included startup configuration can be used as-is by the authorized institution for the current demonstration.

For a different institution, a unique NVFlare client identity and credentials should be provisioned rather than reusing external-site-1.

Because this repository contains client credentials, it must remain private and access-controlled.

## 6. Client Application and Local Data

NVFlare deploys the application code from:
```
app/
```
The external client's local data and split are not packaged into the NVFlare job.

At runtime, ```startup/sub_start.sh``` sets:

EXTERNAL_CLIENT_KIT_ROOT

to the permanent client-kit directory.

The client application uses this location to access:
```
data/cifar10/
splits/
```
This separates the permanent external client kit from the application files temporarily deployed by NVFlare.

## 7. Start the Client

From the client-kit root:
```
cd ~/external-client-kit/
./startup/start.sh
```
The client connects to the configured central NVFlare server and waits for federated learning jobs.

For the current authorized institution, no additional data or split preparation is required to run the demonstrated CIFAR-10 configuration.

## 8. Federated Learning Workflow
```
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
Only model information required by the federated-learning protocol is exchanged. Raw local training data remain at the participating institution.

## 9. Validated Configuration

The current kit has been validated with:
```
NVFlare: 2.7.3
External client: external-site-1
Internal client: site-1
Aggregation server: AWS EC2
Server ports: 8002, 8003
FL algorithm: FedAvg
Test: 40 federated rounds
Status: FINISHED:COMPLETED
```
The validated architecture uses one internal Moffitt client, one external client, and the AWS EC2 server as the aggregation server.

## 10. Repository Security

This repository contains institution-specific NVFlare credentials in startup/ and is intended to remain private.

Access should be granted only to authorized collaborators.

For additional external institutions, provision separate NVFlare client credentials and identities rather than reusing external-site-1.
