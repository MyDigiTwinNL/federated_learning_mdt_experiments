# MyDigiTwin — Federated Learning for ASCVD Risk Prediction  
**Rotterdam Study × Lifelines Biobank using Vantage6**

This repository documents the infrastructure and workflow used in the **MyDigiTwin (MDT)** project for privacy-preserving federated learning (FL) to predict **atherosclerotic cardiovascular disease (ASCVD)** risk.  

The project integrates data from:

- **Rotterdam Study**
- **Lifelines Biobank**

using the **Vantage6 federated learning framework** deployed on **SURF Research Cloud (SRC)** infrastructure.

---

# 1. Data Governance & Access

Anyone who plans to work on this project — especially those who need access to the **SURF Research Cloud (SRC)** — must first ensure that they are authorized to access the underlying datasets.

## Data Access Requirements

- Access to **Rotterdam Study (RS)** data is required.
- Please contact the **Principal Investigators (PIs)** of the project to discuss eligibility and permissions.
- Even for practice purposes, RS data access approval is still required.

Within SRC:

- A workspace exists containing a **pseudo-split Rotterdam Study dataset** for federated learning practice.
- This environment allows users to familiarize themselves with the MDT pipeline before working with production collaborations.

## Access to SURF Research Cloud Portal

To obtain access to the SRC portal and MDT project workspaces, please contact:

- **Djura Smits** — d.smits@esciencecenter.nl  
- **Hyunho Mo** — h.mo@erasmusmc.nl  

These contacts can grant permissions to the SRC project and relevant workspaces.

## Lifelines Node Management

For the MyDigiTwin project, the **Lifelines federated node** is managed externally by:

- **Bolmer, B.R.J.** — b.r.j.bolmer@lifelines.nl  

This node is hosted at **UMCG** and maintained by the Lifelines data management team.

---

# 2. Vantage6 Server for the MyDigiTwin Project

> This section is optional if you do not need to create a new user or collaboration.

The MDT project uses a dedicated Vantage6 server instance for orchestrating federated learning tasks.

## Server Access

Vantage6 web interface: https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl/#/auth/login



To obtain a user account, contact:

- **Djura Smits** — d.smits@esciencecenter.nl  

The server UI allows you to:

- Create and manage user accounts
- Monitor collaborations
- Set nodes online/offline
- Inspect tasks and results

---

# 3. Collaborations Used in the MDT Project

Two main collaborations are configured on the Vantage6 server.

---

## 3.1 `fl_mock_rs` — Rotterdam Study Mock Federated Environment

This collaboration is designed for **practice and development**.

Dataset:

- Non-imaging Rotterdam Study cohorts:
  - RS-I-3
  - RS-II-1
  - RS-III-1

Setup:

- Dataset randomly split into two halves
- Stored in separate SRC workspaces to simulate institutions

Nodes:

| Purpose | Workspace |
|---------|-----------|
| Client Node 1 | RS_MOCK_FIRST |
| Client Node 2 | RS_MOCK_SECOND |
| Aggregation Node | RS_MOCK_AGGR |

Recommended for:

- Learning Vantage6
- Debugging pipelines
- Training new users

---

## 3.2 `lifelines-poc-collaboration` — Production MDT Setup

Main collaboration combining:

- Rotterdam Study (RS-I-3, RS-II-1, RS-III-1)
- Lifelines Biobank

Nodes:

| Purpose | Name |
|---------|------|
| Rotterdam Study Node | RS_FL_node_cpu |
| Lifelines Node | Lifelines-A (UMCG VM) |
| Aggregation Node | MDT_FL_AGGR |

Used for:

- Federated ASCVD prediction experiments
- Proof-of-concept studies
- Publications

---

# 4. Docker Container — Federated Learning Implementation

The MDT federated learning implementation is packaged as a Docker container running inside Vantage6 nodes.

## Source Repository
https://github.com/MyDigiTwinNL/FedAvg_vantage6/tree/main


## Docker Image Registry

Container: federated_cvdm_training_poc

Latest versions: https://github.com/mydigitwinnl/FedAvg_vantage6/pkgs/container/federated_cvdm_training_poc



---

## Core Implementation Files

### `central_ci.py`

- Controls federated training loop
- Creates subtasks per node
- Collects model weights
- Performs weighted averaging
- Produces global model updates

Central orchestration logic.

### `partial_risk_prediction.py`

Executed on each node:

- Load local dataset
- Preprocess data
- Train local model
- Return weights

Local training logic.

---

## Build or Update Docker Image

```bash
docker build -t ghcr.io/mydigitwinnl/federated_cvdm_training_poc:<version_name> .
docker push ghcr.io/mydigitwinnl/federated_cvdm_training_poc:<version_name>
```
Pull Docker Image on Node
```
docker pull ghcr.io/mydigitwinnl/federated_cvdm_training_poc:<version_name>
```


# 5. Setting Up a Vantage6 Node on SURF Workspace

Used when:

- Deploying a new node
- Using custom datasets
- Creating new organizations

Important: MDT uses vantage6==4.8.1
All nodes must match server version.

If you need to setup a v6 node for your data at a SRC workspace, please follow the commands below:

Create System User
```
sudo useradd -m -s /bin/bash v6 || true
sudo mkdir -p /opt/v6-nodes
sudo chown -R v6:v6 /opt/v6-nodes
sudo chmod 700 /opt/v6-nodes
sudo usermod -aG docker v6
sudo usermod -aG sudo v6
sudo passwd v6
sudo -iu v6
cd /opt/v6-nodes

```

Python environment
```
python3 -m venv rs_node_venv
source rs_node_venv/bin/activate

pip install --upgrade pip
pip install "vantage6==4.8.1"
pip install -U "questionary==2.1.1."
pip install -U "prompt_toolkit<4"
pip install "setuptools<81"
```

Data Directory

```
mkdir -p /opt/v6-nodes/node_rs/data
chmod 700 /opt/v6-nodes/node_rs/data
sudo cp /home/hmo/rs_data/data.csv /opt/v6-nodes/node_rs/data
mkdir -p ~/.config
```

Create Node

```
v6 node new --user
```

Verify Configuration

```
sudo mkdir -p /etc/vantage6/node
sudo chmod 755 /etc/vantage6
sudo chmod 755 /etc/vantage6/node
v6 node list
```

YAML location:
```
/opt/v6-nodes/.config/vantage6/node/node_name.yaml
```
or 
```
/home/v6/.config/vantage6/node/node_name.yaml
```
Important: Database label must be identical across nodes.

# 6. SURF Research Cloud (SRC) — MDT Project

The MyDigiTwin infrastructure is hosted on SURF Research Cloud under:

https://sram.surf.nl/collaborations/6187/about

## Access Requirements

To obtain access to any workspace:

- Contact **Hyunho Mo** or **Djura Smits**
- Ensure authorization from the **project PIs**
- Especially required for **Rotterdam Study data**

---

## Key Workspaces

Below are the main workspaces used in the MDT project.

### Core Infrastructure

| Workspace | Purpose |
|-----------|---------|
| `v6-mdt-server` | Vantage6 server |
| `RS_FL_node_cpu` | Rotterdam Study node |
| `MDT_FL_AGGR` | Aggregation node |

These require pulling the Docker image.

---

### Practice / Mock Environment

| Workspace | Purpose |
|-----------|---------|
| `RS_MOCK_FIRST` | Mock RS node (split 1) |
| `RS_MOCK_SECOND` | Mock RS node (split 2) |
| `RS_MOCK_AGGR` | Mock aggregation node |

Used for training and testing the FL pipeline.

---

### Task Control Workspace

| Workspace | Purpose |
|-----------|---------|
| `FL_TASK_CONTROL` | Python environment to send tasks to Vantage6 server |

Contains:

- Federated learning control scripts
- Experiment configurations
- Evaluation scripts

---

## Important — Resource Usage

SURF projects operate with a **limited compute wallet**.

Please remember:

> Pause each workspace when not in use to avoid unnecessary resource consumption.

---

# 7. Submitting Tasks and Running Federated Learning

Once the infrastructure (nodes, Docker images, and SRC workspaces) is ready, you can submit and execute federated learning experiments using the Python scripts available in the **FL_TASK_CONTROL** workspace.

These scripts interact with the Vantage6 server and orchestrate the complete federated training pipeline.

---
## running node
Before starting the federated learning, for all the data and aggregation nodes, please check the status of node at each node workspace with
```
v6 node list
```
You can start the node with
```
v6 node start -n node_name --attach
```

---
## Available Control Scripts

| Script | Purpose |
|--------|---------|
| `discover_ids.py` | List collaboration, organization, and node IDs needed for later scripts |
| `run_central_ci_mock.py` | Submit the main central task (`central_ci`) to the aggregator node |
| `debug_task_runs.py` | Inspect metadata and logs for a task to diagnose issues |
| `list_recent_tasks.py` | Show the most recent task IDs in a collaboration |
| `get_central_results.py` | Retrieve results from a central task |
| `decode_central_outputs.py` | Decode encoded output files returned by the aggregator |
| `delete_tasks_bulk.py` | Delete one or multiple tasks for cleanup |
| `dump_task_logs_range.py` | Download logs for a range of task IDs |
| `extract_ci_from_logs.py` | Parse CI metrics from logs and export to JSON |
| `plot_ci_from_json.py` | Plot CI trajectories for a single fold |
| `ci_across_folds_with_ci.py` | Aggregate results across folds and compute confidence intervals |

---

## Execution Pipeline

```
discover_ids.py
        ↓
run_central_ci_mock.py
        ↓
   (if crash / error)
        ↓
debug_task_runs.py
read_run_log.py
        ↓
run_central_ci_mock.py
        ↓
list_recent_tasks.py
        ↓
get_central_results.py
        ↓
decode_central_outputs.py
```

---

## Discover Collaboration and Organization IDs

```bash
python discover_ids.py
```

Example output:

```
--- Collaborations ---
2 lifelines-poc-collaboration encrypted=True

Selected COLLAB_ID = 2

--- Organizations in collaboration ---
3 Lifelines-A
4 Lifelines B
10 rotterdam-study_mdt
15 mdt_fl_aggr

--- Nodes in collaboration ---
8 lifelines-poc-collaboration - Lifelines B status=online
9 lifelines-poc-collaboration - Lifelines-A status=online
47 lifelines-poc-collaboration - rotterdam-study_mdt status=online
54 lifelines-poc-collaboration - mdt_fl_aggr status=online
```

These IDs are required for submitting tasks.

---

## Federated Execution Logic (Internal)

```
FL_TASK_CONTROL
        ↓
Aggregator node runs central_ci
        ↓
Creates subtask (e.g. task 535)
        ↓
Data node 1 → partial_risk_prediction
Data node 2 → partial_risk_prediction
        ↓
Return local weights
        ↓
Aggregator performs FedAvg
        ↓
Return encoded output files
```

---

## Submit Federated Training Task

```bash
python run_central_ci_mock.py
```

This will:

1. Send a task to the aggregator node
2. Spawn subtasks at each data node
3. Perform federated averaging
4. Return encoded outputs

---

## Debugging Failed or Stuck Tasks

```bash
python debug_task_runs.py --task_id <TASK_ID>
```

You can inspect logs and metadata to determine the cause.

---

## Find Latest Tasks

```bash
python list_recent_tasks.py
```

Useful for identifying the most recent task IDs.

---

## Retrieve and Decode Results

Fetch central task results:

```bash
python get_central_results.py --task_id <TASK_ID>
```

Decode encoded outputs:

```bash
python decode_central_outputs.py --task_id <TASK_ID>
```

---

# 8. Manual Operations and Utilities

## Decode Central Outputs

```bash
python decode_central_outputs.py --task_id 569
```

## Delete Tasks in Bulk

```bash
python delete_tasks_bulk.py --start_id 1030 --end_id 1050
```

## Dump Logs for a Task Range

```bash
python dump_task_logs_range.py \
    --start_id 1042 \
    --end_id 1061 \
    --outdir logs_fedavg_20_100_16_fold_3
```

---

# 9. Extract Metrics and Plot Results

## Extract CI Metrics from Logs

```bash
python extract_ci_from_logs.py \
    --log_dir logs_fedavg_20_100_16_fold_3 \
    --start_id 1042 \
    --end_id 1061 \
    --output ci_results_20_100_16_fold_3.json
```

## Plot CI for a Single Fold

```bash
python plot_ci_from_json.py \
    --json ci_results_20_100_16_fold_3.json \
    --output figures/fedavg_ci_20_100_16_fold_3.png
```

## Aggregate Across Folds with Confidence Intervals

```bash
python ci_across_folds_with_ci.py \
    --glob "ci_results_*_fold_*.json" \
    --output "figures/ci_results_20_100_16.png" \
    --no-shade
```

You can stop the node with
```
v6 node stop -n node_name
```


---

# Contact

For questions related to federated learning execution:

- Hyunho Mo — h.mo@erasmusmc.nl  
- Djura Smits — d.smits@esciencecenter.nl  

---