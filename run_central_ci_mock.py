import os
import json
from vantage6.client import Client


# --- Collaborations ---
# 2 lifelines-poc-collaboration encrypted=True

# Selected COLLAB_ID = 2

# --- Organizations in collaboration ---
# 3 Lifelines-A
# 4 Lifelines B
# 10 rotterdam-study_mdt
# 15 mdt_fl_aggr

# --- Nodes in collaboration (optional but useful) ---
# 8 lifelines-poc-collaboration - Lifelines B status=online
# 9 lifelines-poc-collaboration - Lifelines-A status=online
# 47 lifelines-poc-collaboration - rotterdam-study_mdt status=online
# 54 lifelines-poc-collaboration - mdt_fl_aggr status=online


SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

COLLAB_ID = 2
ORG_AGG = 15

DB_LABEL = "mdt_full"
# IMAGE = "ghcr.io/mydigitwinnl/federated_cvdm_training_poc:v6ver481v3_oomhandlerv4_log"
IMAGE = "ghcr.io/mydigitwinnl/federated_cvdm_training_poc:v6ver481v3_oomhandlerv4_lr"


USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]

PRIVATE_KEY_PATH = "/home/hmo/fl_mdt_practice/privkey_mdt_fl_aggr.pem"

predictor_cols = [
    "GENDER",
    "T2D_STATUS",
    "HYPERTENSION_STATUS",
    "SMOKING_STATUS",
    "TOTAL_CHOLESTEROL_VALUE",
    "HDL_CHOLESTEROL_VALUE",
    "LDL_CHOLESTEROL_VALUE",
    "SYSTOLIC_VALUE",
    "DIASTOLIC_VALUE",
    "EGFR_VALUE",
    "CREATININE_VALUE",
    "AGE",
]
outcome_cols = ["LENFOL", "FSTAT"]

num_update_iter = 20
n_fold = 10
fold_index = 0
agg_weight_filename = "/tmp/agg_weights.pth"

# dl_config = {
#     "network": {"drop": 0.1, "dims": [len(predictor_cols), 32, 16, 1]},
#     "train": {"epochs": 1, "learning_rate": 1e-3, "optimizer": "Adam"},
# }

dl_config = {
    "train": {
        "epochs": 100,                 # for smoke test set to 1
        "learning_rate": 2e-4,
        "lr_decay_rate": 1e-5,
        "optimizer": "Adam",
    },
    "network": {
        "drop": 0.20,
        "norm": True,
        "dims": [len(predictor_cols), 16, 16, 1],  # first dim must match your predictors
        "activation": "ReLU",
        "l2_reg": 0.0,
    },
}

def main():
    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(USERNAME, PASSWORD)
    client.setup_encryption(PRIVATE_KEY_PATH)

    input_ = {
        "method": "central_ci",
        "kwargs": dict(
            predictor_cols=predictor_cols,
            outcome_cols=outcome_cols,
            dl_config=dl_config,
            num_update_iter=num_update_iter,
            n_fold=n_fold,
            fold_index=fold_index,
            agg_weight_filename=agg_weight_filename,
        ),
    }

    task = client.task.create(
        collaboration=COLLAB_ID,
        organizations=[ORG_AGG],
        name="MDT_FL - central_ci (smoke)",
        description= "mdt v6 fl rs lifelines",
        image=IMAGE,
        input_=input_,
        databases=[{"label": "mdt_full"}],
    )
    task_id = task["id"]
    print("Submitted task_id:", task_id)

    results = client.wait_for_results(task_id)
    print("\n=== wait_for_results raw ===")
    print(json.dumps(results, indent=2))

    # Convenience: show run id(s) and the decrypted "result" payload(s)
    print("\n=== extracted results ===")
    for item in results.get("data", []):
        run_id = item.get("run", {}).get("id")
        res = item.get("result")
        print(f"\n--- run_id={run_id} ---")
        print("type(result) =", type(res))
        print("result =", res)

if __name__ == "__main__":
    main()
