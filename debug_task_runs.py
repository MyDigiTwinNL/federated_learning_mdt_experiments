import os
import json
from vantage6.client import Client
import argparse

SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]
PRIVATE_KEY_PATH = "/home/hmo/fl_mdt_practice/privkey_mdt_fl_aggr.pem"

# TASK_ID = 560  # <-- subtask id from your run log

def main():
    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(USERNAME, PASSWORD)
    client.setup_encryption(PRIVATE_KEY_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help="fold index out of k", type=int, required = True)
    args = parser.parse_args()
    TASK_ID = args.task_id  # <-- change

    # List runs for this task
    runs_payload = client.request("run", method="GET", params={"task_id": TASK_ID})
    runs = runs_payload["data"] if isinstance(runs_payload, dict) and "data" in runs_payload else runs_payload

    print(f"\nFound {len(runs)} runs for task_id={TASK_ID}\n")
    for r in runs:
        run_id = r["id"]
        run = client.request(f"run/{run_id}", method="GET")
        if "data" in run and isinstance(run["data"], list) and run["data"]:
            run = run["data"][0]

        print("=" * 80)
        print("run_id:", run.get("id"))
        print("status:", run.get("status"))
        print("org_id:", run.get("organization", {}).get("id") if isinstance(run.get("organization"), dict) else run.get("organization_id"))
        print("node:", run.get("node", {}).get("name") if isinstance(run.get("node"), dict) else None)
        print("\n--- LOG ---\n")
        print(run.get("log") or run.get("logs") or "(no log field)")
        print()

if __name__ == "__main__":
    main()
