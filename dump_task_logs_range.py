import os
import argparse
from vantage6.client import Client

SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]
PRIVATE_KEY_PATH = "/home/hmo/fl_rs_mock_practice/privkey_rotterdam-study_mdt.pem"


def get_runs(client, task_id):
    """Fetch runs for a task id"""
    runs_payload = client.request("run", method="GET", params={"task_id": task_id})
    runs = (
        runs_payload["data"]
        if isinstance(runs_payload, dict) and "data" in runs_payload
        else runs_payload
    )
    return runs or []


def get_run_detail(client, run_id):
    """Fetch full run info including log"""
    run = client.request(f"run/{run_id}", method="GET")

    if "data" in run and isinstance(run["data"], list) and run["data"]:
        run = run["data"][0]

    return run


def save_log(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def main(start_id, end_id, outdir):

    os.makedirs(outdir, exist_ok=True)

    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(USERNAME, PASSWORD)
    client.setup_encryption(PRIVATE_KEY_PATH)

    print(f"Dumping logs from task {start_id} → {end_id}")
    print(f"Output folder: {outdir}")

    for task_id in range(start_id, end_id + 1):

        print(f"\nProcessing task_id={task_id}")

        try:
            runs = get_runs(client, task_id)

            if not runs:
                print("  No runs found")
                continue

            for r in runs:
                run_id = r["id"]

                run = get_run_detail(client, run_id)

                org_id = (
                    run.get("organization", {}).get("id")
                    if isinstance(run.get("organization"), dict)
                    else run.get("organization_id")
                )

                node_name = (
                    run.get("node", {}).get("name")
                    if isinstance(run.get("node"), dict)
                    else "unknown_node"
                )

                status = run.get("status", "unknown")

                log_text = run.get("log") or run.get("logs") or "(no log)"

                filename = (
                    f"task_{task_id}_run_{run_id}"
                    f"_org{org_id}_{status}.log"
                )

                filepath = os.path.join(outdir, filename)

                header = (
                    f"task_id: {task_id}\n"
                    f"run_id: {run_id}\n"
                    f"org_id: {org_id}\n"
                    f"node: {node_name}\n"
                    f"status: {status}\n"
                    + "=" * 80
                    + "\n\n"
                )

                save_log(filepath, header + log_text)

                print(f"  Saved {filename}")

        except Exception as e:
            print(f"  ERROR for task {task_id}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--outdir", type=str, default="task_logs")

    args = parser.parse_args()

    main(args.start_id, args.end_id, args.outdir)
