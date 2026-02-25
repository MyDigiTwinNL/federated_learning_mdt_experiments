import os
import json
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO

from vantage6.client import Client

SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]
PRIVATE_KEY_PATH = "/home/hmo/fl_mdt_practice/privkey_mdt_fl_aggr.pem"

TASK_ID = 659   # <-- change if needed

# ---------- output file ----------
OUTPUT_DIR = os.path.join(os.getcwd(), "central_results_logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, f"central_task_{TASK_ID}_{TIMESTAMP}.log"
)
# ---------------------------------


def main():

    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(USERNAME, PASSWORD)
    client.setup_encryption(PRIVATE_KEY_PATH)

    buf = StringIO()

    with redirect_stdout(buf):

        print("========== CENTRAL RESULT FETCH ==========")
        print("TASK_ID:", TASK_ID)

        # -------------------------------------------------
        # OPTION A: wait_for_results (decrypted payload)
        # -------------------------------------------------
        try:
            print("\n=== wait_for_results ===")
            res_wait = client.wait_for_results(TASK_ID)
            print(json.dumps(res_wait, indent=2)[:20000])
        except Exception as e:
            print("wait_for_results failed:", e)

        # -------------------------------------------------
        # OPTION B: raw result endpoint
        # -------------------------------------------------
        print("\n=== GET /result?task_id ===")

        res = client.request(
            "result",
            method="GET",
            params={"task_id": TASK_ID}
        )

        print(json.dumps(res, indent=2)[:20000])

        # -------------------------------------------------
        # Short decoded summaries
        # -------------------------------------------------
        print("\n=== DECODED RESULT SUMMARY ===")

        for item in res.get("data", []):

            run_id = item.get("run", {}).get("id")
            print("\n--- run_id:", run_id, "---")

            result_payload = item.get("result")

            print("type:", type(result_payload))

            if isinstance(result_payload, dict):
                keys = list(result_payload.keys())
                print("keys:", keys)

            print("preview:", str(result_payload)[:500])

    # ---------- write file ----------
    text = buf.getvalue()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    # ---------- print console ----------
    print(text, end="")
    print(f"\n[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
