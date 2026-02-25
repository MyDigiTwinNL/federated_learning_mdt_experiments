import os
from vantage6.client import Client

SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]
PRIVATE_KEY_PATH = "/home/hmo/fl_mdt_practice/privkey_mdt_fl_aggr.pem"

COLLAB_ID = 2
N = 50          # how many tasks you want to print
PER_PAGE = 10    # server page size (safe value)

def fetch_tasks_paged(client, collaboration_id, n, per_page=50):
    """
    Fetch tasks via raw API pagination. Works even if client.task.list() can't set page_size.
    """
    out = []
    page = 1
    while len(out) < n:
        # Vantage6 typically supports: page, per_page on list endpoints
        resp = client.request(
            f"task?collaboration={collaboration_id}&page={page}&per_page={per_page}",
            method="GET"
        )

        data = resp.get("data", resp)
        if not data:
            break

        out.extend(data)

        # stop if last page
        if len(data) < per_page:
            break

        page += 1

    return out[:n]

def main():
    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(USERNAME, PASSWORD)
    client.setup_encryption(PRIVATE_KEY_PATH)

    tasks = fetch_tasks_paged(client, COLLAB_ID, N, PER_PAGE)

    # sort newest first (IDs are monotonic)
    tasks = sorted(tasks, key=lambda t: t["id"], reverse=True)

    print(f"COLLAB_ID {COLLAB_ID} | showing {len(tasks)} most recent tasks")
    print("-" * 120)
    for t in tasks:
        print(f"{t['id']:5d}  {t.get('name','')}")
    print("-" * 120)

if __name__ == "__main__":
    main()
