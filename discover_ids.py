import os
from vantage6.client import Client

SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

COLLAB_NAME = "lifelines-poc-collaboration"

def unwrap(resp):
    """
    Vantage6 client sometimes returns:
      - list[dict]
      - {"data": list[dict], ...}
      - {"data": {"data": list[dict], ...}, ...}  (rare)
    This makes it consistent.
    """
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        data = resp.get("data", resp)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
    raise TypeError(f"Unexpected response type/shape: {type(resp)} -> {resp}")

USERNAME = os.environ["V6_USER"]
PASSWORD = os.environ["V6_PASS"]

client = Client(SERVER, PORT, API_PATH, log_level="info")
client.authenticate(USERNAME, PASSWORD)

print("\n--- Collaborations ---")
collabs = unwrap(client.collaboration.list())
for c in collabs:
    print(c["id"], c["name"], "encrypted=" + str(c.get("encrypted", None)))

collab = next(c for c in collabs if c["name"] == COLLAB_NAME)
COLLAB_ID = collab["id"]
print("\nSelected COLLAB_ID =", COLLAB_ID)

print("\n--- Organizations in collaboration ---")
orgs = unwrap(client.organization.list(collaboration=COLLAB_ID))
for o in orgs:
    print(o["id"], o["name"])

print("\n--- Nodes in collaboration (optional but useful) ---")
nodes = unwrap(client.node.list(collaboration=COLLAB_ID))
for n in nodes:
    # not all fields exist in all versions, so use .get
    print(n.get("id"), n.get("name"), "status=" + str(n.get("status")))
