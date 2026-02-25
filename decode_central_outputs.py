import os
import json
import base64
from vantage6.client import Client
import argparse


SERVER = "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
PORT = 443
API_PATH = "/api"

PRIVATE_KEY_PATH = "/home/hmo/fl_mdt_practice/privkey_mdt_fl_aggr.pem"


def extract_b64_or_bytes(item: dict):
    """
    Returns bytes for one encoded file item.
    Supports multiple shapes:
      - {"name":..., "data":"<b64>"}
      - {"name":..., "data": {"data":"<b64>", ...}}
      - {"name":..., "data": {"content":"<b64>", ...}}
      - {"name":..., "data": b"..."}  (already bytes)
    """
    data = item.get("data") or item.get("content") or item.get("blob")

    # If nested dict, peel common keys
    if isinstance(data, dict):
        # most common nested keys
        data = (
            data.get("data")
            or data.get("content")
            or data.get("blob")
            or data.get("base64")
        )

    if data is None:
        raise RuntimeError(f"Missing file payload in item. Keys={list(item.keys())}, item={item}")

    if isinstance(data, (bytes, bytearray)):
        return bytes(data)

    if isinstance(data, str):
        # base64 string
        return base64.b64decode(data)

    raise TypeError(f"Unexpected data type: {type(data)} in item={item}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help="fold index out of k", type=int, required = True)
    args = parser.parse_args()


    # CENTRAL_TASK_ID = 548  # <-- change
    CENTRAL_TASK_ID = args.task_id  # <-- change
    OUTDIR = "outputs_%s" %CENTRAL_TASK_ID  # <-- change if you want

    os.makedirs(OUTDIR, exist_ok=True)

    client = Client(SERVER, PORT, API_PATH, log_level="info")
    client.authenticate(os.environ["V6_USER"], os.environ["V6_PASS"])
    client.setup_encryption(PRIVATE_KEY_PATH)
    print("client.result methods:", [m for m in dir(client.result) if not m.startswith("_")])


    # --- Get results for this task ---
    # Different v6-client versions expose different methods. Common one is from_task().
    results = client.result.from_task(CENTRAL_TASK_ID)

    # Normalize shape
    if isinstance(results, dict) and "data" in results:
        results = results["data"]

    if not results:
        raise RuntimeError(f"No results found for task {CENTRAL_TASK_ID}")

    result_obj = results[0].get("result")

    # If still bytes, decode
    if isinstance(result_obj, (bytes, bytearray)):
        result_obj = result_obj.decode("utf-8", errors="strict")


    payload = json.loads(result_obj)

    # If payload is still a JSON string, decode again (double-encoded JSON)
    if isinstance(payload, str):
        payload = json.loads(payload)

    encoded = payload.get("encoded_output_files", None)
    if encoded is None:
        raise RuntimeError("No encoded_output_files found in payload.")

    # --- Normalize encoded_output_files into a list of {"name":..., "data":...} dicts ---
    if isinstance(encoded, str):
        # could be JSON string of list/dict
        try:
            encoded = json.loads(encoded)
        except Exception:
            raise RuntimeError(
                f"encoded_output_files is a string (len={len(encoded)}), but not JSON. "
                "This likely means encode_files returned a raw base64 blob. "
                "Print the first 200 chars to inspect."
            )

    if isinstance(encoded, dict):
        # common format: { "file1": "base64...", "file2": "base64..." }
        encoded = [{"name": k, "data": v} for k, v in encoded.items()]

    if not isinstance(encoded, list):
        raise RuntimeError(f"encoded_output_files has unexpected type: {type(encoded)}")

    print("Number of files:", len(encoded))
    print("First item type:", type(encoded[0]))
    print("First item keys (if dict):", list(encoded[0].keys()) if isinstance(encoded[0], dict) else None)



    if not encoded:
        raise RuntimeError("No encoded_output_files found in payload.")

    print("Keys in payload:", list(payload.keys()))
    print("Number of encoded files:", len(encoded))

    # for item in encoded:
    #     # encode_files typically returns {"name": "...", "data": "..."}
    #     name = (item.get("name") or item.get("filename") or item.get("path") or "unknown").lstrip("/")
    #     data = item.get("data") or item.get("content") or item.get("blob")

    #     if data is None:
    #         raise RuntimeError(f"Missing data field in item: keys={list(item.keys())}")

    #     # some encoders return bytes already
    #     if isinstance(data, str):
    #         raw = base64.b64decode(data)
    #     else:
    #         raw = data

    #     # IMPORTANT: make paths safe (avoid writing outside OUTDIR)
    #     safe_name = name.replace("..", "__").lstrip("/")

    #     outpath = os.path.join(OUTDIR, safe_name)
    #     os.makedirs(os.path.dirname(outpath), exist_ok=True)

    #     # with open(outpath, "wb") as f:
    #     #     f.write(base64.b64decode(item["data"]))

    #     with open(outpath, "wb") as f:
    #         f.write(raw)

    #     print("Saved:", outpath)

    for item in encoded:
        name = (item.get("name") or item.get("filename") or item.get("path") or "unknown").lstrip("/")
        raw = extract_b64_or_bytes(item)

        safe_name = name.replace("..", "__").lstrip("/")
        outpath = os.path.join(OUTDIR, safe_name)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        print("DEBUG item['name']:", item.get("name"))
        print("DEBUG type(item['data']):", type(item.get("data")))
        if isinstance(item.get("data"), dict):
            print("DEBUG nested keys:", list(item["data"].keys()))


        with open(outpath, "wb") as f:
            f.write(raw)

        print("Saved:", outpath)


    print("\nDone. Files written under:", OUTDIR)



if __name__ == "__main__":
    import os
    main()
