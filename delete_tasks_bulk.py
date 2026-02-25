#!/usr/bin/env python3
import argparse
import os
from typing import List
from vantage6.client import Client


SERVER = os.environ.get(
    "V6_SERVER",
    "https://v6mdtserver.mydigitwin-umcu.src.surf-hosted.nl"
)
PORT = int(os.environ.get("V6_PORT", "443"))
API_PATH = os.environ.get("V6_API_PATH", "/api")


def make_client(log_level: str = "info") -> Client:
    username = os.environ["V6_USER"]
    password = os.environ["V6_PASS"]

    client = Client(SERVER, PORT, API_PATH, log_level=log_level)
    client.authenticate(username, password)

    privkey = os.environ.get("V6_PRIVATE_KEY")
    if privkey:
        client.setup_encryption(privkey)

    return client


def delete_tasks(client: Client, task_ids: List[int], dry_run: bool = False):
    for tid in task_ids:
        if dry_run:
            print(f"[DRY RUN] Would delete task_id={tid}")
            continue

        try:
            print(f"Deleting task_id={tid} ...")
            resp = client.task.delete(tid)
            print(f"  ✔ Deleted {tid}")
        except Exception as e:
            print(f"  ✖ Failed to delete {tid}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete Vantage6 task(s) by ID or ID range"
    )

    parser.add_argument("--task_id", type=int, help="Single task ID to delete")

    parser.add_argument(
        "--start_id",
        type=int,
        help="Start of task ID range (inclusive)"
    )

    parser.add_argument(
        "--end_id",
        type=int,
        help="End of task ID range (inclusive)"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be deleted"
    )

    parser.add_argument(
        "--log_level",
        default="info",
        help="Client log level (info/debug)"
    )

    args = parser.parse_args()

    if args.task_id is None and (args.start_id is None or args.end_id is None):
        parser.error(
            "Provide either --task_id OR both --start_id and --end_id"
        )

    client = make_client(log_level=args.log_level)

    # Build task list
    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        if args.start_id > args.end_id:
            parser.error("start_id must be <= end_id")

        task_ids = list(range(args.start_id, args.end_id + 1))

    print(f"\nTasks selected: {task_ids[0]} → {task_ids[-1]} "
          f"({len(task_ids)} tasks)")

    if not args.yes and not args.dry_run:
        confirm = input("Proceed with deletion? (y/N): ").lower().strip()
        if confirm != "y":
            print("Cancelled.")
            return

    delete_tasks(client, task_ids, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
