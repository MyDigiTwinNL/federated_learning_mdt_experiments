import os
import re
import json
import argparse


ORG_MAP = {
    3: "Lifelines",
    10: "Rotterdam Study",
}


CI_PATTERN = re.compile(r"test ci\s+([0-9\.]+)")


def extract_ci_from_file(filepath):
    """Extract CI value from a log file"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    match = CI_PATTERN.search(text)
    if match:
        return float(match.group(1))

    return None


def main(log_dir, start_id, end_id, output_file):

    results = {
        "Lifelines": [],
        "Rotterdam Study": []
    }

    for task_id in range(start_id, end_id + 1):

        print(f"Processing task {task_id}")

        # find files belonging to this task
        for fname in os.listdir(log_dir):

            if not fname.startswith(f"task_{task_id}_"):
                continue

            filepath = os.path.join(log_dir, fname)

            # detect org id from filename
            org_match = re.search(r"_org(\d+)_", fname)
            if not org_match:
                continue

            org_id = int(org_match.group(1))

            if org_id not in ORG_MAP:
                continue

            dataset_name = ORG_MAP[org_id]

            ci_value = extract_ci_from_file(filepath)

            if ci_value is not None:
                results[dataset_name].append(ci_value)
                print(f"  {dataset_name}: {ci_value}")
            else:
                print(f"  WARNING: no CI found in {fname}")

    # save dictionary
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to:", output_file)
    print("\nFinal dictionary:")
    print(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--output", default="ci_results.json")

    args = parser.parse_args()

    main(
        log_dir=args.log_dir,
        start_id=args.start_id,
        end_id=args.end_id,
        output_file=args.output
    )
