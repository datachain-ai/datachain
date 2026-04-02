"""Delete intermediate .json files after enrichment.

Reads the plan JSON to find which datasets and buckets were processed,
then removes their .json data files. Keeps .plan.json intact.

Usage:
    python3 cleanup_json.py --plan datachain/graph/.plan.json
"""

import argparse
import json
import os
import sys


def cleanup(plan_path: str, dry_run: bool = False) -> int:
    """Delete .json files for processed datasets and buckets. Returns count deleted."""
    with open(plan_path) as f:
        plan = json.load(f)

    graph_dir = os.path.dirname(plan_path) or "datachain/graph"
    deleted = 0

    for section in ("datasets", "buckets"):
        for entry in plan.get(section, []):
            if entry.get("status") == "ok":
                continue
            file_path = entry.get("file_path")
            if not file_path:
                continue
            json_path = os.path.join(graph_dir, file_path + ".json")
            if os.path.exists(json_path):
                if dry_run:
                    print(f"Would delete: {json_path}")
                else:
                    os.remove(json_path)
                    print(f"Deleted: {json_path}")
                deleted += 1

    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Delete intermediate .json files after enrichment."
    )
    parser.add_argument("--plan", required=True, help="Path to .plan.json file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print files without deleting"
    )
    args = parser.parse_args()

    if not os.path.exists(args.plan):
        print(f"Plan file not found: {args.plan}", file=sys.stderr)
        sys.exit(1)

    count = cleanup(args.plan, dry_run=args.dry_run)
    if not count:
        print("No .json files to clean up.")


if __name__ == "__main__":
    main()
