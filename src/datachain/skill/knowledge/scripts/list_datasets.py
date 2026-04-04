"""List all user datasets as JSON."""

import argparse
import json

from utils import collect_datasets, dc_import


def cmd_list(studio: bool = False):
    dc = dc_import()

    datasets = []
    seen = set()  # (name, version) dedup across sources

    if studio:
        for entry in collect_datasets(dc, studio=True):
            key = (entry["name"], entry["version"])
            if key not in seen:
                seen.add(key)
                datasets.append(entry)
    else:
        for entry in collect_datasets(dc, studio=False):
            key = (entry["name"], entry["version"])
            if key not in seen:
                seen.add(key)
                datasets.append(entry)

    print(json.dumps({"datasets": datasets}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="List DataChain datasets.")
    parser.add_argument(
        "--studio",
        action="store_true",
        help="List Studio datasets instead of local",
    )
    args = parser.parse_args()
    cmd_list(studio=args.studio)


if __name__ == "__main__":
    main()
