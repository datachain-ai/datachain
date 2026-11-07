import argparse
import json
from typing import TYPE_CHECKING

from schema import extract_preview, extract_schema, get_catalog, parse_dataset_name
from utils import dc_import

from datachain.dataset import parse_dataset_with_version
from datachain.error import DataChainError, NotFoundError
from datachain.lib.dataset_metadata import read_dataset_metadata
from datachain.skill.knowledge.collect import snapshot_from_metadata

if TYPE_CHECKING:
    from datachain.skill.knowledge.types import DatasetSnapshot


def fetch_dataset_snapshot(
    name_version: str, *, summarize: bool = False
) -> "DatasetSnapshot":
    name, version = parse_dataset_with_version(name_version)
    # Skills also accept the legacy namespace/project/name notation.
    namespace, project, bare_name = parse_dataset_name(name)
    if namespace and project and "://" not in name:
        name = f"{namespace}.{project}.{bare_name}"
    metadata = read_dataset_metadata(
        get_catalog(), name, version=version, include_preview=True
    )
    snapshot = snapshot_from_metadata(metadata)
    selected = snapshot["versions"][-1]
    warnings = snapshot.setdefault("warnings", [])
    try:
        chain = dc_import().read_dataset(
            f"{metadata.name}@{metadata.selected_version.version}"
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"read_dataset: {exc}")
        return snapshot

    try:
        selected["schema"] = extract_schema(chain)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"schema: {exc}")
    preview = extract_preview(chain)
    if preview is None:
        warnings.append("preview: extraction failed")
    else:
        selected["preview"] = preview
    if summarize:
        try:
            from summary import dataset_summary_from_chain

            selected["summary"] = dataset_summary_from_chain(chain)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"summary: {exc}")
    return snapshot


def fetch_version_data(name_version: str) -> dict:
    """Fetch one version's schema, preview, query script, changes, and deps."""
    snapshot = fetch_dataset_snapshot(name_version)
    versions = snapshot["versions"]
    selected = versions[-1]
    changes = selected["changes"]
    if (
        changes
        and changes["script_changed"]
        and changes["previous_script"] is None
        and len(versions) > 1
    ):
        changes["previous_script"] = versions[-2]["query_script"]
    result = {
        "name": snapshot["name"],
        "attrs": snapshot["attrs"],
        "description": snapshot["description"],
        **{
            key: selected[key]
            for key in (
                "uuid",
                "schema",
                "preview",
                "query_script",
                "changes",
                "dependencies",
            )
        },
    }
    if snapshot.get("warnings"):
        result["warnings"] = snapshot["warnings"]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch schema, preview, and deps for a dataset version."
    )
    parser.add_argument(
        "name",
        help="Dataset name, optionally with @version (e.g. my_dataset@1.0.0)",
    )
    args = parser.parse_args()
    try:
        result = fetch_version_data(args.name)
    except (DataChainError, NotFoundError) as exc:
        parser.exit(1, json.dumps({"error": str(exc)}) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
