"""Fetch schema, preview, query_script, changes, and deps for one dataset version."""

import argparse
import json
import sys

from changes import build_changes, dep_to_dict
from schema import extract_preview, extract_schema, get_catalog, parse_dataset_name
from utils import dc_import


def _warn(msg: str) -> None:
    print(f"[dc-graph warning] {msg}", file=sys.stderr)


def fetch_version_data(name_version: str) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Fetch schema, preview, query_script, changes, and deps.

    Returns a dict with keys:
    name, schema, preview, query_script, changes, dependencies.
    """
    dc = dc_import()
    from datachain.dataset import parse_dataset_with_version

    name, version = parse_dataset_with_version(name_version)
    _namespace, _project, _bare_name = parse_dataset_name(name)
    is_studio_dataset = bool(_namespace and _project)

    # --- Schema and Preview: requires live data access ---
    chain = None
    try:
        chain = dc.read_dataset(name_version)
    except Exception as e:  # noqa: BLE001
        _warn(f"read_dataset({name_version}): {e}")

    schema = {}
    if chain is not None:
        try:
            schema = extract_schema(chain)
        except Exception as e:  # noqa: BLE001
            _warn(f"extract_schema({name}): {e}")

    preview = None
    if chain is not None:
        preview = extract_preview(chain)

    # --- Metadata: query_script, uuid, version history ---
    query_script = None
    uuid = None
    _prev_version_info = None
    dependencies = []

    if is_studio_dataset:
        # Fetch full dataset record directly from Studio API.
        try:
            from datachain.dataset import DatasetRecord
            from datachain.remote.studio import StudioClient

            client = StudioClient()
            response = client.dataset_info(_namespace, _project, _bare_name)
            if response.ok and response.data:
                dataset = DatasetRecord.from_dict(response.data)
                resolved_ver = version or dataset.latest_version
                if version is None:
                    version = resolved_ver
                dv = dataset.get_version(resolved_ver)
                query_script = dv.query_script or None
                uuid = getattr(dv, "uuid", None)
                sorted_vers = sorted(dataset.versions, key=lambda v: v.version_value)
                idx = next(
                    (i for i, v in enumerate(sorted_vers) if v.version == resolved_ver),
                    None,
                )
                if idx is not None and idx > 0:
                    p = sorted_vers[idx - 1]
                    _prev_version_info = (
                        p.version,
                        p.query_script or None,
                    )
        except Exception as e:  # noqa: BLE001
            _warn(f"Studio dataset_info({name}): {e}")
    else:
        # Get catalog independently.
        catalog = None
        if chain is not None:
            catalog = chain.session.catalog
        if catalog is None:
            try:
                catalog = get_catalog()
            except Exception as e:  # noqa: BLE001
                _warn(f"get_catalog(): {e}")

        # Resolve version for local datasets
        if version is None and catalog is not None:
            try:
                ds = catalog.get_dataset(_bare_name, include_incomplete=False)
                version = ds.latest_version
            except Exception as e:  # noqa: BLE001
                _warn(f"resolve version for {_bare_name}: {e}")
        if version is None:
            for row in dc.datasets(column="dataset").to_iter():
                info = row[0]
                if info.name == _bare_name:
                    version = str(info.version)
                    break

        # Fetch from local catalog
        try:
            if catalog is not None:
                dataset = catalog.get_dataset(_bare_name, include_incomplete=False)
                resolved_ver = version or dataset.latest_version
                dv = dataset.get_version(resolved_ver)
                query_script = dv.query_script or None
                uuid = getattr(dv, "uuid", None)
                sorted_vers = sorted(dataset.versions, key=lambda v: v.version_value)
                idx = next(
                    (i for i, v in enumerate(sorted_vers) if v.version == resolved_ver),
                    None,
                )
                if idx is not None and idx > 0:
                    p = sorted_vers[idx - 1]
                    _prev_version_info = (
                        p.version,
                        p.query_script or None,
                    )
        except Exception as e:  # noqa: BLE001
            _warn(f"local catalog metadata for {_bare_name}: {e}")

        # Dependencies (local metastore only)
        try:
            if catalog is not None and version:
                deps = catalog.get_dataset_dependencies(
                    name=_bare_name, version=version, indirect=True
                )
                for dep in deps or []:
                    if not dep:
                        continue
                    dependencies.append(dep_to_dict(dep))
        except Exception as e:  # noqa: BLE001
            _warn(f"dependencies for {_bare_name}@{version}: {e}")

    # --- Changes section ---
    changes = None
    if _prev_version_info is not None:
        prev_version_str, prev_script = _prev_version_info
        # For local datasets, fetch previous deps for diffing
        prev_deps = []
        if not is_studio_dataset:
            try:
                if catalog is not None:
                    prev_deps_raw = (
                        catalog.get_dataset_dependencies(
                            name=_bare_name,
                            version=prev_version_str,
                            indirect=True,
                        )
                        or []
                    )
                    prev_deps = [dep_to_dict(d) for d in prev_deps_raw if d]
            except Exception as e:  # noqa: BLE001
                _warn(f"prev dependencies for {_bare_name}@{prev_version_str}: {e}")
            changes = build_changes(
                query_script,
                prev_version_str,
                prev_script,
                dependencies,
                prev_deps,
                catalog=catalog,
            )
        else:
            changes = build_changes(
                query_script,
                prev_version_str,
                prev_script,
                dependencies,
                [],
            )

    # Return fully-qualified dot-separated name for Studio datasets;
    # bare name for local datasets.
    output_name = (
        f"{_namespace}.{_project}.{_bare_name}" if is_studio_dataset else _bare_name
    )

    return {
        "name": output_name,
        "uuid": uuid,
        "schema": schema,
        "preview": preview,
        "query_script": query_script,
        "changes": changes,
        "dependencies": dependencies,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch schema, preview, and deps for a dataset version."
    )
    parser.add_argument(
        "name",
        help="Dataset name, optionally with @version (e.g. my_dataset@1.0.0)",
    )
    args = parser.parse_args()
    print(json.dumps(fetch_version_data(args.name), indent=2))


if __name__ == "__main__":
    main()
