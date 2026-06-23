"""Fetch schema, preview, query_script, changes, and deps for one dataset version."""

import argparse
import json
import sys

from changes import build_changes, dep_to_dict
from schema import extract_preview, extract_schema, get_catalog, parse_dataset_name
from utils import dc_import


def _warn(msg: str) -> None:
    print(f"[dc-knowledge warning] {msg}", file=sys.stderr)


def _read_chain(dc, name_version: str):
    try:
        return dc.read_dataset(name_version)
    except Exception as e:
        _warn(f"read_dataset({name_version}): {e}")
        return None


def _extract_schema_preview(chain, dataset_name: str):
    schema = {}
    if chain is not None:
        try:
            schema = extract_schema(chain)
        except Exception as e:
            _warn(f"extract_schema({dataset_name}): {e}")
    preview = None
    if chain is not None:
        preview = extract_preview(chain)
    return schema, preview


def _fetch_studio_metadata(namespace, project, bare_name, version):
    """Fetch metadata from Studio API."""
    attrs = []
    description = None
    query_script = None
    uuid = None
    prev_version_info = None
    try:
        from datachain.dataset import DatasetRecord
        from datachain.remote.studio import StudioClient

        response = StudioClient().dataset_info(namespace, project, bare_name)
        if response.ok and response.data:
            dataset = DatasetRecord.from_dict(response.data)
            attrs = list(dataset.attrs or [])
            description = dataset.description
            if version is None:
                version = dataset.latest_version
            query_script = dataset.get_version(version).query_script or None
            uuid = getattr(dataset.get_version(version), "uuid", None)
            prev_version = max(
                (
                    v
                    for v in dataset.versions
                    if v.version_value < dataset.get_version(version).version_value
                ),
                key=lambda x: x.version_value,
                default=None,
            )
            if prev_version is not None:
                prev_version_info = (prev_version.version, prev_version.query_script or None)
    except Exception as e:
        _warn(f"Studio dataset_info({namespace}.{project}.{bare_name}): {e}")
    return {
        "attrs": attrs,
        "description": description,
        "version": version,
        "query_script": query_script,
        "uuid": uuid,
        "prev_version_info": prev_version_info,
    }


def _resolve_catalog(chain):
    catalog = None
    if chain is not None:
        catalog = chain.session.catalog
    if catalog is None:
        try:
            catalog = get_catalog()
        except Exception as e:
            _warn(f"get_catalog(): {e}")
    return catalog


def _resolve_version(dc, catalog, bare_name, version):
    if version is None and catalog is not None:
        try:
            ds = catalog.get_dataset(bare_name, include_incomplete=False)
            version = ds.latest_version
        except Exception as e:
            _warn(f"resolve version for {bare_name}: {e}")
    if version is None:
        for row in dc.datasets(column="dataset").to_iter():
            info = row[0]
            if info.name == bare_name:
                version = str(info.version)
                break
    return version


def _fetch_local_metadata(dc, chain, bare_name, version):
    """Fetch metadata from local catalog."""
    catalog = _resolve_catalog(chain)
    version = _resolve_version(dc, catalog, bare_name, version)

    attrs = []
    description = None
    query_script = None
    uuid = None
    prev_version_info = None
    try:
        if catalog is not None:
            dataset = catalog.get_dataset(bare_name, include_incomplete=False)
            attrs = list(dataset.attrs or [])
            description = dataset.description
            if version is None:
                version = dataset.latest_version
            query_script = dataset.get_version(version).query_script or None
            uuid = getattr(dataset.get_version(version), "uuid", None)
            prev_version = max(
                (
                    v
                    for v in dataset.versions
                    if v.version_value < dataset.get_version(version).version_value
                ),
                key=lambda x: x.version_value,
                default=None,
            )
            if prev_version is not None:
                prev_version_info = (prev_version.version, prev_version.query_script or None)
    except Exception as e:
        _warn(f"local catalog metadata for {bare_name}: {e}")

    dependencies = []
    try:
        if catalog is not None and version:
            dependencies = [
                dep_to_dict(d)
                for d in (catalog.get_dataset_dependencies(
                    name=bare_name, version=version, indirect=True
                ) or [])
                if d
            ]
    except Exception as e:
        _warn(f"dependencies for {bare_name}@{version}: {e}")

    return {
        "catalog": catalog,
        "attrs": attrs,
        "description": description,
        "version": version,
        "query_script": query_script,
        "uuid": uuid,
        "prev_version_info": prev_version_info,
        "dependencies": dependencies,
    }


def _compute_changes(query_script, prev_version_info, dependencies, catalog, bare_name):
    changes = None
    if prev_version_info is not None:
        prev_version_str, prev_script = prev_version_info
        prev_deps = []
        if catalog is not None:
            try:
                prev_deps_raw = catalog.get_dataset_dependencies(
                    name=bare_name,
                    version=prev_version_str,
                    indirect=True,
                ) or []
                prev_deps = [dep_to_dict(d) for d in prev_deps_raw if d]
            except Exception as e:
                _warn(f"prev dependencies for {bare_name}@{prev_version_str}: {e}")
        changes = build_changes(
            query_script,
            prev_version_str,
            prev_script,
            dependencies,
            prev_deps,
            catalog=catalog,
        )
    return changes


def fetch_version_data(name_version: str) -> dict:
    """Fetch schema, preview, query_script, changes, and deps.

    Returns a dict with keys:
    name, schema, preview, query_script, changes, dependencies.
    """
    dc = dc_import()
    from datachain.dataset import parse_dataset_with_version

    name, version = parse_dataset_with_version(name_version)
    _namespace, _project, _bare_name = parse_dataset_name(name)
    is_studio_dataset = bool(_namespace and _project)

    chain = _read_chain(dc, name_version)
    schema, preview = _extract_schema_preview(chain, name)

    meta = (
        _fetch_studio_metadata(_namespace, _project, _bare_name, version)
        if is_studio_dataset
        else _fetch_local_metadata(dc, chain, _bare_name, version)
    )

    changes = _compute_changes(
        meta.get("query_script"),
        meta.get("prev_version_info"),
        meta.get("dependencies", []),
        meta.get("catalog"),
        _bare_name,
    )

    output_name = (
        f"{_namespace}.{_project}.{_bare_name}" if is_studio_dataset else _bare_name
    )

    return {
        "name": output_name,
        "uuid": meta.get("uuid"),
        "attrs": meta.get("attrs", []),
        "description": meta.get("description"),
        "schema": schema,
        "preview": preview,
        "query_script": meta.get("query_script"),
        "changes": changes,
        "dependencies": meta.get("dependencies", []),
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
