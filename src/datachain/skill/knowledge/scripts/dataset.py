"""Fetch schema, preview, query_script, changes, and deps for one dataset version."""

import argparse
import json
import sys

from changes import build_changes, dep_to_dict
from schema import extract_preview, extract_schema, get_catalog, parse_dataset_name
from utils import dc_import


def _warn(msg: str) -> None:
    print(f"[dc-knowledge warning] {msg}", file=sys.stderr)


def _try_read_chain(dc, name_version: str):
    try:
        return dc.read_dataset(name_version)
    except Exception as e:
        _warn(f"read_dataset({name_version}): {e}")
        return None


def _try_extract_schema(chain, name):
    if chain is None:
        return {}
    try:
        return extract_schema(chain)
    except Exception as e:
        _warn(f"extract_schema({name}): {e}")
        return {}


def _try_extract_preview(chain):
    if chain is None:
        return None
    return extract_preview(chain)


def _fetch_studio_meta(namespace, project, bare_name, version):
    try:
        from datachain.dataset import DatasetRecord
        from datachain.remote.studio import StudioClient

        client = StudioClient()
        response = client.dataset_info(namespace, project, bare_name)
        if response.ok and response.data:
            dataset = DatasetRecord.from_dict(response.data)
            attrs = list(dataset.attrs or [])
            description = dataset.description
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
            prev_version_info = None
            if idx is not None and idx > 0:
                p = sorted_vers[idx - 1]
                prev_version_info = (
                    p.version,
                    p.query_script or None,
                )
            return {
                "query_script": query_script,
                "uuid": uuid,
                "attrs": attrs,
                "description": description,
                "version": version,
                "prev_version_info": prev_version_info,
            }
    except Exception as e:
        _warn(f"Studio dataset_info({namespace}.{project}.{bare_name}): {e}")

    return {
        "query_script": None,
        "uuid": None,
        "attrs": [],
        "description": None,
        "version": version,
        "prev_version_info": None,
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


def _resolve_local_version(dc, catalog, bare_name, version):
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


def _fetch_local_dataset_info(catalog, bare_name, version):
    query_script = None
    uuid = None
    attrs: list[str] = []
    description: str | None = None
    prev_version_info = None
    try:
        if catalog is not None:
            dataset = catalog.get_dataset(bare_name, include_incomplete=False)
            attrs = list(dataset.attrs or [])
            description = dataset.description
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
                prev_version_info = (
                    p.version,
                    p.query_script or None,
                )
    except Exception as e:
        _warn(f"local catalog metadata for {bare_name}: {e}")

    return {
        "query_script": query_script,
        "uuid": uuid,
        "attrs": attrs,
        "description": description,
        "version": version,
        "prev_version_info": prev_version_info,
    }


def _fetch_local_dependencies(catalog, bare_name, version):
    dependencies: list = []
    try:
        if catalog is not None and version:
            deps = catalog.get_dataset_dependencies(
                name=bare_name, version=version, indirect=True
            )
            for dep in deps or []:
                if not dep:
                    continue
                dependencies.append(dep_to_dict(dep))
    except Exception as e:
        _warn(f"dependencies for {bare_name}@{version}: {e}")
    return dependencies


def _compute_version_changes(
    prev_version_info, query_script, dependencies,
    is_studio_dataset, catalog, bare_name,
):
    changes = None
    if prev_version_info is not None:
        prev_version_str, prev_script = prev_version_info
        prev_deps = []
        if not is_studio_dataset:
            try:
                if catalog is not None:
                    prev_deps_raw = (
                        catalog.get_dataset_dependencies(
                            name=bare_name,
                            version=prev_version_str,
                            indirect=True,
                        )
                        or []
                    )
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
        else:
            changes = build_changes(
                query_script,
                prev_version_str,
                prev_script,
                dependencies,
                [],
            )
    return changes


def fetch_version_data(name_version: str) -> dict:
    dc = dc_import()
    from datachain.dataset import parse_dataset_with_version

    name, version = parse_dataset_with_version(name_version)
    _namespace, _project, _bare_name = parse_dataset_name(name)
    is_studio_dataset = bool(_namespace and _project)

    chain = _try_read_chain(dc, name_version)
    schema = _try_extract_schema(chain, name)
    preview = _try_extract_preview(chain)

    query_script = None
    uuid = None
    attrs: list[str] = []
    description: str | None = None
    prev_version_info = None
    dependencies: list = []
    catalog = None

    if is_studio_dataset:
        meta = _fetch_studio_meta(_namespace, _project, _bare_name, version)
        query_script = meta["query_script"]
        uuid = meta["uuid"]
        attrs = meta["attrs"]
        description = meta["description"]
        version = meta["version"]
        prev_version_info = meta["prev_version_info"]
    else:
        catalog = _resolve_catalog(chain)
        version = _resolve_local_version(dc, catalog, _bare_name, version)
        meta = _fetch_local_dataset_info(catalog, _bare_name, version)
        query_script = meta["query_script"]
        uuid = meta["uuid"]
        attrs = meta["attrs"]
        description = meta["description"]
        version = meta["version"]
        prev_version_info = meta["prev_version_info"]
        dependencies = _fetch_local_dependencies(catalog, _bare_name, version)

    changes = _compute_version_changes(
        prev_version_info, query_script, dependencies,
        is_studio_dataset, catalog, _bare_name,
    )

    output_name = (
        f"{_namespace}.{_project}.{_bare_name}" if is_studio_dataset else _bare_name
    )

    return {
        "name": output_name,
        "uuid": uuid,
        "attrs": attrs,
        "description": description,
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
