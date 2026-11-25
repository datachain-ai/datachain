import inspect
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from datachain import json
from datachain.lib.data_model import DataType
from datachain.lib.dc.storage_pattern import storage_sources
from datachain.lib.dc.utils import DatasetPrepareError
from datachain.lib.file import File
from datachain.lib.storage_config import client_configs, session_for
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain

_TABLE_KWARGS = frozenset(("in_memory", "anon", "versions", "client_config"))


def _open_lance_table(
    uri: str,
    target: File,
    version: int | str | None,
    session: Session,
    opts: dict[str, Any] | None,
) -> tuple[File, Any, Any] | None:
    from datachain.lib.arrow import (
        _decode_lance_schema,
        _has_lance_versions,
        _open_lance_dataset,
    )

    scheme = urlsplit(target.source).scheme.lower()
    has_ver = False
    try:
        if scheme in ("http", "https", "hf"):
            if version is None:
                return None
            raise ValueError(f"Lance tables are not supported for '{scheme}://' URIs")
        client = session.catalog.get_client(target.source)
        has_ver = bool(target.path and _has_lance_versions(target, client))
        if not has_ver:
            if version is None:
                return None
            raise ValueError(f"'{uri}' is not a Lance table")
        tbl: dict[str, Any] = {
            "source": target.source,
            "path": target.path,
            "version": version,
        }
        if opts:
            tbl["storage_options"] = opts
        ds = _open_lance_dataset(target, tbl)
        return target, ds, _decode_lance_schema(ds.schema)
    except (OSError, ValueError) as exc:
        if version is None and not has_ver:
            return None
        raise DatasetPrepareError(uri, exc) from exc


def read_lance(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    output: dict[str, DataType] | None = None,
    column: str = "",
    model_name: str = "",
    source: bool = True,
    session: Session | None = None,
    settings: dict | None = None,
    columns: Sequence[str] | None = None,
    version: int | str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "DataChain":
    """Generate chain from Lance tables or standalone Lance v2 data files.

    Supports column projection and filter pushdown on external tables and
    standalone files. Lance tables are always read via range reads and are never
    cached; remote standalone `.lance` files are downloaded in full, or cached
    when `settings(cache=True)`.

    Parameters:
        path: Local path or storage URI (`s3://`, `gs://`, `az://`, or
            `http(s)://`/`hf://` for standalone files), or list of paths. Glob
            patterns match standalone `.lance` files only; pass one path per
            Lance table.
        output: Dictionary mapping result column names to Python types.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source row (`ArrowRow`). For
            Lance tables, `source.index` is Lance's row id and `source.file` is
            the fragment data file under `data/`.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns (or DataChain signal paths) to read, in order.
        version: Optional Lance table version number (`int`) or tag name (`str`).
        storage_options: Optional Lance storage options (Lance key names),
            merged over translated `client_config`.
        kwargs: Parameters passed to `read_storage` (`client_config` uses fsspec
            key names).
    """
    from datachain.lib.arrow import _lance_commit_time, _tabular_generator

    from .records import read_records
    from .storage import read_storage

    uris = [os.fspath(u) for u in (path if isinstance(path, (list, tuple)) else [path])]
    if not uris:
        raise ValueError("No URIs provided")
    inspect.signature(read_storage).bind(uris, **kwargs)
    if isinstance(version, bool):
        raise TypeError("version must be an int or tag name, not bool")
    if version is not None and len(uris) > 1:
        raise DatasetPrepareError(uris[0], "Cannot specify version for multiple paths")

    client_config, anon = kwargs.get("client_config"), kwargs.get("anon")
    versions, in_memory = kwargs.get("versions", True), kwargs.get("in_memory", False)
    base_session = Session.get(session, in_memory=in_memory)
    base_cfg = (
        base_session.catalog.client_config if client_config is None else client_config
    )
    tbl_session = session_for(base_session, base_cfg, in_memory)
    tbl_session.catalog.source_client_configs = dict(
        base_session.catalog.source_client_configs
    )
    tables: list[tuple[File, Any, Any]] = []
    for u in uris:
        routes = storage_sources([u])
        is_glob = len(routes) > 1 or routes[0][1] is not None
        if is_glob and version is not None:
            raise DatasetPrepareError(u, f"Cannot specify version for glob '{u}'")
        if is_glob:
            continue
        route = routes[0][0].rstrip("/\\")
        target = File.at(route, session=tbl_session)
        src = target.source
        base = (
            base_session.catalog.client_config_for(src)
            if client_config is None
            else client_config
        )
        _, cfg = client_configs([route], base_session, anon, base, versions)
        tbl_session.catalog.source_client_configs[src] = cfg
        if opened := _open_lance_table(
            u, target, version, tbl_session, storage_options
        ):
            tables.append(opened)

    if not tables:
        return read_storage(
            path, session=session, settings=settings, **kwargs
        ).parse_tabular(
            output, column, model_name, source, columns=columns, format="lance"
        )
    if len(tables) != len(uris):
        raise DatasetPrepareError(
            uris[0],
            "Cannot mix Lance tables and file paths in read_lance; "
            "read each and union() them",
        )
    if bad := set(kwargs) - _TABLE_KWARGS:
        raise TypeError(f"Argument {min(bad)!r} is not supported for Lance tables")

    table_records: list[dict[str, Any]] = []
    table_schemas: list[tuple[str, Any]] = []
    for t, ds, schema in tables:
        ts = _lance_commit_time(ds)
        frags = [(f.fragment_id, f.data_files()[0]) for f in ds.get_fragments()] or [
            (None, None)
        ]
        for fid, df in frags:
            f_path = f"{t.path}/data/{df.path}".lstrip("/") if df else t.path
            sz = (df.file_size_bytes or 0) if df else 0
            row_file = File(source=t.source, path=f_path, size=sz)
            tbl = {
                "source": t.source,
                "path": t.path,
                "version": ds.version,
                "timestamp": ts,
                "fragment": fid,
            }
            table_records.append({"file": row_file, "lance_fragment": json.dumps(tbl)})
        table_schemas.append((table_records[-len(frags)]["file"].path, schema))
    ordered = [s for _, s in sorted(table_schemas, key=lambda x: x[0])]
    # Kept on the generator, never in rows: options can hold secrets.
    gen_opts: dict[str, Any] = (
        {"storage_options": storage_options} if storage_options else {}
    )
    gen, out = _tabular_generator(
        None,
        output,
        column,
        model_name,
        source,
        columns,
        schemas=ordered,
        name=uris[0],
        **gen_opts,
    )
    tbl_chain = read_records(
        table_records,
        schema={"file": File, "lance_fragment": str},
        session=tbl_session,
        settings=settings,
        in_memory=in_memory,
    )
    return tbl_chain.gen(gen, params=["file", "lance_fragment"], output=out)  # type: ignore[arg-type]
