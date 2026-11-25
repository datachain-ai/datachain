import os
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from datachain import json
from datachain.lib.arrow import (
    _decode_lance_schema,
    _lance_manifest,
    _open_lance_dataset,
    _tabular_generator,
)
from datachain.lib.data_model import DataType
from datachain.lib.dc.storage_pattern import storage_sources
from datachain.lib.dc.utils import DatasetPrepareError
from datachain.lib.file import File
from datachain.lib.storage_config import client_configs, session_for
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def _open_lance_table(
    target: File, version: int | str | None, session: Session
) -> tuple[File, Any, Any, File] | None:
    scheme = urlsplit(target.source).scheme.lower()
    res = None
    try:
        if scheme in ("http", "https", "hf"):
            if version is None:
                return None
            raise ValueError(f"Lance tables are not supported for '{scheme}://' URIs")
        if target.path:
            res = _lance_manifest(
                target, version, session.catalog.get_client(target.source)
            )
        if not res:
            if version is None:
                return None
            raise ValueError(f"'{target.source}/{target.path}' is not a Lance table")
        ver, m = res
        tbl = {
            "source": target.source,
            "path": target.path,
            "version": ver,
            "manifest": [m.path, m.etag],
        }
        ds = _open_lance_dataset(target, tbl)
        return target, ds, _decode_lance_schema(ds.schema), m
    except OSError as exc:
        if version is None and not res:
            return None
        raise DatasetPrepareError("lance", exc) from exc
    except ValueError as exc:
        raise DatasetPrepareError("lance", exc) from exc


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
    **kwargs: Any,
) -> "DataChain":
    """Generate chain from a Lance table or standalone Lance v2 data files.

    Supports column projection and filter pushdown like `read_parquet`.
    Lance tables are always read natively via range reads, regardless of
    `settings(cache=...)`.

    Parameters:
        path: Local path or storage URI (`s3://`, `gs://`, `az://`), or list of paths.
        output: Dictionary mapping result column names to Python types.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source row (`ArrowRow`).
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns (or DataChain signal paths) to read, in order.
        version: Optional Lance table version number or tag name.
        kwargs: Parameters passed to `read_storage` (e.g. `client_config`, `anon`).
    """
    from .records import read_records
    from .storage import read_storage

    uris = [os.fspath(u) for u in (path if isinstance(path, (list, tuple)) else [path])]
    if not uris:
        raise ValueError("No URIs provided")

    client_config, anon = kwargs.get("client_config"), kwargs.get("anon")
    versions, in_memory = kwargs.get("versions", True), kwargs.get("in_memory", False)
    base_session = Session.get(session, in_memory=in_memory)
    tbl_session = session_for(base_session, {}, in_memory)
    tbl_session.catalog.source_client_configs = dict(
        base_session.catalog.source_client_configs
    )
    tables: list[tuple[File, Any, Any, File]] = []
    by_source: dict[str, list[str]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for u in uris:
        routes = storage_sources([u])
        is_glob = len(routes) > 1 or routes[0][1] is not None
        if is_glob and version is not None:
            raise DatasetPrepareError("lance", f"Cannot specify version for glob '{u}'")
        route = routes[0][0].rstrip("/\\")
        target = File.at(route, session=tbl_session)
        src = target.source
        if (cfg := configs.get(src)) is None:
            base = (
                base_session.catalog.client_config_for(src)
                if client_config is None
                else client_config
            )
            _, cfg = client_configs([route], base_session, anon, base, versions)
            configs[src] = tbl_session.catalog.source_client_configs[src] = cfg
        if not is_glob and (opened := _open_lance_table(target, version, tbl_session)):
            tables.append(opened)
        else:
            by_source.setdefault(src, []).append(u)

    chain = None
    if by_source:
        chain = reduce(
            lambda a, b: a.union(b),
            (
                read_storage(
                    u_list,
                    session=base_session,
                    settings=settings,
                    **{**kwargs, "client_config": c, "anon": c.get("anon")},
                )
                for s, u_list in by_source.items()
                for c in (configs[s],)
            ),
        )
        if not tables:
            return chain.parse_tabular(
                output, column, model_name, source, columns=columns, format="lance"
            )

    table_records: list[dict[str, Any]] = []
    table_schemas: list[tuple[str, Any]] = []
    for t, ds, schema, m in tables:
        bases = ds.base_paths()
        frags = [(f.fragment_id, f.data_files()[0]) for f in ds.get_fragments()] or [
            (None, None)
        ]
        for fid, df in frags:
            bp = bases[df.base_id] if df and df.base_id is not None else None
            b = File.at(bp.path, session=tbl_session) if bp else t
            d = "" if bp and not bp.is_dataset_root else "data/"
            f_path = f"{b.path}/{d}{df.path}".lstrip("/") if df else t.path
            sz = (df.file_size_bytes or 0) if df else 0
            row_file = File(source=b.source, path=f_path, size=sz)
            tbl = {
                "source": t.source,
                "path": t.path,
                "version": ds.version,
                "manifest": [m.path, m.etag],
                "fragment": fid,
            }
            table_records.append({"file": row_file, "lance_fragment": json.dumps(tbl)})
        table_schemas.append((table_records[-len(frags)]["file"].path, schema))
    if chain is not None:
        chain = chain.mutate(lance_fragment="") if chain.count() else None
    gen, out = _tabular_generator(
        chain, output, column, model_name, source, columns, schemas=table_schemas
    )
    tbl_chain = read_records(
        table_records,
        schema={"file": File, "lance_fragment": str},
        session=tbl_session,
        settings=settings,
        in_memory=in_memory,
    )
    chain = tbl_chain.union(chain) if chain is not None else tbl_chain
    return chain.gen(gen, params=["file", "lance_fragment"], output=out)  # type: ignore[arg-type]
