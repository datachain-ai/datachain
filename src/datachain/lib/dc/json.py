import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import cloudpickle

from datachain.lib import meta_formats
from datachain.lib.arrow import PARTIAL_READ_PREFETCH_MAX_SIZE
from datachain.lib.data_model import DataType
from datachain.lib.file import File, FileType
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


class _ReadJson(Generator):
    def __init__(self, parse: Callable, prefetch_max_size: int | None):
        super().__init__()
        # The function's signature gives the step its params and output.
        self.process = parse  # type: ignore[method-assign]
        self.prefetch_max_size = prefetch_max_size


def read_json(
    path: str | os.PathLike[str],
    type: FileType = "text",
    spec: DataType | None = None,
    schema_from: str | None = "auto",
    jmespath: str | None = None,
    column: str | None = "",
    model_name: str | None = None,
    format: str | None = "json",
    nrows: int | None = None,
    **kwargs,
) -> "DataChain":
    """Get data from JSON. It returns the chain itself.

    Parameters:
        path: storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///"
        type: read file as "binary", "text", or "image" data. Default is "text".
        spec: optional Data Model
        schema_from: path to sample to infer spec (if schema not provided)
        column: generated column name
        model_name: optional generated model name
        format: "json", "jsonl"
        jmespath: optional JMESPATH expression to reduce JSON
        nrows: optional row limit per file, for jsonl and JSON arrays

    Example:
        infer JSON schema from data, reduce using JMESPATH
        ```py
        import datachain as dc
        chain = dc.read_json("gs://json", jmespath="key1.key2")
        ```

        infer JSON schema from a particular path
        ```py
        import datachain as dc
        chain = dc.read_json("gs://json_ds", schema_from="gs://json/my.json")
        ```
    """
    from .storage import read_storage

    if schema_from == "auto":
        schema_from = os.fspath(path)

    def jmespath_to_name(s: str):
        name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
        return s[:name_end]

    if (not column) and jmespath:
        column = jmespath_to_name(jmespath)
    if not column:
        column = format
    chain = read_storage(uri=path, type=type, **kwargs)
    parse = meta_formats.read_meta(
        schema_from=schema_from,
        format=format,
        spec=spec,
        model_name=model_name,
        jmespath=jmespath,
        nrows=nrows,
        session=kwargs.get("session"),
    )
    cloudpickle.register_pickle_by_value(meta_formats)

    # Prefetch downloads whole files, so reading some JSONL lines prefetches only
    # small ones. JSON is parsed whole anyway.
    partial = nrows and format == "jsonl"
    max_size = PARTIAL_READ_PREFETCH_MAX_SIZE if partial else None
    return chain.gen(params={"file": File}, **{column: _ReadJson(parse, max_size)})  # type: ignore[arg-type]
