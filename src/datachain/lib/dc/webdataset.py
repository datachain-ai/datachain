import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from datachain.lib.webdataset import WDSAllFile, WDSBasic, process_webdataset
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def read_webdataset(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    *,
    spec: type[WDSBasic] = WDSAllFile,
    core_extensions: Sequence[str] = ("jpg", "png"),
    encoding: str = "utf-8",
    column: str = "wds",
    session: Session | None = None,
    settings: dict | None = None,
    **kwargs: Any,
) -> "DataChain":
    """Generate a chain with one row per sample from WebDataset tar shards.

    Files in a shard that share a stem form one sample. The file with a core
    extension becomes `file`; the others fill the `spec` fields named after
    their extensions.

    Parameters:
        path: Shard file(s), directory or glob, e.g. `s3://`, `gs://`, `az://`,
            `hf://`. Only uncompressed `.tar` files are read.
        spec: Sample model, `WDSBasic` or a subclass of it.
        core_extensions: Extensions of the sample's main file.
        encoding: Encoding of text members.
        column: Created column name.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        kwargs: Parameters to pass to `read_storage`.

    Example:
        ```py
        import datachain as dc

        chain = dc.read_webdataset("gs://bucket/shards/", anon=True)
        chain.filter(dc.func.string.length("wds.txt") > 5).show()
        ```
    """
    from .datachain import C
    from .storage import read_storage

    chain = read_storage(path, session=session, settings=settings, **kwargs)
    wds = process_webdataset(core_extensions, spec, encoding)
    return chain.filter(C("file.path").glob("*.tar")).gen(wds, output={column: spec})
