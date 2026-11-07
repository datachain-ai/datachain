import hashlib
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager

from datachain.lib.file import File, FileError, TarVFile


def build_tar_member(parent: File, info: tarfile.TarInfo) -> File:
    etag_string = "-".join([parent.etag, info.name, str(info.mtime)])
    etag = hashlib.md5(etag_string.encode(), usedforsecurity=False).hexdigest()
    return File(
        source=parent.source,
        path=f"{parent.path}/{info.name}",
        version=parent.version,
        size=info.size,
        etag=etag,
        location=[
            {
                "vtype": TarVFile.get_vtype(),
                "parent": parent.model_dump_custom(),
                "size": info.size,
                "offset": info.offset_data,
            }
        ],
    )


@contextmanager
def open_tar(file: File) -> Iterator[tarfile.TarFile]:
    """Open an uncompressed tar; compression breaks member offsets."""
    with file.open() as fd:
        try:
            tar = tarfile.open(fileobj=fd, mode="r:")  # noqa: SIM115
        except tarfile.ReadError as exc:
            raise FileError(
                f"not an uncompressed tar archive ({exc}); "
                "compressed tars are not supported",
                file.source,
                file.path,
            ) from exc
        with tar:
            yield tar


def process_tar(file: File) -> Iterator[File]:
    with open_tar(file) as tar:
        for entry in tar.getmembers():
            if entry.isfile():
                yield build_tar_member(file, entry)
