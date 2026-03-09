import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import attrs

from datachain.dataset import StorageURI
from datachain.lib.file import File
from datachain.utils import TIME_ZERO, time_to_str

if TYPE_CHECKING:
    from typing_extensions import Self

    from datachain.client import Client


class DirType:
    FILE = 0
    DIR = 1
    TAR_ARCHIVE = 5


class DirTypeGroup:
    """
    Groups of DirTypes for selecting storage nodes or dataset entries.

    When filtering with FILE and DIR together or alternatively when
    using SUBOBJ_FILE and SUBOBJ_DIR together, we achieve a
    filesystem-compatible view of a storage location. Such a view
    avoids path conflicts and could be downloaded as a directory tree.

    FILE, DIR
      The respective types which appear on the indexed filesystem or
      object store as a file or directory.

    SUBOBJ_FILE, SUBOBJ_DIR
      The respective types that we want to consider to be a file or
      directory when including subobjects which are generated from other
      files. In this case, we treat tar archives as directories so tar
      subobjects (TAR_FILE) can be viewed under the directory tree of
      the parent tar archive.
    """

    FILE = (DirType.FILE, DirType.TAR_ARCHIVE)
    DIR = (DirType.DIR,)
    SUBOBJ_FILE = (DirType.FILE,)
    SUBOBJ_DIR = (DirType.DIR, DirType.TAR_ARCHIVE)


@attrs.define
class Node:
    sys__id: int = 0
    sys__rand: int = 0
    path: str = ""
    etag: str = ""
    version: str | None = None
    is_latest: bool = True
    last_modified: datetime | None = None
    size: int = 0
    location: str | None = None
    source: StorageURI = StorageURI("")  # noqa: RUF009
    dir_type: int = DirType.FILE

    @property
    def is_dir(self) -> bool:
        return self.dir_type == DirType.DIR

    @property
    def is_container(self) -> bool:
        return self.dir_type in DirTypeGroup.SUBOBJ_DIR

    @property
    def is_downloadable(self) -> bool:
        return bool(not self.is_dir and self.name)

    def append_to_file(self, fd, path: str):
        fd.write(f"- name: {path}\n")
        fd.write(f"  etag: {self.etag}\n")
        version = self.version
        if version:
            fd.write(f"  version: {self.version}\n")
        fd.write(f"  last_modified: '{time_to_str(self.last_modified)}'\n")
        size = self.size
        fd.write(f"  size: {self.size}\n")
        return size

    @property
    def full_path(self) -> str:
        if self.is_dir and self.path:
            return self.path + "/"
        return self.path

    def to_file(self, source: StorageURI | None = None) -> File:
        if source is None:
            source = self.source
        return File(
            source=source,
            path=self.path,
            size=self.size,
            version=self.version or "",
            etag=self.etag,
            is_latest=self.is_latest,
            location=self.location,
            last_modified=self.last_modified or TIME_ZERO,
        )

    @classmethod
    def from_file(cls, f: File) -> "Self":
        return cls(
            source=StorageURI(f.source),
            path=f.path,
            etag=f.etag,
            is_latest=f.is_latest,
            size=f.size,
            last_modified=f.last_modified,
            version=f.version,
            location=str(f.location) if f.location else None,
            dir_type=DirType.FILE,
        )

    @classmethod
    def from_row(cls, d: dict[str, Any], file_prefix: str = "file") -> "Self":
        def _dval(field_name: str):
            return d.get(f"{file_prefix}__{field_name}")

        return cls(
            sys__id=d["sys__id"],
            sys__rand=d["sys__rand"],
            source=_dval("source"),
            path=_dval("path"),
            etag=_dval("etag"),
            is_latest=_dval("is_latest"),
            size=_dval("size"),
            last_modified=_dval("last_modified"),
            version=_dval("version"),
            location=_dval("location"),
            dir_type=DirType.FILE,
        )

    @classmethod
    def from_dir(cls, path, **kwargs) -> "Node":
        return cls(sys__id=-1, dir_type=DirType.DIR, path=path, **kwargs)

    @classmethod
    def root(cls) -> "Node":
        return cls(sys__id=-1, dir_type=DirType.DIR)

    @property
    def name(self):
        return self.path.rsplit("/", 1)[-1]

    @property
    def parent(self):
        split = self.path.rsplit("/", 1)
        if len(split) <= 1:
            return ""
        return split[0]


def get_path(parent: str, name: str):
    return f"{parent}/{name}" if parent else name


@attrs.define
class NodeWithPath:
    n: Node
    path: list[str] = attrs.field(factory=list)

    def append_to_file(self, fd):
        return self.n.append_to_file(fd, "/".join(self.path))

    @property
    def full_path(self) -> str:
        path = "/".join(self.path)
        if self.n.is_dir and path:
            path += "/"
        return path

    def instantiate(
        self, client: "Client", output: str, progress_bar, *, force: bool = False
    ):
        dst = os.path.join(output, *self.path)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        file = self.n.to_file(client.uri)
        client.instantiate_object(file, dst, progress_bar, force)


TIME_FMT = "%Y-%m-%d %H:%M"


def long_line_str(name: str, timestamp: datetime | None) -> str:
    if timestamp is None:
        time = "-"
    else:
        time = timestamp.strftime(TIME_FMT)
    return f"{time: <19} {name}"
