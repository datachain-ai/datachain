import os
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from dvc_data.hashfile.db.local import LocalHashFileDB
from dvc_objects.fs.local import LocalFileSystem
from dvc_objects.fs.utils import remove
from fsspec.callbacks import Callback, TqdmCallback

if TYPE_CHECKING:
    from datachain.client import Client
    from datachain.lib.file import File


def try_scandir(path):
    try:
        with os.scandir(path) as it:
            yield from it
    except OSError:
        pass


def get_temp_cache(
    tmp_dir: str,
    prefix: str | None = None,
    fallback: "Cache | None" = None,
) -> "Cache":
    cache_dir = mkdtemp(prefix=prefix, dir=tmp_dir)
    return Cache(cache_dir, tmp_dir=tmp_dir, fallback=fallback)


@contextmanager
def temporary_cache(
    tmp_dir: str,
    prefix: str | None = None,
    delete: bool = True,
    fallback: "Cache | None" = None,
) -> Iterator["Cache"]:
    cache = get_temp_cache(tmp_dir, prefix=prefix, fallback=fallback)
    try:
        yield cache
    finally:
        if delete:
            cache.destroy()


class Cache:  # noqa: PLW1641
    def __init__(
        self,
        cache_dir: str,
        tmp_dir: str,
        fallback: "Cache | None" = None,
    ):
        self.odb = LocalHashFileDB(
            LocalFileSystem(),
            cache_dir,
            tmp_dir=tmp_dir,
        )
        # Read-only fallback consulted on cache misses.
        self._fallback = fallback

    def __eq__(self, other) -> bool:
        return self.odb == other.odb

    @property
    def cache_dir(self):
        return self.odb.path

    @property
    def tmp_dir(self):
        return self.odb.tmp_dir

    def as_readonly(self) -> "ReadonlyCache":
        """Return a read-only Cache backed by the same files."""
        return ReadonlyCache(self.cache_dir, self.tmp_dir)

    def get_path(self, file: "File") -> str | None:
        oid = file.get_hash()
        if self.odb.exists(oid):
            return self.path_from_checksum(oid)
        if self._fallback is not None:
            return self._fallback.get_path(file)
        return None

    def contains(self, file: "File") -> bool:
        if self.odb.exists(file.get_hash()):
            return True
        if self._fallback is not None:
            return self._fallback.contains(file)
        return False

    def path_from_checksum(self, checksum: str) -> str:
        assert checksum
        return self.odb.oid_to_path(checksum)

    def remove(self, file: "File") -> None:
        oid = file.get_hash()
        if self.odb.exists(oid):
            self.odb.delete(oid)

    async def download(
        self, file: "File", client: "Client", callback: Callback | None = None
    ) -> None:
        from dvc_objects.fs.utils import tmp_fname

        from_path = file.get_fs_path()
        odb_fs = self.odb.fs
        tmp_info = odb_fs.join(self.odb.tmp_dir, tmp_fname())  # type: ignore[arg-type]
        size = file.size
        if size < 0:
            size = await client.get_size(file)
        from datachain.progress import tqdm

        cb = callback or TqdmCallback(
            tqdm_kwargs={
                "desc": odb_fs.name(from_path),
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
                "leave": False,
            },
            tqdm_cls=tqdm,
            size=size,
        )
        try:
            await client.get_file(
                from_path, tmp_info, callback=cb, version_id=file.version
            )
        finally:
            if not callback:
                cb.close()

        try:
            oid = file.get_hash()
            self.odb.add(tmp_info, self.odb.fs, oid)
        finally:
            os.unlink(tmp_info)

    def store_data(self, file: "File", contents: bytes) -> None:
        self.odb.add_bytes(file.get_hash(), contents)

    def clear(self) -> None:
        """
        Completely clear the cache.
        """
        self.odb.clear()

    def destroy(self) -> None:
        # `clear` leaves the prefix directory structure intact.
        remove(self.cache_dir)

    def get_total_size(self) -> int:
        total = 0
        for subdir in try_scandir(self.odb.path):
            for file in try_scandir(subdir):
                try:
                    total += file.stat().st_size
                except OSError:
                    pass
        return total


class ReadonlyCache(Cache):
    """A read-only view over an existing cache directory.

    Used as a fallback for short-lived temp caches: writes/eviction stay
    scoped to the temp cache, reads see hits in the persistent one.
    """

    def _readonly_error(self, name: str) -> RuntimeError:
        return RuntimeError(f"cannot call {name}() on a read-only cache")

    def remove(self, file: "File") -> None:
        raise self._readonly_error("remove")

    async def download(
        self, file: "File", client: "Client", callback: Callback | None = None
    ) -> None:
        raise self._readonly_error("download")

    def store_data(self, file: "File", contents: bytes) -> None:
        raise self._readonly_error("store_data")

    def clear(self) -> None:
        raise self._readonly_error("clear")

    def destroy(self) -> None:
        raise self._readonly_error("destroy")
