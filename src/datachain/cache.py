import functools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import iscoroutinefunction
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


def _readonly_guard(method):
    """Decorator that prevents calling a Cache method when readonly."""
    if iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(self, *args, **kwargs):
            if self._readonly:
                raise RuntimeError(
                    f"cannot call {method.__name__}() on a read-only cache"
                )
            return await method(self, *args, **kwargs)

        return async_wrapper

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._readonly:
            raise RuntimeError(f"cannot call {method.__name__}() on a read-only cache")
        return method(self, *args, **kwargs)

    return wrapper


class Cache:  # noqa: PLW1641
    def __init__(
        self,
        cache_dir: str,
        tmp_dir: str,
        fallback: "Cache | None" = None,
        readonly: bool = False,
    ):
        self.odb = LocalHashFileDB(
            LocalFileSystem(),
            cache_dir,
            tmp_dir=tmp_dir,
        )
        # Read-only fallback consulted on cache misses.
        self._fallback = fallback
        self._readonly = readonly

    def __eq__(self, other) -> bool:
        return self.odb == other.odb

    @property
    def cache_dir(self):
        return self.odb.path

    @property
    def tmp_dir(self):
        return self.odb.tmp_dir

    def as_readonly(self) -> "Cache":
        """Return a read-only Cache backed by the same files."""
        return Cache(self.cache_dir, self.tmp_dir, readonly=True)

    def get_path(self, file: "File") -> str | None:
        if self.odb.exists(file.get_hash()):
            return self.path_from_checksum(file.get_hash())
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

    @_readonly_guard
    def remove(self, file: "File") -> None:
        if self.odb.exists(file.get_hash()):
            self.odb.delete(file.get_hash())

    @_readonly_guard
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

    @_readonly_guard
    def store_data(self, file: "File", contents: bytes) -> None:
        self.odb.add_bytes(file.get_hash(), contents)

    @_readonly_guard
    def clear(self) -> None:
        """
        Completely clear the cache.
        """
        self.odb.clear()

    @_readonly_guard
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
