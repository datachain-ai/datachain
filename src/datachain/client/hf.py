import asyncio
import functools
import posixpath
import re
from contextvars import ContextVar
from threading import Event
from typing import Any

from huggingface_hub.utils import EntryNotFoundError

from datachain.lib.file import File

from .fsspec import Client

COMMIT = re.compile(r"[0-9a-f]{40}")
_EXPECTED_SIZE: ContextVar[int | None] = ContextVar("_EXPECTED_SIZE", default=None)


class classproperty:  # noqa: N801
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


def _wrap_class(sync_fs_class):
    """
    Analog of `AsyncFileSystemWrapper.wrap_class` from fsspec, but sets
    asynchronous to False by default. This is similar to other Async FS
    we initialize. E.g. it means we don't break things in Jupyter where code
    run in async.

    This also fixes write operations by ensuring they are properly forwarded
    to the underlying filesystem without async buffering issues.
    """
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    class GeneratedAsyncFileSystemWrapper(AsyncFileSystemWrapper):
        def __init__(self, *args, **kwargs):
            sync_fs = sync_fs_class(*args, **kwargs)
            super().__init__(sync_fs, asynchronous=False)

        def open(self, path, mode="rb", **kwargs):
            # Override open to ensure write operations work correctly.
            # It seems to be a bug in the fsspec wrapper. It avoids
            # wrapping open() explicitly but also doesn't redirect it to
            # sync filesystem.
            return self.sync_fs.open(path, mode, **kwargs)

    GeneratedAsyncFileSystemWrapper.__name__ = f"Async{sync_fs_class.__name__}Wrapper"
    return GeneratedAsyncFileSystemWrapper


@functools.cache
def get_hf_filesystem_cls():
    import fsspec
    from packaging.version import Version, parse

    fsspec_version = parse(fsspec.__version__)
    minver = Version("2024.12.0")

    if fsspec_version < minver:
        raise ImportError(
            f"datachain requires 'fsspec>={minver}' but version "
            f"{fsspec_version} is installed."
        )

    from .hf_file_system import PinnedHfFileSystem

    fs_cls = _wrap_class(PinnedHfFileSystem)
    # AsyncFileSystemWrapper does not set class properties, so we set them back.
    fs_cls.protocol = PinnedHfFileSystem.protocol
    return fs_cls


class HfClient(Client):
    PREFIX = "hf://"
    protocol = "hf"
    CREDENTIAL_KEYS = frozenset({"token"})

    @classproperty
    def FS_CLASS(cls):  # noqa: N802, N805
        return get_hf_filesystem_cls()

    @classmethod
    def create_fs(cls, **kwargs):
        if kwargs.pop("anon", False):
            kwargs["token"] = False
        return super().create_fs(**kwargs)

    def _is_version_pinned(self, file: "File") -> bool:
        return bool(COMMIT.fullmatch(file.version))

    def _version_kwargs(self, version_id: str | None) -> dict[str, Any]:
        return {"revision": version_id} if version_id else {}

    def _file_info_kwargs(self, version_id: str | None = None) -> dict[str, Any]:
        return {
            **self._version_kwargs(version_id),
            "expand_info": True,
        }

    def _ls_dir_kwargs(self) -> dict[str, Any]:
        return {"expand_info": True}

    def _open_remote(self, file: "File", block_size: int | None = None):
        self.check_unchanged(file)
        return self.fs.open(
            file.get_fs_path(),
            **self._block_size_kwargs(block_size),
            **self._version_kwargs(file.version),
            **({"size": file.size} if file.size and file.size > 0 else {}),
        )

    async def get_current_etag(self, file: "File") -> str:
        full_path = file.get_fs_path()
        info = await self.fs._info(full_path, **self._version_kwargs(file.version))
        etag = info.get("blob_id") or info.get("xet_hash")
        if etag is None:
            info = await self.fs._info(
                full_path, **self._file_info_kwargs(file.version)
            )
            etag = info.get("blob_id") or info.get("xet_hash")
        return etag or ""

    def _is_bucket(self, info: dict[str, Any] | None = None) -> bool:
        return (
            self.name == "buckets"
            or self.name.startswith("buckets/")
            or (
                info is not None
                and "last_commit" not in info
                and any(k in info for k in ("xet_hash", "mtime", "uploaded_at"))
            )
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        last_commit = v.get("last_commit")
        if last_commit is None and not self._is_bucket(v):
            raise RuntimeError(
                "huggingface_hub returned file metadata without last_commit; "
                "expected expanded HF metadata for file version tracking"
            )
        last_modified = (
            last_commit.date if last_commit else v.get("mtime") or v.get("uploaded_at")
        )
        return File(
            source=self.uri,
            path=path,
            size=v.get("size") or 0,
            etag=v.get("blob_id") or v.get("xet_hash") or "",
            version=last_commit.oid if last_commit else "",
            **({"last_modified": last_modified} if last_modified else {}),
        )

    CHUNK_SIZE = 8 * 1024 * 1024

    async def _put_in_cache(self, file: "File", *, callback=None) -> None:
        has_size = file.size > 0 or (file.size == 0 and bool(file.etag))
        token = _EXPECTED_SIZE.set(file.size if has_size else None)
        try:
            await super()._put_in_cache(file, callback=callback)
        finally:
            _EXPECTED_SIZE.reset(token)

    async def get_file(
        self,
        lpath,
        rpath,
        callback,
        version_id: str | None = None,
        expected_size: int | None = None,
    ):
        fs = self.fs.sync_fs
        p = await asyncio.to_thread(
            fs.resolve_path, lpath, **self._version_kwargs(version_id)
        )
        if not getattr(p, "repo_id", None):
            return await super().get_file(lpath, rpath, callback, version_id)
        expected = expected_size if expected_size is not None else _EXPECTED_SIZE.get()
        stop = Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                self._stream, p.unresolve(), rpath, callback, stop, expected
            )
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            cleanup = asyncio.gather(task, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    pass
            raise

    def _stream(
        self,
        path: str,
        dst: str,
        callback,
        stop=None,
        expected_size: int | None = None,
    ) -> None:
        try:
            with (
                self.fs.sync_fs.open(path, "rb", block_size=0) as src,
                open(dst, "wb") as out,
            ):
                written = 0
                while (stop is None or not stop.is_set()) and (
                    chunk := src.read(self.CHUNK_SIZE)
                ):
                    out.write(chunk)
                    written += len(chunk)
                    callback.relative_update(len(chunk))
                if stop is None or not stop.is_set():
                    for expected in (expected_size, getattr(src, "size", None)):
                        if isinstance(expected, int) and written != expected:
                            raise OSError(
                                f"Expected {expected} bytes for {path}, got {written}"
                            )
        except EntryNotFoundError as e:
            raise FileNotFoundError(path) from e

    async def _check_unchanged(
        self, file: "File", *, info: dict[str, Any] | None = None
    ) -> None:
        if self._is_version_pinned(file):
            return
        fs_path = file.get_fs_path()
        if not self._has_version(file) and "@" in fs_path:
            p = await asyncio.to_thread(
                self.fs.sync_fs.resolve_path,
                fs_path,
            )
            if COMMIT.fullmatch(getattr(p, "revision", None) or ""):
                return
        await super()._check_unchanged(file, info=info)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
