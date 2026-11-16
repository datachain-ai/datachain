import asyncio
import functools
import posixpath
import re
from typing import Any

from datachain.lib.file import File

from .fsspec import Client

CHUNK_SIZE = 8 * 1024 * 1024
COMMIT = re.compile("[0-9a-f]{40}")


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
    PINNED_VERSION_IS_IMMUTABLE = True

    @classproperty
    def FS_CLASS(cls):  # noqa: N802, N805
        return get_hf_filesystem_cls()

    def _version_kwargs(self, version_id: str | None) -> dict[str, Any]:
        return {"revision": version_id} if version_id else {}

    def _file_info_kwargs(self, version_id: str | None = None) -> dict[str, Any]:
        return {
            **self._version_kwargs(version_id),
            "expand_info": True,
        }

    def _ls_dir_kwargs(self) -> dict[str, Any]:
        return {"expand_info": True}

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        last_commit = v.get("last_commit")
        if last_commit is None:
            raise RuntimeError(
                "huggingface_hub returned file metadata without last_commit; "
                "expected expanded HF metadata for file version tracking"
            )

        return File(
            source=self.uri,
            path=path,
            size=v["size"],
            etag=v.get("blob_id", ""),
            version=last_commit.oid,
            last_modified=last_commit.date,
        )

    async def get_file(self, lpath, rpath, callback, version_id: str | None = None):
        fs = self.fs.sync_fs
        p = await asyncio.to_thread(
            fs.resolve_path, lpath, **self._version_kwargs(version_id)
        )
        if not getattr(p, "repo_id", None):
            return await super().get_file(lpath, rpath, callback, version_id)
        task = asyncio.create_task(
            asyncio.to_thread(self._stream, p.unresolve(), rpath, callback)
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cleanup = asyncio.gather(task, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    pass
            raise

    def _stream(self, path: str, dst: str, callback) -> None:
        with (
            self.fs.sync_fs.open(path, "rb", block_size=0) as src,
            open(dst, "wb") as out,
        ):
            while chunk := src.read(CHUNK_SIZE):
                out.write(chunk)
                callback.relative_update(len(chunk))

    async def _check_unchanged(
        self, file: "File", *, info: dict[str, Any] | None = None
    ) -> None:
        if COMMIT.fullmatch(file.version):
            return
        p = await asyncio.to_thread(
            self.fs.sync_fs.resolve_path,
            file.get_fs_path(),
            **self._version_kwargs(file.version),
        )
        if not COMMIT.fullmatch(getattr(p, "revision", None) or ""):
            await super()._check_unchanged(file, info=info)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
