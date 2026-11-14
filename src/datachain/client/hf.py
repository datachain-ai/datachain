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

    from huggingface_hub import HfFileSystem

    fs_cls = _wrap_class(HfFileSystem)
    # AsyncFileSystemWrapper does not set class properties, so we need to set them back.
    fs_cls.protocol = HfFileSystem.protocol
    return fs_cls


class HfClient(Client):
    PREFIX = "hf://"
    protocol = "hf"

    @classproperty
    def FS_CLASS(cls):  # noqa: N802, N805
        return get_hf_filesystem_cls()

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
        p = await asyncio.to_thread(fs.resolve_path, lpath)
        if not getattr(p, "repo_id", None):  # a bucket
            return await super().get_file(lpath, rpath, callback, version_id)
        await asyncio.to_thread(self._stream, lpath, p, rpath, callback)

    def _stream(self, path: str, p, dst: str, callback) -> None:
        """One GET of the file's URL. HfFileSystem's own download, and its
        ``url()``, first list the folder: an API request, rate limited."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.hf_file_system import HfFileSystemStreamFile

        fs = self.fs.sync_fs
        url = hf_hub_url(
            p.repo_id,
            p.path_in_repo,
            repo_type=p.repo_type,
            revision=p.revision,
            endpoint=fs.endpoint,
        )
        src = HfFileSystemStreamFile(fs, path)
        src.url = lambda: url  # type: ignore[method-assign]
        with src, open(dst, "wb") as out:
            while chunk := src.read(CHUNK_SIZE):
                out.write(chunk)
                callback.relative_update(len(chunk))

    async def _check_unchanged(
        self, file: "File", *, info: dict[str, Any] | None = None
    ) -> None:
        p = await asyncio.to_thread(self.fs.sync_fs.resolve_path, file.get_fs_path())
        if not COMMIT.fullmatch(getattr(p, "revision", None) or ""):
            await super()._check_unchanged(file, info=info)  # a commit never changes

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
