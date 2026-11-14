import asyncio
import functools
import posixpath
from typing import Any

from datachain.lib.file import File, FileRow

from .fsspec import Client, ResultQueue

SIBLINGS_LIMIT = 100_000  # the most files the Hub returns with a repository's info


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

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        await result_queue.put(await asyncio.to_thread(self._list, start_prefix))

    def _list(self, prefix: str) -> list[FileRow]:
        """Files under ``prefix``: from the repository info, one request, unless
        the Hub caps its file list; else from a recursive listing, which the Hub
        pages through a whole tree at a time."""
        prefix = prefix.strip("/")
        path = f"{self.name}/{prefix}".rstrip("/")
        fs = self.fs.sync_fs
        p = fs.resolve_path(path)
        subpath = getattr(p, "path_in_repo", None) or getattr(p, "path", "")
        siblings = None
        if getattr(p, "repo_id", None):  # a repository, not a bucket
            info = fs._api.repo_info(
                p.repo_id,
                repo_type=p.repo_type,
                revision=p.revision,
                files_metadata=True,
            )
            siblings = getattr(info, "siblings", None) or []
        if siblings is not None and len(siblings) < SIBLINGS_LIMIT:
            under = f"{subpath}/" if subpath else ""
            entries = [(s.rfilename, s.size, s.blob_id) for s in siblings]
        else:
            under = f"{p.unresolve()}/"
            found = fs.find(path, detail=True)
            entries = [(k, v.get("size"), v.get("blob_id")) for k, v in found.items()]
        base = f"{prefix}/" if prefix else ""
        rows = [
            FileRow(self.uri, base + name[len(under) :], size or 0, etag=etag or "")
            for name, size, etag in entries
            if name.startswith(under)
        ]
        if not rows and subpath:
            raise FileNotFoundError(path)
        return rows

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        """No version or date: for those the Hub finds each file's last commit,
        which makes listing slow. The etag, the git blob id, follows the content."""
        return File(
            source=self.uri, path=path, size=v["size"], etag=v.get("blob_id") or ""
        )

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
