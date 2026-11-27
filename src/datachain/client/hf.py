import asyncio
import functools
import posixpath
import re
import threading
from typing import Any

from datachain.lib.file import File, FileRow
from datachain.progress import tqdm

from .fsspec import Client, ResultQueue

COMMIT = re.compile(r"[0-9a-f]{40}")


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

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        with tqdm(desc=f"Listing {self.uri}", unit=" objects", leave=False) as pbar:

            def emit(page: list[FileRow]) -> None:
                if stop.is_set():
                    return
                pbar.update(len(page))
                put = asyncio.run_coroutine_threadsafe(result_queue.put(page), loop)
                while not stop.is_set():  # a stopped reader leaves the queue full
                    try:
                        put.result(timeout=0.1)
                        return
                    except TimeoutError:
                        pass
                put.cancel()

            try:
                await asyncio.to_thread(self._list, start_prefix, emit, stop)
            finally:
                stop.set()

    def _list(
        self,
        prefix: str,
        emit=lambda _: None,
        stop: threading.Event | None = None,
    ) -> None:
        prefix = prefix.strip("/")
        path = f"{self.name}/{prefix}".rstrip("/")
        fs = self.fs.sync_fs
        resolved = fs.resolve_path(path)
        repository = bool(getattr(resolved, "repo_id", None))
        if repository and "@" in path and not getattr(resolved, "_raw_revision", None):
            resolved = fs.resolve_path(path, revision=resolved.revision)
        path = resolved.unresolve()
        under = f"{path}/"
        base = f"{prefix}/" if prefix else ""
        expand = {"expand_info": True} if repository else {}
        # fresh: the SDK caches trees, and a branch or a bucket changes
        found = fs.find(path, detail=True, refresh=True, **expand)
        page: list[FileRow] = []
        matched = False
        for name, info in found.items():
            if stop is not None and stop.is_set():
                return
            if name == path:
                info = fs.info(name, refresh=True, **expand)
                relative = prefix
            elif name.startswith(under):
                relative = base + name[len(under) :]
            else:
                continue
            matched = True
            size, etag, version, last_modified = self._entry_metadata(
                info, repository=repository
            )
            metadata = {
                **({"version": version} if version else {}),
                **({"last_modified": last_modified} if last_modified else {}),
            }
            page.append(
                FileRow(
                    self.uri,
                    relative,
                    size,
                    etag=etag,
                    **metadata,
                )
            )
            if len(page) >= self.LIST_PAGE_SIZE:
                emit(page)
                page = []
        if not matched:
            fs.info(path, refresh=True)
            return
        if page:
            emit(page)

    def _open_remote(self, file: "File", block_size: int | None = None):
        self.check_unchanged(file)
        return self.fs.open(
            file.get_fs_path(),
            **self._block_size_kwargs(block_size),
            **self._version_kwargs(file.version),
            **({"size": file.size} if file.size and file.size > 0 else {}),
        )

    async def get_current_etag(self, file: "File") -> str:
        # a listing fills the SDK's cache, and a branch or a bucket can change
        return await asyncio.to_thread(
            self.fs.sync_fs.current_etag, file.get_fs_path(), file.version or None
        )

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

    def _entry_metadata(
        self, info: dict[str, Any], *, repository: bool | None = None
    ) -> tuple[int, str, str, Any]:
        last_commit = info.get("last_commit")
        is_repo = not self._is_bucket(info) if repository is None else repository
        if is_repo and last_commit is None:
            raise RuntimeError(
                "huggingface_hub returned file metadata without last_commit; "
                "expected expanded HF metadata for file version tracking"
            )
        last_modified = (
            last_commit.date
            if last_commit
            else info.get("mtime") or info.get("uploaded_at")
        )
        return (
            info.get("size") or 0,
            info.get("blob_id") or info.get("xet_hash") or "",
            last_commit.oid if last_commit else "",
            last_modified,
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        size, etag, version, last_modified = self._entry_metadata(v)
        return File(
            source=self.uri,
            path=path,
            size=size,
            etag=etag,
            version=version,
            **({"last_modified": last_modified} if last_modified else {}),
        )

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
