import functools
import posixpath
from typing import Any

from datachain.lib.file import File

from .fsspec import Client


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

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
