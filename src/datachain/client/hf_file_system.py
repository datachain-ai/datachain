"""Hugging Face filesystem support, imported only for HF clients."""

from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from urllib.parse import quote

from huggingface_hub import HfFileSystem, hf_file_system, hf_hub_url
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_file_system import (
    SPECIAL_REFS_REVISION_REGEX,
    HfFileSystemStreamFile,
)
from huggingface_hub.utils import HfHubHTTPError


class HfPermissionError(PermissionError, FileNotFoundError):
    pass


def _restore_permission_error(
    cls: type[Exception], args: tuple, state: dict
) -> Exception:
    perm_cls = _permission_error_cls(cls)
    err = perm_cls.__new__(perm_cls, *args)
    Exception.__init__(err, *args)
    err.__dict__.update(state)
    return err


_PERMISSION_ERROR_CLASSES: dict[type[Exception], Any] = {}


def _permission_error_cls(cls: type[Exception]) -> Any:
    if (perm_cls := _PERMISSION_ERROR_CLASSES.get(cls)) is None:
        perm_cls = _PERMISSION_ERROR_CLASSES[cls] = type(
            cls.__name__,
            (cls, PermissionError),
            {
                "__module__": cls.__module__,
                "__reduce_ex__": lambda self, _proto: (
                    _restore_permission_error,
                    (cls, self.args, self.__dict__.copy()),
                ),
            },
        )
    return perm_cls


def _mark_permission_error(err: BaseException | None) -> bool:
    if isinstance(err, HfHubHTTPError):
        status = getattr(getattr(err, "response", None), "status_code", None)
        if status in (401, 403):
            if not isinstance(err, PermissionError):
                err.__class__ = _permission_error_cls(type(err))
            return True
    return False


@contextmanager
def _translate_auth_error():
    try:
        yield
    except HfHubHTTPError as e:
        _mark_permission_error(e)
        raise
    except FileNotFoundError as e:
        if _mark_permission_error(e.__cause__) and not isinstance(e, PermissionError):
            raise HfPermissionError(*e.args) from e.__cause__
        raise


def _file_url(file_obj) -> str:
    resolved = file_obj.resolved_path
    if not getattr(resolved, "repo_id", None):
        return file_obj.fs.url(file_obj.path)
    return hf_hub_url(
        resolved.repo_id,
        resolved.path_in_repo,
        repo_type=resolved.repo_type,
        revision=resolved.revision,
        endpoint=file_obj.fs.endpoint,
    )


class _PinnedStreamFile(HfFileSystemStreamFile):
    def url(self) -> str:
        return _file_url(self)

    def _open_connection(self):
        with _translate_auth_error():
            super()._open_connection()
        if getattr(self.response, "status_code", None) == 200:
            headers = getattr(self.response, "headers", None) or {}
            if (
                headers.get("Content-Encoding", "identity").lower() == "identity"
                and (length := headers.get("Content-Length"))
                and str(length).isdigit()
            ):
                self.size = int(length)
            if self.loc > 0:
                resume_loc, self.loc = self.loc, 0
                super().read(resume_loc)

    def close(self):
        try:
            super().close()
        finally:
            if exit_stack := getattr(self, "_exit_stack", None):
                exit_stack.close()


class PinnedHfFileSystem(HfFileSystem):
    """Use the listed commit even when a URI names a branch or tag."""

    def resolve_path(self, path, revision=None):
        path = self._strip_protocol(path)
        with _translate_auth_error():
            if revision is None or path.startswith("buckets/"):
                return super().resolve_path(path)

            if parse_uri := getattr(hf_file_system, "parse_hf_uri", None):
                parsed = parse_uri(f"hf://{path}")
                uri = replace(parsed, revision=revision).to_uri()
                return super().resolve_path(uri, revision=revision)

            path, separator, rest = path.partition("@")
            prefix = next(
                (
                    p
                    for p in REPO_TYPES_URL_PREFIXES.values()
                    if p and path.startswith(p)
                ),
                "",
            )
            root = path[len(prefix) :]
            has_uri_revision = separator and "/" in rest and root.count("/") < 2
            resolve_path = (
                f"{path}@{quote(revision, safe='')}/{rest}"
                if has_uri_revision
                else path
            )
            resolved = super().resolve_path(resolve_path, revision=revision)
        if not separator:
            return resolved
        if (resolved.path_in_repo and not has_uri_revision) or path.endswith("/"):
            path_in_repo = f"{resolved.path_in_repo}@{rest}"
        else:
            match = SPECIAL_REFS_REVISION_REGEX.match(rest)
            path_in_repo = (
                rest[match.end() :].lstrip("/") if match else rest.partition("/")[2]
            )
        return replace(
            resolved, path_in_repo=path_in_repo, _raw_revision=quote(revision, safe="")
        )

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        with _translate_auth_error():
            effective_block_size = (
                block_size
                if block_size is not None
                else getattr(self, "block_size", None)
            )
            if effective_block_size == 0 and "a" not in mode:
                kwargs["block_size"] = 0
                return _PinnedStreamFile(
                    self, path, mode=mode, revision=revision, **kwargs
                )
            return super()._open(
                path, mode=mode, block_size=block_size, revision=revision, **kwargs
            )

    def get_file(self, rpath, *args, **kwargs):
        revision = kwargs.get("revision")
        if revision is not None:
            rpath = self.resolve_path(rpath, revision=revision).unresolve()
        with _translate_auth_error():
            return super().get_file(rpath, *args, **kwargs)

    def info(self, path, *args, **kwargs):
        kwargs.setdefault("expand_info", bool(getattr(self, "expand_info", False)))
        with _translate_auth_error():
            return super().info(path, *args, **kwargs)

    def ls(self, path, *args, **kwargs):
        with _translate_auth_error():
            return super().ls(path, *args, **kwargs)

    def find(self, path, *args, **kwargs):
        with _translate_auth_error():
            return super().find(path, *args, **kwargs)
