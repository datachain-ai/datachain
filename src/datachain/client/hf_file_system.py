"""Hugging Face filesystem support, imported only for HF clients."""

import functools
from contextlib import contextmanager
from dataclasses import replace
from urllib.parse import quote

from huggingface_hub import HfFileSystem, hf_file_system, hf_hub_url
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_file_system import (
    SPECIAL_REFS_REVISION_REGEX,
    HfFileSystemStreamFile,
)
from huggingface_hub.utils import HfHubHTTPError


@functools.cache
def _permission_error_cls(cls: type[Exception]) -> type[Exception]:
    return type(cls.__name__, (cls, PermissionError), {"__module__": cls.__module__})


@contextmanager
def _translate_auth_error():
    try:
        yield
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403) and not isinstance(e, PermissionError):
            e.__class__ = _permission_error_cls(type(e))
        raise


class _PinnedStreamFile(HfFileSystemStreamFile):
    def _open_connection(self):
        with _translate_auth_error():
            return super()._open_connection()

    def close(self):
        try:
            super().close()
        finally:
            self._exit_stack.close()


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
                block_size if block_size is not None else self.block_size
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

    def url(self, path, revision=None, **kwargs):
        resolved = self.resolve_path(path, revision=revision)
        if not getattr(resolved, "repo_id", None):
            return super().url(path)
        return hf_hub_url(
            resolved.repo_id,
            resolved.path_in_repo,
            repo_type=resolved.repo_type,
            revision=resolved.revision,
            endpoint=self.endpoint,
        )

    def info(self, path, *args, **kwargs):
        kwargs.setdefault("expand_info", False)
        with _translate_auth_error():
            return super().info(path, *args, **kwargs)
