"""Hugging Face filesystem support, imported only for HF clients."""

from dataclasses import replace
from urllib.parse import quote, urlparse

import huggingface_hub
from huggingface_hub import HfFileSystem, hf_file_system, hf_hub_url
from huggingface_hub.constants import (
    HF_HUB_DOWNLOAD_TIMEOUT,
    REPO_TYPES_URL_PREFIXES,
)
from huggingface_hub.hf_file_system import (
    SPECIAL_REFS_REVISION_REGEX,
    HfFileSystemFile,
)


class PinnedHfFileSystemFile(HfFileSystemFile):
    """Reuse a cross-origin redirect URL across range reads on an open file."""

    _resolved_url: str | None = None

    def _request_range(self, url: str, start: int, end: int, *, auth: bool = True):
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        if not auth:
            headers = {
                key: val
                for key, val in headers.items()
                if key.lower() != "authorization"
            }
        return hf_file_system.http_backoff(
            "GET",
            url,
            headers=headers,
            retry_on_status_codes=(408, 429, 500, 502, 503, 504),
            timeout=HF_HUB_DOWNLOAD_TIMEOUT,
        )

    def _fetch_range(self, start: int, end: int) -> bytes:
        if not getattr(self.resolved_path, "repo_id", None):
            return super()._fetch_range(start, end)
        if resolved_url := self._resolved_url:
            r = self._request_range(resolved_url, start, end, auth=False)
            if r.status_code not in {400, 401, 403, 404, 410}:
                hf_file_system.hf_raise_for_status(r)
                return r.content
            self._resolved_url = None
        url = self.url()
        r = self._request_range(url, start, end)
        hf_file_system.hf_raise_for_status(r)
        final = str(r.url)
        req_headers = getattr(getattr(r, "request", None), "headers", None)
        if (
            req_headers is not None
            and urlparse(final).hostname != urlparse(url).hostname
            and all(key.lower() != "authorization" for key in req_headers)
        ):
            self._resolved_url = final
        return r.content


class PinnedHfFileSystem(HfFileSystem):
    """Use the listed commit even when a URI names a branch or tag."""

    def resolve_path(self, path, revision=None):
        path = self._strip_protocol(path).removeprefix("models/")
        if revision is None:
            return super().resolve_path(path)

        if parse_uri := getattr(huggingface_hub, "parse_hf_uri", None):
            uri = replace(parse_uri(f"hf://{path}"), revision=revision).to_uri()
            return super().resolve_path(uri, revision=revision)

        path, separator, rest = path.partition("@")
        prefix = next(
            (p for p in REPO_TYPES_URL_PREFIXES.values() if p and path.startswith(p)),
            "",
        )
        root = path[len(prefix) :]
        has_uri_revision = separator and "/" in rest and root.count("/") < 2
        resolve_path = (
            f"{path}@{quote(revision, safe='')}/{rest}" if has_uri_revision else path
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

    def get_file(self, rpath, *args, **kwargs):
        revision = kwargs.get("revision")
        if revision is not None:
            rpath = self.resolve_path(rpath, revision=revision).unresolve()
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
        return super().info(path, *args, **kwargs)

    def _open(self, path, mode="rb", block_size=None, revision=None, **kwargs):
        if block_size is None:
            block_size = getattr(self, "block_size", None)
        if mode != "rb" or block_size == 0:
            return super()._open(
                path, mode=mode, block_size=block_size, revision=revision, **kwargs
            )
        return PinnedHfFileSystemFile(
            self, path, mode=mode, block_size=block_size, revision=revision, **kwargs
        )
