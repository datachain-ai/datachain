"""Hugging Face filesystem support, imported only for HF clients."""

from dataclasses import replace
from urllib.parse import quote

import huggingface_hub
from huggingface_hub import HfFileSystem
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_file_system import SPECIAL_REFS_REVISION_REGEX


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

    def info(self, path, *args, **kwargs):
        kwargs.setdefault("expand_info", False)
        return super().info(path, *args, **kwargs)
