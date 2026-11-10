"""Hugging Face filesystem support, imported only for HF clients."""

from dataclasses import replace

from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import SPECIAL_REFS_REVISION_REGEX


class PinnedHfFileSystem(HfFileSystem):
    """Use the listed commit even when a URI names a branch or tag."""

    def resolve_path(self, path, revision=None):
        if revision is None:
            return super().resolve_path(path)

        path = self._strip_protocol(path).removeprefix("models/")
        path, separator, rest = path.partition("@")
        resolved = super().resolve_path(path, revision=revision)
        if not separator:
            return resolved
        if resolved.path_in_repo or path.endswith("/"):
            path_in_repo = f"{resolved.path_in_repo}@{rest}"
        else:
            match = SPECIAL_REFS_REVISION_REGEX.match(rest)
            path_in_repo = (
                rest[match.end() :].lstrip("/") if match else rest.partition("/")[2]
            )
        return replace(resolved, path_in_repo=path_in_repo, _raw_revision=None)

    def get_file(self, rpath, *args, **kwargs):
        revision = kwargs.get("revision")
        if revision is not None:
            rpath = self.resolve_path(rpath, revision=revision).unresolve()
        return super().get_file(rpath, *args, **kwargs)

    def info(self, path, *args, **kwargs):
        kwargs.setdefault("expand_info", False)
        return super().info(path, *args, **kwargs)
