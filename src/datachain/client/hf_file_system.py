"""Hugging Face filesystem support, imported only for HF clients."""

from huggingface_hub import HfFileSystem


class PinnedHfFileSystem(HfFileSystem):
    """Use the listed commit even when a URI names a branch or tag."""

    def resolve_path(self, path, revision=None):
        if revision is not None:
            path = _drop_path_revision(self._strip_protocol(path))
        return super().resolve_path(path, revision=revision)


def _drop_path_revision(path: str) -> str:
    """Remove a repository's URI revision before reading its pinned commit."""
    from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES
    from huggingface_hub.hf_file_system import SPECIAL_REFS_REVISION_REGEX

    prefix = next(
        (
            p
            for p in (*REPO_TYPES_URL_PREFIXES.values(), "models/")
            if path.startswith(p)
        ),
        "",
    )
    repo, separator, rest = path[len(prefix) :].partition("@")
    if not separator or repo.count("/") != 1:
        return path
    match = SPECIAL_REFS_REVISION_REGEX.match(rest)
    name = rest[match.end() :].lstrip("/") if match else rest.partition("/")[2]
    return f"{prefix}{repo}/{name}".rstrip("/")
