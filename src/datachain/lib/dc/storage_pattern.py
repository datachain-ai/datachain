import glob
import os
from collections.abc import Sequence

from datachain.client.fsspec import is_cloud_uri
from datachain.lib.listing import ListScope, PathFilter


def validate_cloud_bucket_name(uri: str) -> None:
    """Validate that cloud storage bucket names don't contain glob patterns.

    Raises:
        ValueError: If a cloud storage bucket name contains glob patterns
    """
    if not is_cloud_uri(uri):
        return

    if "://" in uri:
        scheme_end = uri.index("://") + 3
        path_part = uri[scheme_end:]

        if "/" in path_part:
            bucket_name = path_part.split("/")[0]
        else:
            bucket_name = path_part

        glob_chars = ["*", "?", "[", "]", "{", "}"]
        if any(char in bucket_name for char in glob_chars):
            raise ValueError(f"Glob patterns in bucket names are not supported: {uri}")


def normalize_storage_uri(uri: str | os.PathLike[str]) -> str:
    """Resolve local paths before splitting patterns; preserve cloud object keys."""
    from datachain.client import Client
    from datachain.client.local import FileClient

    uri = os.fspath(uri)
    if issubclass(Client.get_implementation(uri), FileClient):
        FileClient.validate_source(uri)
        storage, path = Client.parse_url(uri)
        return f"{storage}/{path}"
    return uri


def split_uri_pattern(uri: str) -> tuple[str, str | None]:
    """Split a URI into base and pattern; a glob base ends with a directory slash."""
    if not glob.has_magic(uri):
        return uri, None

    if "://" in uri:
        scheme_end = uri.index("://") + 3
        scheme_part = uri[:scheme_end]
        path_part = uri[scheme_end:]
        path_segments = path_part.split("/")

        pattern_start_idx = None
        for i, segment in enumerate(path_segments):
            if glob.has_magic(segment):
                pattern_start_idx = i
                break

        if pattern_start_idx is None:
            return uri, None

        if pattern_start_idx == 0:
            base = scheme_part + path_segments[0]
            pattern = "/".join(path_segments[1:]) if len(path_segments) > 1 else "*"
        else:
            base = scheme_part + "/".join(path_segments[:pattern_start_idx])
            pattern = "/".join(path_segments[pattern_start_idx:])

        return base + "/", pattern

    path_segments = uri.split("/")

    pattern_start_idx = None
    for i, segment in enumerate(path_segments):
        if glob.has_magic(segment):
            pattern_start_idx = i
            break

    if pattern_start_idx is None:
        return uri, None

    base = (
        "/".join(path_segments[:pattern_start_idx]) + "/"
        if pattern_start_idx > 0
        else "/"
    )
    pattern = "/".join(path_segments[pattern_start_idx:])

    return base, pattern


def should_use_recursion(pattern: str, user_recursive: bool) -> bool:
    """Return whether listing should recurse for the given glob pattern."""
    if not user_recursive:
        return False

    if "**" in pattern:
        return True

    return "/" in pattern


def expand_brace_pattern(pattern: str) -> list[str]:
    """Recursively expand brace patterns into multiple glob patterns.

    Supports:
    - Comma-separated lists: *.{mp3,wav}
    - Numeric ranges: file{1..10}
    - Zero-padded numeric ranges: file{01..10}
    - Character ranges: file{a..z}

    Examples:
        "*.{mp3,wav}" -> ["*.mp3", "*.wav"]
        "file{1..3}" -> ["file1", "file2", "file3"]
        "file{01..03}" -> ["file01", "file02", "file03"]
        "file{a..c}" -> ["filea", "fileb", "filec"]
        "{a,b}/{c,d}" -> ["a/c", "a/d", "b/c", "b/d"]
    """
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    return _expand_single_braces(pattern)


def _expand_single_braces(pattern: str) -> list[str]:
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    start = pattern.index("{")
    end = start
    depth = 0
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if start >= end:
        return [pattern]

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    brace_content = pattern[start + 1 : end]

    if ".." in brace_content:
        options = _expand_range(brace_content)
    else:
        options = [opt.strip() for opt in brace_content.split(",")]

    expanded = []
    for option in options:
        combined = prefix + option + suffix
        expanded.extend(_expand_single_braces(combined))

    return expanded


def _expand_range(range_spec: str) -> list[str]:  # noqa: PLR0911
    if ".." not in range_spec:
        return [range_spec]

    parts = range_spec.split("..")
    if len(parts) != 2:
        return [range_spec]

    start, end = parts[0], parts[1]

    if start.isdigit() and end.isdigit():
        pad_width = max(len(start), len(end)) if start[0] == "0" or end[0] == "0" else 0
        start_num = int(start)
        end_num = int(end)

        if start_num <= end_num:
            if pad_width > 0:
                return [str(i).zfill(pad_width) for i in range(start_num, end_num + 1)]
            return [str(i) for i in range(start_num, end_num + 1)]
        if pad_width > 0:
            return [str(i).zfill(pad_width) for i in range(start_num, end_num - 1, -1)]
        return [str(i) for i in range(start_num, end_num - 1, -1)]

    if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
        start_ord = ord(start)
        end_ord = ord(end)

        if start_ord <= end_ord:
            return [chr(i) for i in range(start_ord, end_ord + 1)]
        return [chr(i) for i in range(start_ord, end_ord - 1, -1)]

    return [range_spec]


def convert_globstar_to_glob(filter_pattern: str) -> str:
    """Convert a ``**`` globstar pattern into an equivalent GLOB pattern."""
    if "**" not in filter_pattern:
        return filter_pattern

    parts = filter_pattern.split("/")
    globstar_positions = [i for i, p in enumerate(parts) if p == "**"]

    num_globstars = len(globstar_positions)

    if num_globstars <= 1:
        if filter_pattern == "**/*":
            return "*"
        if filter_pattern.startswith("**/"):
            remaining = filter_pattern[3:]
            if "/" not in remaining:
                # Pattern like **/*.ext or **/temp?.*
                # The ** means zero or more directories
                # For zero directories: pattern should be just the filename pattern
                # For one or more: pattern should be */filename
                # Since we can't OR in GLOB, we choose the more permissive option
                # that works with recursive listing
                # Special handling: if it's a simple extension pattern, match broadly
                if remaining.startswith("*."):
                    return remaining
                return f"*/{remaining}"

        return filter_pattern.replace("**", "*")

    middle_parts = []
    start_idx = globstar_positions[0] + 1
    end_idx = globstar_positions[-1]
    for i in range(start_idx, end_idx):
        if parts[i] != "**":
            middle_parts.append(parts[i])

    if not middle_parts:
        result = filter_pattern.replace("**", "*")
    else:
        middle_pattern = "/".join(middle_parts)
        last_part = parts[-1] if parts[-1] != "**" else "*"

        if last_part != "*":
            result = f"*{middle_pattern}*{last_part}"
        else:
            result = f"*{middle_pattern}*"

    return result


def storage_sources(uris: Sequence) -> list[tuple[str, str | None]]:
    """The URIs to list for ``uris``, in order and with repeats, each with the
    glob pattern that selects from its listing: braces expanded, a glob listed
    from its base.
    """
    sources = []
    for uri in uris:
        for expanded in expand_brace_pattern(str(uri)):
            expanded = normalize_storage_uri(expanded)
            base_uri, pattern = split_uri_pattern(expanded)
            sources.append((base_uri if pattern else expanded, pattern))
    return sources


def path_filter(
    list_path: str, pattern: str | None, recursive: bool | None
) -> PathFilter:
    """The paths a URI with glob ``pattern`` selects from its base's listing."""
    if pattern is None:
        return PathFilter.ls(list_path, recursive)
    glob, use_recursive = _pattern(pattern, list_path, recursive)
    return PathFilter.ls(list_path, use_recursive).glob(glob)


def _pattern(pattern: str, list_path: str, recursive: bool | None) -> tuple[str, bool]:
    """``pattern`` as a glob of full paths, and whether it recurses."""
    return (
        _glob_pattern(pattern, list_path),
        should_use_recursion(pattern, recursive or False),
    )


def list_scope(
    list_path: str, pattern: str | None, recursive: bool | None
) -> ListScope | None:
    """The part of its folder's listing that a URI's paths can be in: the keys
    that start with the pattern's literal head, one level only if the URI
    doesn't recurse. None when it is the whole listing.
    """
    folder = list_path.lstrip("/").rstrip("/*")
    folder = f"{folder}/" if folder else ""
    if pattern is None:
        return None if recursive else ListScope(folder, folder, True)
    head, use_recursive = _pattern(pattern, list_path, recursive)
    magic = [i for c in "*?[" if (i := head.find(c)) >= 0]
    head = head[: min(magic, default=len(head))]
    prefix = head if head.startswith(folder) else folder
    one_level = not use_recursive
    if not one_level and prefix == folder:
        return None
    return ListScope(folder, prefix, one_level)


def _glob_pattern(pattern: str, list_path: str) -> str:
    glob_pattern = convert_globstar_to_glob(pattern)
    if subpath := list_path.strip("/"):
        glob_pattern = f"{subpath}/{glob_pattern}"
    return glob_pattern
