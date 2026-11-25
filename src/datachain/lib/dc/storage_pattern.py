import glob
import os
import re
from collections.abc import Sequence
from itertools import pairwise

from datachain.client.fsspec import is_cloud_uri
from datachain.lib.listing import ListScope, PathFilter, list_folder

_RANGE = re.compile(r"(-?[0-9]+|[A-Za-z])\.\.(-?[0-9]+|[A-Za-z])(?:\.\.(-?[0-9]+))?")


def validate_cloud_bucket_name(uri: str) -> None:
    """Validate that cloud storage bucket names don't contain glob patterns.

    Raises:
        ValueError: If a cloud storage bucket name contains glob patterns
    """
    if not is_cloud_uri(uri):
        return
    bucket = uri.partition("://")[2].split("/")[0]
    if any(char in bucket for char in "*?[]{}"):
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
    """Split a URI into base and pattern at its first segment with a wildcard;
    a glob base ends with a directory slash.
    """
    scheme, sep, path = uri.partition("://")
    if not sep:
        scheme, path = "", uri
    segments = path.split("/")
    start = next((i for i, s in enumerate(segments) if glob.has_magic(s)), None)
    if start is None:
        return uri, None
    if sep and start == 0:  # a wildcard in the bucket name
        return f"{scheme}://{segments[0]}/", "/".join(segments[1:]) or "*"
    base = "/".join(segments[:start])
    return f"{scheme}{sep}{base}/", "/".join(segments[start:])


def expand_brace_pattern(pattern: str) -> list[str]:
    """``pattern`` with its braces expanded, as bash does: ``{a,b}`` lists the
    alternatives, ``{1..9}`` and ``{a..z}`` a range, ``{1..9..2}`` one with a
    step, and ``{01..10}`` one with zero-padded numbers. A brace with neither,
    such as ``{a}``, stays as it is. Unlike bash, spaces around an alternative
    are dropped: ``{jpg, png}``.

    Examples:
        "*.{mp3,wav}" -> ["*.mp3", "*.wav"]
        "file{1..3}" -> ["file1", "file2", "file3"]
        "file{01..03}" -> ["file01", "file02", "file03"]
        "file{a..e..2}" -> ["filea", "filec", "filee"]
        "{a,b}/{c,d}" -> ["a/c", "a/d", "b/c", "b/d"]
    """
    start = 0
    while (open_ := pattern.find("{", start)) >= 0:
        start = open_ + 1
        depth, cuts = 0, [open_]
        for close in range(open_, len(pattern)):
            if pattern[close] == "{":
                depth += 1
            elif pattern[close] == "}":
                depth -= 1
                if not depth:
                    break
            elif pattern[close] == "," and depth == 1:
                cuts.append(close)
        else:
            continue  # never closed: the "{" is a character
        cuts.append(close)
        options = (
            [pattern[a + 1 : b].strip() for a, b in pairwise(cuts)]
            if len(cuts) > 2
            else _brace_range(pattern[open_ + 1 : close])
        )
        if options is None:
            continue  # "{a}" stays, but braces inside it can expand
        tails = expand_brace_pattern(pattern[close + 1 :])
        return [
            f"{pattern[:open_]}{option}{tail}"
            for alternative in options
            for option in expand_brace_pattern(alternative)
            for tail in tails
        ]
    return [pattern]


def _brace_range(body: str) -> list[str] | None:
    """The terms of a ``{x..y}`` or ``{x..y..step}`` range, or None."""
    if not (match := _RANGE.fullmatch(body)):
        return None
    first, last, step = match.groups()
    if first.isalpha() != last.isalpha():
        return None
    step = abs(int(step or 1)) or 1
    if first.isalpha():
        start, stop = ord(first), ord(last)
        sign = 1 if stop >= start else -1
        return [chr(i) for i in range(start, stop + sign, sign * step)]
    start, stop = int(first), int(last)
    sign = 1 if stop >= start else -1
    # an end such as "01" or "-01" pads every term to the longer end, as bash
    padded = any(re.fullmatch(r"-?0\d+", end) for end in (first, last))
    width = max(len(first), len(last)) if padded else 0
    return [f"{n:0{width}d}" for n in range(start, stop + sign, sign * step)]


def storage_sources(uris: Sequence) -> list[tuple[str, tuple[str, ...]]]:
    """The URIs to list for ``uris``, in order and with repeats, each with the
    glob patterns that select from its listing, none for all of it. Braces
    expand; the alternatives of one URI with one base share its source, where
    the folder itself is the pattern ``**``.
    """
    sources = []
    for uri in uris:
        bases: dict[str, dict[str | None, None]] = {}
        for expanded in expand_brace_pattern(str(uri)):
            base, pattern = split_uri_pattern(normalize_storage_uri(expanded))
            bases.setdefault(base, {})[pattern] = None
        sources += [
            (base, tuple(p or "**" for p in patterns) if any(patterns) else ())
            for base, patterns in bases.items()
        ]
    return sources


def path_filter(
    list_path: str, patterns: Sequence[str], recursive: bool | None
) -> PathFilter:
    """The paths of its base's listing that a URI selects: those that match
    one of its glob ``patterns``, or all of the base when it has none. GLOB's
    ``*`` crosses ``/``, so a GLOB alone is exact in one folder or for a name
    such as ``**/*.jpg``; other patterns are also matched folder by folder.
    """
    if not patterns:
        return PathFilter.ls(list_path, recursive)
    folder = _folder(list_path)
    alternatives = [_segments(p, recursive) for p in patterns]
    descends = any(_descends(s) for s in alternatives)
    paths = PathFilter.ls(list_path, descends)
    conditions = []
    for segments in alternatives:
        alternative = [("glob", folder + _glob(segments))]
        if not _glob_is_exact(segments, descends):
            alternative.append(("pattern", folder + "/".join(segments)))
        conditions.append(tuple(alternative))
    return PathFilter(paths.any_of, tuple(conditions))


def list_scope(
    list_path: str, patterns: Sequence[str], recursive: bool | None
) -> ListScope | None:
    """The part of its folder's listing that a URI's paths can be in: the keys
    that start with its patterns' literal head, one level only if none of them
    descends. None when it is the whole listing.
    """
    folder = _folder(list_path)
    if not patterns:
        return None if recursive else ListScope(folder, folder, True)
    heads, descends = [], False
    for pattern in patterns:
        segments = _segments(pattern, recursive)
        heads.append(re.split(r"[*?[]", _glob(segments), maxsplit=1)[0])
        descends = descends or _descends(segments)
    prefix = folder + os.path.commonprefix(heads)  # noqa: RUF071 a key prefix
    if descends and prefix == folder:
        return None
    return ListScope(folder, prefix, not descends)


def _folder(list_path: str) -> str:
    """``list_path`` as the start of the paths in it: ``dir/``, or nothing."""
    folder = list_folder(list_path)
    return f"{folder}/" if folder else ""


def _segments(pattern: str, recursive: bool | None) -> list[str]:
    """``pattern``'s folders and name, with one ``**`` for a run of them.
    Without ``recursive``, a ``**`` is no folders, or the files of its folder
    when it is the name.
    """
    segments: list[str] = []
    for segment in pattern.split("/"):
        if segment != "**" or segments[-1:] != ["**"]:
            segments.append(segment)
    if recursive:
        return segments
    *folders, name = segments
    return [s for s in folders if s != "**"] + ["*" if name == "**" else name]


def _descends(segments: list[str]) -> bool:
    """Whether the paths of ``segments`` can be below the listing's folder."""
    return len(segments) > 1 or segments == ["**"]


def _glob(segments: list[str]) -> str:
    """A GLOB that every path of ``segments`` matches: GLOB's ``*`` crosses
    ``/``, so ``**/`` is just ``*``.
    """
    *folders, name = segments
    glob_pattern = "".join("*" if s == "**" else f"{s}/" for s in folders)
    if name == "**" or (glob_pattern[-1:] == "*" and name[:1] == "*"):
        name = name[1:]  # one star for "**" or for "**/*.jpg"
    return glob_pattern + name


def _glob_is_exact(segments: list[str], descends: bool) -> bool:
    """Whether the GLOB of ``segments`` matches only their paths: a name in the
    listing's folder when the filter doesn't descend, all files, or a name
    that ends a path, as ``**/*.jpg``.
    """
    if "**" not in segments:
        return len(segments) == 1 and not descends
    *folders, name = segments
    if folders not in ([], ["**"]):
        return False
    return name == "**" or (name[:1] == "*" and not glob.has_magic(name[1:]))
