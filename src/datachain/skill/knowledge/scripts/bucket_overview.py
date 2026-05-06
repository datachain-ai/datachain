"""Sample-of-files overview — bounded subset of `read_storage`.

Saves a DataChain dataset with File rows, identical schema to
`read_storage()`, just a bounded sample drawn via hierarchical
sampling. Useful when full enumeration is impractical.

Stand-in for `dc.bucket_overview()` — see datachain-ai/datachain#1750.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from urllib.parse import urlparse

SAMPLE = 300
REMOTE = {"gs", "s3", "az"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "root"


def _open(uri: str, anon: bool):
    import fsspec

    scheme = urlparse(uri).scheme
    if scheme in REMOTE:
        path = uri.split("://", 1)[1].rstrip("/") + "/"
        if anon:
            kw = {"token": "anon"} if scheme == "gs" else {"anon": True}
        else:
            kw = {}
        return fsspec.filesystem(scheme, **kw), path, scheme
    raw = uri.removeprefix("file://") if scheme == "file" else uri
    return (
        fsspec.filesystem("file"),
        os.path.abspath(os.path.expanduser(raw)).rstrip("/") + "/",
        "file",
    )


def _sample(fs, root: str, limit: int, max_depth: int = 5) -> list[dict]:
    found: list[dict] = []
    queue: list[tuple[str, int]] = [(root.rstrip("/"), 0)]
    while queue and len(found) < limit:
        cur, depth = queue.pop(0)
        try:
            items = fs.ls(cur, detail=True)
        except (OSError, PermissionError):
            continue
        for e in items:
            if len(found) >= limit:
                break
            if e.get("type") == "file":
                found.append(e)
            elif e.get("type") == "directory" and depth < max_depth:
                queue.append((e["name"], depth + 1))
    return found


def _make_file(scheme: str, netloc: str, entry: dict):
    import datachain as dc

    name = entry["name"]
    if scheme in REMOTE:
        path = name.removeprefix(netloc + "/") if netloc else name
        source = f"{scheme}://{netloc}/" if netloc else f"{scheme}://"
    else:
        path = name
        source = "file://"
    return dc.File(source=source, path=path, size=entry.get("size") or 0)


def bucket_overview(
    uri: str, limit: int = SAMPLE, anon: bool = False, name: str | None = None
):
    import datachain as dc

    fs, root, scheme = _open(uri, anon)
    netloc = urlparse(uri).netloc if scheme in REMOTE else ""
    files = [_make_file(scheme, netloc, e) for e in _sample(fs, root, limit)]
    name = name or f"overview_{_slug(uri)}_{int(time.time())}"
    return dc.read_values(file=files).save(name)


def _to_bucket_json(ds, uri: str, anon: bool, dataset_name: str, output: str) -> None:
    """Derive bucket-shape JSON from the sampled dataset.

    Reuses `bucket_scan.compute_bucket_metadata` so a sampled overview
    produces the same JSON shape as a full scan, with `sampled=True` and
    a `dataset_name` link to the underlying File rows.
    """
    from bucket_scan import compute_bucket_metadata
    from utils import write_json

    payload = compute_bucket_metadata(
        ds, uri, anon, sampled=True, dataset_name=dataset_name
    )
    write_json(output, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("uri")
    p.add_argument("--name")
    p.add_argument("--limit", type=int, default=SAMPLE)
    p.add_argument("--anon", action="store_true")
    p.add_argument("--bucket-json", help="Also write bucket-shape JSON for enrichment")
    a = p.parse_args()
    name = a.name or f"overview_{_slug(a.uri)}_{int(time.time())}"
    ds = bucket_overview(a.uri, a.limit, a.anon, name)
    print(f"Saved {ds.count()} files as dataset {name}")
    if a.bucket_json:
        _to_bucket_json(ds, a.uri, a.anon, name, a.bucket_json)
        print(f"Wrote bucket JSON: {a.bucket_json}")
    ds.select("file.path", "file.size").limit(5).show()


if __name__ == "__main__":
    main()
