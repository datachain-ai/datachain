import json
import os
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from datachain.skill.knowledge.types import DatasetVersionEntry, DependencyEntry


def write_text(path: str, content: str) -> None:
    """Write text to a file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def write_json(path: str, data, **kwargs) -> None:
    """Write JSON to a file, creating parent directories as needed.

    Defaults to `indent=2, default=str` for human-readable output that
    handles non-JSON-native types (datetimes, etc.). Always appends a
    trailing newline.
    """
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("default", str)
    write_text(path, json.dumps(data, **kwargs) + "\n")


def dc_import():
    """Import and return the datachain module, or exit with an error."""
    try:
        import datachain as dc

        return dc
    except ImportError:
        print(json.dumps({"error": "datachain not installed"}), file=sys.stderr)
        sys.exit(1)


def studio_available() -> bool:
    """Return True if a Studio token is configured."""
    try:
        from datachain.remote.studio import is_token_set

        return is_token_set()
    except Exception:  # noqa: BLE001
        return False


def parse_semver(v: object) -> tuple[int, int, int]:
    """Parse a dotted version into a (major, minor, patch) tuple for sorting.

    Short versions pad with zeros, so "1", "1.0" and "1.0.0" all sort equal.
    Anything past patch is dropped. Unparsable input returns (-1, -1, -1) so it
    sorts below every real version, including "0.0.0".
    """
    try:
        parts = [int(x) for x in str(v).split(".")]
    except (ValueError, AttributeError):
        return -1, -1, -1
    parts = [*parts, 0, 0, 0][:3]
    return parts[0], parts[1], parts[2]


def split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Split YAML-lite frontmatter from markdown text into (frontmatter, body).

    Tolerates a single outer ```/```markdown ... ``` fence — the enrichment
    prompt's Output Format section shows the document inside such a fence, and
    some models echo it. Only a bare wrapper fence is stripped: a document
    that legitimately begins with a language-tagged code block (e.g. ```python)
    is left intact. Returns ({}, body) when there's no frontmatter block.
    """
    text = (content or "").strip()
    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower() in ("```", "```markdown", "```md"):
        lines = rest.rstrip().splitlines()
        # Strip the wrapper's closing fence, but only when it's an unmatched
        # standalone fence line (a line that is only ```).
        fence_lines = sum(1 for ln in lines if ln.lstrip().startswith("```"))
        if lines and lines[-1].strip() == "```" and fence_lines % 2 == 1:
            lines = lines[:-1]
        text = "\n".join(lines).rstrip()
    if not text.startswith("---"):
        return {}, text
    try:
        end = text.index("\n---", 3)
    except ValueError:
        return {}, text
    result: dict[str, str] = {}
    for line in text[4:end].splitlines():  # skip first "---\n"
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result, text[end + 4 :].strip()


def read_frontmatter(path: str) -> dict[str, str]:
    """Read YAML frontmatter from a markdown file. Returns dict or {}."""
    try:
        with open(path) as f:
            content = f.read()
    except Exception:  # noqa: BLE001
        return {}
    return split_frontmatter(content)[0]


_ORDERED_LIST_RE = re.compile(
    r"\d+[.)]\s"
)  # ordered-list markers: "1. ", "2) ", "10. ", etc.


def _first_prose_paragraph(lines: list[str]) -> str:
    """First contiguous block of prose lines in the body.

    Skips headings, tables, code fences, and list items (both unordered
    `-`/`*`/`+` and ordered `1.`/`1)`) — anything that isn't plain prose.
    """
    paragraph: list[str] = []
    in_fence = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            if paragraph:
                break
            continue
        if in_fence:
            continue
        if not line:
            if paragraph:
                break
            continue
        is_list = line.startswith(("- ", "* ", "+ ")) or _ORDERED_LIST_RE.match(line)
        is_hr = line == "---"
        if line.startswith(("#", "|")) or is_hr or is_list:
            if paragraph:
                break
            continue
        paragraph.append(line)
    return " ".join(paragraph)


def extract_description(lines: list[str]) -> str:
    """Short description of an enriched document.

    The intro paragraph after the `# ` H1 heading (up to the first `##`); if the
    document has no H1 or no prose under it, the first prose paragraph anywhere
    (skipping headings, tables, list items, and code fences). `lines` is the
    markdown body split on newlines (frontmatter already removed, e.g. via
    `split_frontmatter`). Returns "" when there's no prose at all.
    """
    desc_lines: list[str] = []
    past_heading = False
    for line in lines:
        if not past_heading:
            if line.startswith("# "):
                past_heading = True
            continue
        if line.startswith("##"):
            break
        stripped = line.strip()
        if not stripped and desc_lines:
            break
        if stripped:
            desc_lines.append(stripped)
    return " ".join(desc_lines) or _first_prose_paragraph(lines)


def is_sys_column(col: str) -> bool:
    """True for DataChain system columns — the whole `sys` signal namespace."""
    return col == "sys" or col.startswith(("sys.", "sys__"))


def escape_table_cell(value: object) -> str:
    """Escape a value for a markdown table cell.

    `|` breaks columns and newlines break rows.
    """
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def read_json_versions(path):
    """Read version list from a dataset JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [v["version"] for v in data.get("versions", []) if v.get("version")]
    except Exception:  # noqa: BLE001
        return []


def read_json_metadata(path):
    """Read last_version and records from a dataset JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        versions = data.get("versions", [])
        if not versions:
            return {}
        latest = versions[-1]
        return {
            "last_version": latest.get("version", ""),
            "records": str(latest.get("records", "")),
        }
    except Exception:  # noqa: BLE001
        return {}


def read_md_versions(path):
    """Read known_versions from .md frontmatter. Returns list of version strings."""
    fm = read_frontmatter(path)
    raw = fm.get("known_versions", "")
    if raw.startswith("[") and raw.endswith("]"):
        return [v.strip() for v in raw[1:-1].split(",") if v.strip()]
    return []


def read_md_metadata(path):
    """Read last_version and records from .md frontmatter."""
    fm = read_frontmatter(path)
    return {
        "last_version": fm.get("last_version", ""),
        "records": fm.get("records", ""),
    }


def read_md_scanned(path):
    """Read scanned from bucket .md frontmatter."""
    fm = read_frontmatter(path)
    return fm.get("scanned")


def dataset_file_path(name, source):
    """Derive the relative file path (from dc-knowledge/) for a dataset.

    Returns the path without extension.
    """
    dot_parts = name.split(".", 2)
    if source == "studio" and len(dot_parts) == 3:
        namespace, project, bare_name = dot_parts
        bare_name_slug = bare_name.lower().replace(".", "_")
        return f"datasets/{namespace}/{project}/{bare_name_slug}"
    name_slug = name.lower().replace(".", "_")
    return f"datasets/{name_slug}"


def serialize(val):
    """Serialize a value to a JSON-safe type."""
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    return str(val)


def collect_datasets(dc, studio: bool) -> list[dict]:
    """Return a list of dataset dicts from local or Studio source."""
    results = []
    try:
        for row in dc.datasets(column="dataset", studio=studio).to_iter():
            info = row[0]
            if getattr(info, "namespace", None) in (
                "system",
                "listing",
            ):
                continue
            if getattr(info, "project", None) == "listing":
                continue
            if getattr(info, "is_temp", False):
                continue
            namespace = getattr(info, "namespace", None)
            project = getattr(info, "project", None)
            # Fully-qualify Studio dataset names as namespace.project.name.
            if studio and namespace and project:
                full_name = f"{namespace}.{project}.{info.name}"
            else:
                full_name = info.name
            results.append(
                {
                    "name": full_name,
                    "version": (
                        str(info.version) if info.version is not None else None
                    ),
                    "records": getattr(info, "num_objects", None),
                    "status": getattr(info, "status", None),
                    "namespace": namespace,
                    "project": project,
                    "source": "studio" if studio else "local",
                    "created": (
                        info.created_at.isoformat()
                        if getattr(info, "created_at", None) is not None
                        else None
                    ),
                    "finished": (
                        info.finished_at.isoformat()
                        if getattr(info, "finished_at", None) is not None
                        else None
                    ),
                    "updated": (
                        info.updated_at.isoformat()
                        if getattr(info, "updated_at", None) is not None
                        else None
                    ),
                }
            )
    except Exception as e:  # noqa: BLE001
        print(
            f"[dc-knowledge warning] collect_datasets(studio={studio}): {e}",
            file=sys.stderr,
        )
    return results


def parse_uri(uri: str) -> dict:
    """Parse a storage URI into scheme, bucket, and prefix.

    Examples:
        s3://my-bucket/       -> scheme=s3, bucket=my-bucket, prefix=""
        gs://demo/dogs-cats/  -> scheme=gs, bucket=demo, prefix="dogs-cats/"
    """
    parsed = urlparse(uri)
    return {
        "scheme": parsed.scheme,
        "bucket": parsed.netloc,
        "prefix": parsed.path.lstrip("/"),
    }


def _sanitize(s: str) -> str:
    """Sanitize a name segment: lowercase, replace non-alnum with _."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def bucket_file_path(uri: str) -> str:
    """Derive the relative file path for a bucket, without extension.

    Whole-bucket listings produce a flat file; partial listings (with a prefix)
    go into a subdirectory named after the bucket.

    Examples:
        s3://my-bucket/                  -> buckets/s3/my_bucket
        gs://demo/dogs-cats/             -> buckets/gs/demo/dogs_cats
        gs://demo/dogs-cats/annotations/ -> buckets/gs/demo/dogs_cats__annotations
    """
    parts = parse_uri(uri)
    bucket_slug = _sanitize(parts["bucket"])
    prefix = parts["prefix"]
    if prefix:
        segments = [s for s in prefix.split("/") if s]
        if segments:
            dir_slug = "__".join(_sanitize(s) for s in segments)
            return f"buckets/{parts['scheme']}/{bucket_slug}/{dir_slug}"
    return f"buckets/{parts['scheme']}/{bucket_slug}"


def clean_dep_name(name: str) -> str:
    """Convert listing dataset names (lst__...) to clean URIs."""
    try:
        from datachain.lib.listing import is_listing_dataset, listing_uri_from_name

        if is_listing_dataset(name):
            return listing_uri_from_name(name)
    except Exception:  # noqa: BLE001, S110
        pass
    return name


def dep_entry(
    raw_name: str | None,
    version: str | None,
    type: str | None,
) -> "DependencyEntry":
    """Assemble one dependency entry from primitives, shared by the cluster scan and
    the Studio-side collector.

    `raw_name` is the source-side name (a `lst__...` listing name or a dataset name),
    cleaned to a URI for listings. `file_path` points at the dependency's own
    knowledge doc so links resolve: a storage URI maps to its bucket doc, a dataset
    name to its dataset doc."""
    name = clean_dep_name(raw_name) if raw_name is not None else None
    entry: DependencyEntry = {
        "name": name,
        "version": str(version) if version is not None else None,
        "type": str(type) if type is not None else None,
    }
    if name is not None:
        if "://" in name:
            # Storage URIs map to a bucket doc; `bucket_file_path` tolerates a
            # missing trailing slash.
            entry["file_path"] = bucket_file_path(name)
        else:
            # A 3-part dotted name means a Studio dataset; anything else is a flat
            # local dataset. That's all `source` controls in `dataset_file_path`.
            source = "studio" if len(name.split(".", 2)) == 3 else "local"
            entry["file_path"] = dataset_file_path(name, source)
    return entry


def drop_unchanged_scripts(versions: "list[DatasetVersionEntry]") -> None:
    """Null `query_script` on older versions whose script didn't change — the prompt
    renders no code block for them, so an identical script would be dead weight. The
    latest version (last entry) and the initial version (`changes is None`) keep their
    script. `versions` must be oldest-first; entries are mutated in place. Run before
    `dedupe_previous_scripts`, which keys off the previous entry's `query_script`."""
    last = len(versions) - 1
    for i, version in enumerate(versions):
        changes = version["changes"]
        if i != last and changes is not None and not changes["script_changed"]:
            version["query_script"] = None


def dedupe_previous_scripts(versions: "list[DatasetVersionEntry]") -> None:
    """Null each version's `changes.previous_script` when the previous entry already
    carries that script as its own `query_script` — the prompt would otherwise render
    the same script twice. `versions` must be oldest-first; entries are mutated in
    place. A `previous_script` is kept only when the previous entry omitted its script
    (an unchanged-older version), so the diff isn't lost."""
    for i in range(1, len(versions)):
        changes = versions[i]["changes"]
        if changes and versions[i - 1]["query_script"] is not None:
            changes["previous_script"] = None


def read_json_data(path: str) -> dict | None:
    """Read a JSON data file. Returns dict or None."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def human_size(nbytes: float) -> str:
    """Convert bytes to human-readable string."""
    if nbytes < 1024:
        return f"{int(nbytes)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        nbytes /= 1024
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes:.1f} PB"


def source_to_https(source: str, account_name: str | None = None) -> str | None:
    """Convert a storage URI to an HTTPS URL prefix for the bucket root.

    File paths in listings are relative to the bucket root, so the prefix
    must point to the bucket root — not the subdirectory being listed.

    Returns None for local paths or unrecognized schemes. For `az://`, the
    netloc is the *container* and the storage account is external config that
    isn't part of the URI — so a link can only be built from `account_name`
    (falling back to `AZURE_STORAGE_ACCOUNT_NAME`); without it, az returns None.

    Examples:
        s3://my-bucket/prefix/  -> https://my-bucket.s3.amazonaws.com
        gs://demo/data/         -> https://storage.googleapis.com/demo
        az://container/data/    -> https://<account>.blob.core.windows.net/container
                                   (only when the account name is available)
    """
    parts = parse_uri(source)
    scheme = parts["scheme"]
    bucket = parts["bucket"]
    if not bucket:
        return None

    if scheme == "s3":
        return f"https://{bucket}.s3.amazonaws.com"
    if scheme == "gs":
        return f"https://storage.googleapis.com/{bucket}"
    if scheme == "az":
        account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        if account_name:
            return f"https://{account_name}.blob.core.windows.net/{bucket}"
    return None
