"""Shared pure helpers for the dc-graph skill scripts."""

import json
import re
import sys
from urllib.parse import urlparse


def dc_import():
    """Import and return the datachain module, or exit with an error."""
    try:
        import datachain as dc

        return dc
    except ImportError:
        print(json.dumps({"error": "datachain not installed"}), file=sys.stderr)
        sys.exit(1)


def studio_available() -> bool:
    """Return True if a Studio token is configured (env var or config file)."""
    try:
        from datachain.remote.studio import is_token_set

        return is_token_set()
    except Exception:
        return False


def parse_semver(v):
    """Parse version string into a tuple for sorting."""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def read_frontmatter(path):
    """Read YAML frontmatter from a markdown file. Returns dict or {}."""
    try:
        with open(path) as f:
            content = f.read()
        if not content.startswith("---"):
            return {}
        end = content.index("\n---", 3)
        fm_text = content[4:end]  # skip first "---\n"
        result = {}
        for line in fm_text.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                result[key.strip()] = val.strip().strip('"').strip("'")
        return result
    except Exception:
        return {}


def read_json_versions(path):
    """Read version list from a dataset JSON file. Returns list of version strings."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [v["version"] for v in data.get("versions", []) if v.get("version")]
    except Exception:
        return []


def read_json_metadata(path):
    """Read latest_version and num_objects from a dataset JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        versions = data.get("versions", [])
        if not versions:
            return {}
        latest = versions[-1]
        return {
            "latest_version": latest.get("version", ""),
            "num_objects": str(latest.get("num_objects", "")),
        }
    except Exception:
        return {}


def dataset_file_path(name, source):
    """Derive the relative file path (from datachain/graph/) for a dataset, without extension."""
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
            if getattr(info, "namespace", None) in ("system", "listing"):
                continue
            if getattr(info, "project", None) == "listing":
                continue
            if getattr(info, "is_temp", False):
                continue
            namespace = getattr(info, "namespace", None)
            project = getattr(info, "project", None)
            # Fully-qualify Studio dataset names using dot-notation (namespace.project.name).
            # Dots are used in all human-visible content; / is used only for file paths.
            if studio and namespace and project:
                full_name = f"{namespace}.{project}.{info.name}"
            else:
                full_name = info.name
            results.append(
                {
                    "name": full_name,
                    "version": str(info.version) if info.version is not None else None,
                    "num_objects": getattr(info, "num_objects", None),
                    "status": getattr(info, "status", None),
                    "namespace": namespace,
                    "project": project,
                    "source": "studio" if studio else "local",
                    "created_at": (
                        info.created_at.isoformat()
                        if getattr(info, "created_at", None) is not None
                        else None
                    ),
                    "updated_at": (
                        info.updated_at.isoformat()
                        if getattr(info, "updated_at", None) is not None
                        else None
                    ),
                }
            )
    except Exception as e:
        print(
            f"[dc-graph warning] collect_datasets(studio={studio}): {e}",
            file=sys.stderr,
        )
    return results


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def parse_uri(uri: str) -> dict:
    """Parse a storage URI into scheme, bucket, and prefix.

    Examples:
        s3://my-bucket/         -> scheme=s3, bucket=my-bucket, prefix=""
        gs://demo/dogs-cats/    -> scheme=gs, bucket=demo,      prefix="dogs-cats/"
    """
    parsed = urlparse(uri)
    return {
        "scheme": parsed.scheme,
        "bucket": parsed.netloc,
        "prefix": parsed.path.lstrip("/"),
    }


def _sanitize(s: str) -> str:
    """Sanitize a name segment: lowercase, replace non-alphanumeric with _."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def bucket_file_path(uri: str) -> str:
    """Derive the relative file path (from datachain/graph/) for a bucket, without extension.

    Examples:
        s3://my-bucket/           -> buckets/s3/my_bucket
        gs://demo/dogs-cats/      -> buckets/gs/demo__dogs_cats
    """
    parts = parse_uri(uri)
    slug = _sanitize(parts["bucket"])
    prefix = parts["prefix"]
    if prefix:
        segments = [s for s in prefix.split("/") if s]
        if segments:
            slug += "__" + "__".join(_sanitize(s) for s in segments)
    return f"buckets/{parts['scheme']}/{slug}"


def read_json_data(path: str) -> dict | None:
    """Read a JSON data file. Returns dict or None."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
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


def get_listing_finished_at(uri: str) -> str | None:
    """Get the listing finished_at timestamp for a URI from the DataChain catalog."""
    try:
        from datachain.query import Session

        session = Session.get()
        catalog = session.catalog
        listings = catalog.listings()

        for listing in listings:
            if listing.uri.rstrip("/") == uri.rstrip("/") or uri.rstrip("/").startswith(
                listing.uri.rstrip("/")
            ):
                if listing.finished_at:
                    return listing.finished_at.isoformat()
        return None
    except Exception:
        return None
