"""Shared pure helpers for the dc-graph skill scripts."""

import json
import re
import sys


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


def read_file_versions(path):
    """Find all '### X.Y.Z' headings in a markdown file, in order."""
    try:
        with open(path) as f:
            content = f.read()
        return re.findall(r"^### (\d+\.\d+\.\d+)", content, re.MULTILINE)
    except Exception:
        return []


def dataset_file_path(name, source):
    """Derive the relative file path (from .datachain/graph/) for a dataset."""
    dot_parts = name.split(".", 2)
    if source == "studio" and len(dot_parts) == 3:
        namespace, project, bare_name = dot_parts
        bare_name_slug = bare_name.lower().replace(".", "_")
        return f"datasets/{namespace}/{project}/{bare_name_slug}.md"
    name_slug = name.lower().replace(".", "_")
    return f"datasets/{name_slug}.md"


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
