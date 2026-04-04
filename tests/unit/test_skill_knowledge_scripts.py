"""Tests for knowledge skill scripts (pure utility functions)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Insert the scripts directory so bare imports work (matches runtime behavior).
SCRIPTS_DIR = str(
    Path(__file__).resolve().parents[2] / "src/datachain/skill/knowledge/scripts"
)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from changes import build_changes, compute_dep_changes  # noqa: E402
from render_index import render_index  # noqa: E402
from schema import parse_dataset_name, type_name  # noqa: E402
from utils import (  # noqa: E402
    bucket_file_path,
    dataset_file_path,
    human_size,
    parse_semver,
    parse_uri,
    read_frontmatter,
    read_json_data,
    read_json_metadata,
    read_json_versions,
    serialize,
    source_to_https,
)

# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


def test_parse_semver_valid():
    assert parse_semver("1.2.3") == (1, 2, 3)


def test_parse_semver_two_part():
    assert parse_semver("1.0") == (1, 0)


def test_parse_semver_invalid_string():
    assert parse_semver("bad") == (0, 0, 0)


def test_parse_semver_none():
    assert parse_semver(None) == (0, 0, 0)


def test_read_frontmatter_normal(tmp_path):
    p = tmp_path / "test.md"
    p.write_text("---\nname: foo\ndescription: bar\n---\n# body\n")
    fm = read_frontmatter(str(p))
    assert fm["name"] == "foo"
    assert fm["description"] == "bar"


def test_read_frontmatter_no_frontmatter(tmp_path):
    p = tmp_path / "test.md"
    p.write_text("# Just a heading\n")
    assert read_frontmatter(str(p)) == {}


def test_read_frontmatter_empty_file(tmp_path):
    p = tmp_path / "test.md"
    p.write_text("")
    assert read_frontmatter(str(p)) == {}


def test_read_frontmatter_missing_file():
    assert read_frontmatter("/nonexistent/path.md") == {}


def test_read_frontmatter_colon_in_value(tmp_path):
    p = tmp_path / "test.md"
    p.write_text("---\ndescription: Use for: things\n---\n")
    fm = read_frontmatter(str(p))
    assert fm["description"] == "Use for: things"


def test_read_frontmatter_quoted_value(tmp_path):
    p = tmp_path / "test.md"
    p.write_text('---\nname: "quoted"\n---\n')
    fm = read_frontmatter(str(p))
    assert fm["name"] == "quoted"


def test_parse_uri_s3():
    result = parse_uri("s3://my-bucket/")
    assert result == {"scheme": "s3", "bucket": "my-bucket", "prefix": ""}


def test_parse_uri_gs_with_prefix():
    result = parse_uri("gs://demo/dogs-cats/")
    assert result == {"scheme": "gs", "bucket": "demo", "prefix": "dogs-cats/"}


def test_parse_uri_az():
    result = parse_uri("az://container/path/to/data")
    assert result["scheme"] == "az"
    assert result["bucket"] == "container"
    assert result["prefix"] == "path/to/data"


def test_bucket_file_path_basic():
    assert bucket_file_path("s3://my-bucket/") == "buckets/s3/my_bucket"


def test_bucket_file_path_with_prefix():
    result = bucket_file_path("gs://demo/dogs-cats/")
    assert result == "buckets/gs/demo__dogs_cats"


def test_bucket_file_path_special_chars():
    result = bucket_file_path("s3://My.Bucket-Name/")
    assert result == "buckets/s3/my_bucket_name"


def test_dataset_file_path_local():
    assert dataset_file_path("my_dataset", "local") == "datasets/my_dataset"


def test_dataset_file_path_studio():
    result = dataset_file_path("ns.proj.my_ds", "studio")
    assert result == "datasets/ns/proj/my_ds"


def test_dataset_file_path_local_with_dots():
    result = dataset_file_path("my.dataset", "local")
    assert result == "datasets/my_dataset"


def test_human_size_bytes():
    assert human_size(500) == "500 B"


def test_human_size_kb():
    assert human_size(1536) == "1.5 KB"


def test_human_size_mb():
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


def test_human_size_gb():
    assert human_size(2.5 * 1024**3) == "2.5 GB"


def test_human_size_zero():
    assert human_size(0) == "0 B"


def test_serialize_primitives():
    assert serialize("hello") == "hello"
    assert serialize(42) == 42
    assert serialize(3.14) == 3.14
    assert serialize(True) is True
    assert serialize(None) is None


def test_serialize_object():
    result = serialize(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert isinstance(result, str)


def test_read_json_versions_valid(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"versions": [{"version": "1.0.0"}, {"version": "2.0.0"}]}))
    assert read_json_versions(str(p)) == ["1.0.0", "2.0.0"]


def test_read_json_versions_empty(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"versions": []}))
    assert read_json_versions(str(p)) == []


def test_read_json_versions_missing_file():
    assert read_json_versions("/nonexistent.json") == []


def test_read_json_metadata_valid(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"versions": [{"version": "1.0.0", "num_objects": 42}]}))
    result = read_json_metadata(str(p))
    assert result["latest_version"] == "1.0.0"
    assert result["num_objects"] == "42"


def test_read_json_metadata_empty(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"versions": []}))
    assert read_json_metadata(str(p)) == {}


def test_read_json_data_valid(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"key": "value"}))
    assert read_json_data(str(p)) == {"key": "value"}


def test_read_json_data_missing():
    assert read_json_data("/nonexistent.json") is None


def test_source_to_https_s3():
    assert source_to_https("s3://my-bucket") == "https://my-bucket.s3.amazonaws.com"


def test_source_to_https_s3_trailing_slash():
    assert source_to_https("s3://my-bucket/") == "https://my-bucket.s3.amazonaws.com"


def test_source_to_https_gs():
    assert source_to_https("gs://demo") == "https://storage.googleapis.com/demo"


def test_source_to_https_az():
    assert (
        source_to_https("az://account/container")
        == "https://account.blob.core.windows.net/container"
    )


def test_source_to_https_local_returns_none():
    assert source_to_https("file:///home/user/data") is None


def test_source_to_https_empty_returns_none():
    assert source_to_https("") is None


# ---------------------------------------------------------------------------
# changes.py
# ---------------------------------------------------------------------------


def test_compute_dep_changes_added():
    result = compute_dep_changes(
        [{"name": "a", "version": "1.0"}],
        [],
    )
    assert result["deps_added"] == [{"name": "a", "version": "1.0"}]
    assert result["deps_removed"] == []
    assert result["deps_updated"] == []


def test_compute_dep_changes_removed():
    result = compute_dep_changes(
        [],
        [{"name": "a", "version": "1.0"}],
    )
    assert result["deps_removed"] == [{"name": "a", "version": "1.0"}]


def test_compute_dep_changes_updated():
    result = compute_dep_changes(
        [{"name": "a", "version": "2.0"}],
        [{"name": "a", "version": "1.0"}],
    )
    assert len(result["deps_updated"]) == 1
    assert result["deps_updated"][0]["version_from"] == "1.0"
    assert result["deps_updated"][0]["version_to"] == "2.0"


def test_compute_dep_changes_unchanged():
    deps = [{"name": "a", "version": "1.0"}]
    result = compute_dep_changes(deps, deps)
    assert result["deps_added"] == []
    assert result["deps_removed"] == []
    assert result["deps_updated"] == []


def test_compute_dep_changes_empty():
    result = compute_dep_changes([], [])
    assert result == {
        "deps_added": [],
        "deps_removed": [],
        "deps_updated": [],
    }


def test_build_changes_script_changed():
    result = build_changes(
        query_script="new code",
        prev_version_str="1.0.0",
        prev_script="old code",
        curr_deps=[],
        prev_deps=[],
    )
    assert result["script_changed"] is True
    assert result["previous_script"] == "old code"
    assert result["previous_version"] == "1.0.0"


def test_build_changes_script_unchanged():
    result = build_changes(
        query_script="same",
        prev_version_str="1.0.0",
        prev_script="same",
        curr_deps=[],
        prev_deps=[],
    )
    assert result["script_changed"] is False
    assert result["previous_script"] is None


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


def test_type_name_primitives():
    assert type_name(str) == "str"
    assert type_name(int) == "int"
    assert type_name(float) == "float"


def test_type_name_none_type():
    assert type_name(type(None)) == "None"


def test_type_name_list():
    assert type_name(list[str]) == "list[str]"


def test_type_name_dict():
    assert type_name(dict[str, int]) == "dict"


def test_type_name_union():
    result = type_name(str | None)
    assert "str" in result
    assert "None" in result


def test_type_name_optional():
    from typing import Optional

    result = type_name(Optional[int])  # noqa: UP045
    assert "int" in result
    assert "None" in result


def test_parse_dataset_name_local():
    assert parse_dataset_name("my_dataset") == (None, None, "my_dataset")


def test_parse_dataset_name_dot_separated():
    assert parse_dataset_name("ns.proj.name") == ("ns", "proj", "name")


def test_parse_dataset_name_slash_separated():
    assert parse_dataset_name("ns/proj/name") == ("ns", "proj", "name")


# ---------------------------------------------------------------------------
# render_index.py
# ---------------------------------------------------------------------------


def test_render_index_local_datasets():
    plan = {
        "db_last_updated": "2024-01-01T00:00:00Z",
        "datasets": [
            {
                "name": "test_ds",
                "source": "local",
                "file_path": "datasets/test_ds",
            },
        ],
        "buckets": [],
    }
    result = render_index(plan)
    assert "db_last_updated: 2024-01-01T00:00:00Z" in result
    assert "## Datasets" in result
    assert "test_ds" in result
    assert "Dependencies" in result
    assert "Summary" in result
    assert "Updated" in result
    # Removed columns
    assert "| Last Ver |" not in result
    assert "| # Vers |" not in result
    assert "| Count |" not in result
    assert "| Source |" not in result
    assert "## Buckets" not in result


def test_render_index_empty_plan():
    result = render_index({"datasets": [], "buckets": []})
    assert "generated_at:" in result
    assert "## Datasets" not in result
    assert "## Buckets" not in result


def test_render_index_frontmatter_counts():
    plan = {
        "datasets": [
            {"name": "a", "source": "local", "file_path": "datasets/a"},
            {"name": "b.ns.proj", "source": "studio", "file_path": "datasets/b"},
        ],
        "buckets": [],
    }
    result = render_index(plan)
    assert "local_dataset_count: 1" in result
    assert "studio_dataset_count: 1" in result


def test_render_index_studio_namespace_subsections():
    plan = {
        "datasets": [
            {
                "name": "ns.proj.ds_a",
                "source": "studio",
                "file_path": "datasets/ns/proj/ds_a",
            },
            {
                "name": "ns.proj.ds_b",
                "source": "studio",
                "file_path": "datasets/ns/proj/ds_b",
            },
            {
                "name": "other.team.ds_c",
                "source": "studio",
                "file_path": "datasets/other/team/ds_c",
            },
        ],
        "buckets": [],
    }
    result = render_index(plan)
    assert "## Studio" in result
    assert "### ns.proj" in result
    assert "### other.team" in result
    # Namespace stripped from name column
    assert "| [ds_a](" in result
    assert "| [ds_b](" in result
    assert "| [ds_c](" in result
    # Full qualified name should NOT appear in table cells
    assert "ns.proj.ds_a" not in result.split("### ns.proj")[1].split("###")[0]
    # Namespaces ordered alphabetically
    ns_proj_pos = result.index("### ns.proj")
    other_team_pos = result.index("### other.team")
    assert ns_proj_pos < other_team_pos


def test_render_index_mixed_local_and_studio():
    plan = {
        "datasets": [
            {
                "name": "local_ds",
                "source": "local",
                "file_path": "datasets/local_ds",
            },
            {
                "name": "ns.proj.studio_ds",
                "source": "studio",
                "file_path": "datasets/ns/proj/studio_ds",
            },
        ],
        "buckets": [],
    }
    result = render_index(plan)
    assert "## Datasets" in result
    assert "## Local" not in result
    assert "## Studio" in result
    assert "### ns.proj" in result


def test_render_index_all_metadata_from_md(tmp_path, monkeypatch):
    import render_index as ri

    monkeypatch.setattr(ri, "BASE_DIR", str(tmp_path))
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    (ds_dir / "my_ds.md").write_text(
        "---\nname: my_ds\nlatest_version: 3.0.0\n"
        "num_objects: 12000\nupdated_at: 2025-04-01T10:00:00Z\n"
        "known_versions: [1.0.0, 2.0.0, 3.0.0]\n---\n\n"
        "# my_ds\n\n"
        "Image metadata with EXIF and GPS for 12k photos.\n\n"
        "## Dependencies\n\n"
        "- [raw_images](datasets/raw_images.md)\n"
        "- [labels](datasets/labels.md)\n\n"
        "## Schema\n"
    )
    plan = {
        "datasets": [
            {
                "name": "my_ds",
                "source": "local",
                "file_path": "datasets/my_ds",
            },
        ],
        "buckets": [],
    }
    result = ri.render_index(plan)
    # Summary from description
    assert "Image metadata with EXIF and GPS for 12k photos." in result
    # Updated from frontmatter
    assert "2025-04-01" in result
    # Dependencies are clickable markdown links
    assert "[raw_images](datasets/raw_images.md)" in result
    assert "[labels](datasets/labels.md)" in result
