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
)

# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestParseSemver:
    def test_valid(self):
        assert parse_semver("1.2.3") == (1, 2, 3)

    def test_two_part(self):
        assert parse_semver("1.0") == (1, 0)

    def test_invalid_string(self):
        assert parse_semver("bad") == (0, 0, 0)

    def test_none(self):
        assert parse_semver(None) == (0, 0, 0)


class TestReadFrontmatter:
    def test_normal(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("---\nname: foo\ndescription: bar\n---\n# body\n")
        fm = read_frontmatter(str(p))
        assert fm["name"] == "foo"
        assert fm["description"] == "bar"

    def test_no_frontmatter(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("# Just a heading\n")
        assert read_frontmatter(str(p)) == {}

    def test_empty_file(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("")
        assert read_frontmatter(str(p)) == {}

    def test_missing_file(self):
        assert read_frontmatter("/nonexistent/path.md") == {}

    def test_colon_in_value(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("---\ndescription: Use for: things\n---\n")
        fm = read_frontmatter(str(p))
        # partition on first colon preserves the rest
        assert fm["description"] == "Use for: things"

    def test_quoted_value(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text('---\nname: "quoted"\n---\n')
        fm = read_frontmatter(str(p))
        assert fm["name"] == "quoted"


class TestParseUri:
    def test_s3(self):
        result = parse_uri("s3://my-bucket/")
        assert result == {"scheme": "s3", "bucket": "my-bucket", "prefix": ""}

    def test_gs_with_prefix(self):
        result = parse_uri("gs://demo/dogs-cats/")
        assert result == {"scheme": "gs", "bucket": "demo", "prefix": "dogs-cats/"}

    def test_az(self):
        result = parse_uri("az://container/path/to/data")
        assert result["scheme"] == "az"
        assert result["bucket"] == "container"
        assert result["prefix"] == "path/to/data"


class TestBucketFilePath:
    def test_basic(self):
        assert bucket_file_path("s3://my-bucket/") == "buckets/s3/my_bucket"

    def test_with_prefix(self):
        result = bucket_file_path("gs://demo/dogs-cats/")
        assert result == "buckets/gs/demo__dogs_cats"

    def test_special_chars(self):
        result = bucket_file_path("s3://My.Bucket-Name/")
        assert result == "buckets/s3/my_bucket_name"


class TestDatasetFilePath:
    def test_local(self):
        assert dataset_file_path("my_dataset", "local") == "datasets/my_dataset"

    def test_studio(self):
        result = dataset_file_path("ns.proj.my_ds", "studio")
        assert result == "datasets/ns/proj/my_ds"

    def test_local_with_dots(self):
        # Local names with dots — treated as flat name
        result = dataset_file_path("my.dataset", "local")
        assert result == "datasets/my_dataset"


class TestHumanSize:
    def test_bytes(self):
        assert human_size(500) == "500 B"

    def test_kb(self):
        assert human_size(1536) == "1.5 KB"

    def test_mb(self):
        assert human_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gb(self):
        assert human_size(2.5 * 1024**3) == "2.5 GB"

    def test_zero(self):
        assert human_size(0) == "0 B"


class TestSerialize:
    def test_primitives(self):
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize(3.14) == 3.14
        assert serialize(True) is True
        assert serialize(None) is None

    def test_object(self):
        result = serialize(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert isinstance(result, str)


class TestReadJsonVersions:
    def test_valid(self, tmp_path):
        p = tmp_path / "ds.json"
        p.write_text(
            json.dumps(
                {
                    "versions": [
                        {"version": "1.0.0"},
                        {"version": "2.0.0"},
                    ]
                }
            )
        )
        assert read_json_versions(str(p)) == ["1.0.0", "2.0.0"]

    def test_empty_versions(self, tmp_path):
        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"versions": []}))
        assert read_json_versions(str(p)) == []

    def test_missing_file(self):
        assert read_json_versions("/nonexistent.json") == []


class TestReadJsonMetadata:
    def test_valid(self, tmp_path):
        p = tmp_path / "ds.json"
        p.write_text(
            json.dumps({"versions": [{"version": "1.0.0", "num_objects": 42}]})
        )
        result = read_json_metadata(str(p))
        assert result["latest_version"] == "1.0.0"
        assert result["num_objects"] == "42"

    def test_empty(self, tmp_path):
        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"versions": []}))
        assert read_json_metadata(str(p)) == {}


class TestReadJsonData:
    def test_valid(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "value"}))
        assert read_json_data(str(p)) == {"key": "value"}

    def test_missing(self):
        assert read_json_data("/nonexistent.json") is None


# ---------------------------------------------------------------------------
# changes.py
# ---------------------------------------------------------------------------


class TestComputeDepChanges:
    def test_added(self):
        result = compute_dep_changes(
            [{"name": "a", "version": "1.0"}],
            [],
        )
        assert result["deps_added"] == [{"name": "a", "version": "1.0"}]
        assert result["deps_removed"] == []
        assert result["deps_updated"] == []

    def test_removed(self):
        result = compute_dep_changes(
            [],
            [{"name": "a", "version": "1.0"}],
        )
        assert result["deps_removed"] == [{"name": "a", "version": "1.0"}]

    def test_updated(self):
        result = compute_dep_changes(
            [{"name": "a", "version": "2.0"}],
            [{"name": "a", "version": "1.0"}],
        )
        assert len(result["deps_updated"]) == 1
        assert result["deps_updated"][0]["version_from"] == "1.0"
        assert result["deps_updated"][0]["version_to"] == "2.0"

    def test_unchanged(self):
        deps = [{"name": "a", "version": "1.0"}]
        result = compute_dep_changes(deps, deps)
        assert result["deps_added"] == []
        assert result["deps_removed"] == []
        assert result["deps_updated"] == []

    def test_empty(self):
        result = compute_dep_changes([], [])
        assert result == {
            "deps_added": [],
            "deps_removed": [],
            "deps_updated": [],
        }


class TestBuildChanges:
    def test_script_changed(self):
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

    def test_script_unchanged(self):
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


class TestTypeName:
    def test_primitives(self):
        assert type_name(str) == "str"
        assert type_name(int) == "int"
        assert type_name(float) == "float"

    def test_none_type(self):
        assert type_name(type(None)) == "None"

    def test_list(self):
        assert type_name(list[str]) == "list[str]"

    def test_dict(self):
        assert type_name(dict[str, int]) == "dict"

    def test_union(self):
        result = type_name(str | None)
        assert "str" in result
        assert "None" in result

    def test_optional(self):
        from typing import Optional

        result = type_name(Optional[int])  # noqa: UP045
        assert "int" in result
        assert "None" in result


class TestParseDatasetName:
    def test_local(self):
        assert parse_dataset_name("my_dataset") == (None, None, "my_dataset")

    def test_dot_separated(self):
        assert parse_dataset_name("ns.proj.name") == ("ns", "proj", "name")

    def test_slash_separated(self):
        assert parse_dataset_name("ns/proj/name") == ("ns", "proj", "name")


# ---------------------------------------------------------------------------
# render_index.py
# ---------------------------------------------------------------------------


class TestRenderIndex:
    def test_local_datasets(self):
        plan = {
            "db_last_updated": "2024-01-01T00:00:00Z",
            "datasets": [
                {
                    "name": "test_ds",
                    "source": "local",
                    "latest_version": "1.0.0",
                    "num_objects": 100,
                    "updated_at": None,
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
        # No Source column
        assert "| Source |" not in result
        assert "## Buckets" not in result

    def test_empty_plan(self):
        result = render_index({"datasets": [], "buckets": []})
        assert "generated_at:" in result
        assert "## Datasets" not in result
        assert "## Buckets" not in result

    def test_frontmatter_counts(self):
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

    def test_studio_namespace_subsections(self):
        plan = {
            "datasets": [
                {
                    "name": "ns.proj.ds_a",
                    "source": "studio",
                    "latest_version": "1.0.0",
                    "num_objects": 50,
                    "updated_at": None,
                    "file_path": "datasets/ns/proj/ds_a",
                },
                {
                    "name": "ns.proj.ds_b",
                    "source": "studio",
                    "latest_version": "2.0.0",
                    "num_objects": 200,
                    "updated_at": None,
                    "file_path": "datasets/ns/proj/ds_b",
                },
                {
                    "name": "other.team.ds_c",
                    "source": "studio",
                    "latest_version": "1.0.0",
                    "num_objects": 10,
                    "updated_at": None,
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

    def test_mixed_local_and_studio(self):
        plan = {
            "datasets": [
                {
                    "name": "local_ds",
                    "source": "local",
                    "latest_version": "1.0.0",
                    "file_path": "datasets/local_ds",
                },
                {
                    "name": "ns.proj.studio_ds",
                    "source": "studio",
                    "latest_version": "2.0.0",
                    "file_path": "datasets/ns/proj/studio_ds",
                },
            ],
            "buckets": [],
        }
        result = render_index(plan)
        # Local section uses ## Datasets (no "Local" header)
        assert "## Datasets" in result
        assert "## Local" not in result
        # Studio section separate
        assert "## Studio" in result
        assert "### ns.proj" in result

    def test_deps_and_summary_from_md(self, tmp_path, monkeypatch):
        import render_index as ri

        monkeypatch.setattr(ri, "BASE_DIR", str(tmp_path))
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        (ds_dir / "my_ds.md").write_text(
            "---\nname: my_ds\n---\n\n# my_ds\n\n"
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
                    "latest_version": "1.0.0",
                    "file_path": "datasets/my_ds",
                },
            ],
            "buckets": [],
        }
        result = ri.render_index(plan)
        assert "Image metadata with EXIF and GPS for 12k photos." in result
        assert "raw_images, labels" in result
