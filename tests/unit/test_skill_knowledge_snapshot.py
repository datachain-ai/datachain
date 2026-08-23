"""Tests for the shared dataset knowledge snapshot builder."""

from datetime import datetime, timezone
from types import SimpleNamespace

from datachain.lib.file import File
from datachain.lib.signal_schema import SignalSchema
from datachain.skill.knowledge.scripts.utils import dep_entry
from datachain.skill.knowledge.snapshot import (
    MAX_PREVIEW_CELL_CHARS,
    MAX_VERSION_ENTRIES,
    build_dataset_snapshot,
    version_preview,
    version_schema,
)


def _version(
    version, *, query_script="", schema=None, feature_schema=None, preview=None
):
    return SimpleNamespace(
        version=version,
        uuid=f"uuid-{version}",
        num_objects=10,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        query_script=query_script,
        schema=schema or {},
        feature_schema=feature_schema,
        preview=preview,
    )


def _no_deps(_version):
    return []


def test_build_dataset_snapshot_basic():
    snap = build_dataset_snapshot(
        name="ns.proj.ds",
        source="studio",
        attrs=["a"],
        description="desc",
        versions=[_version("1.0.0", query_script="x")],
        deps_provider=_no_deps,
    )
    assert snap["name"] == "ns.proj.ds"
    assert snap["source"] == "studio"
    assert snap["attrs"] == ["a"]
    assert snap["description"] == "desc"
    assert len(snap["versions"]) == 1
    entry = snap["versions"][0]
    assert entry["version"] == "1.0.0"
    assert entry["uuid"] == "uuid-1.0.0"
    assert entry["updated"] == "2026-01-02T00:00:00+00:00"
    assert entry["changes"] is None
    assert snap["warnings"] == []


def test_build_dataset_snapshot_no_versions_warns():
    snap = build_dataset_snapshot(
        name="ds",
        source="local",
        attrs=[],
        description=None,
        versions=[],
        deps_provider=_no_deps,
    )
    assert snap["versions"] == []
    assert "dataset has no versions" in snap["warnings"]


def test_build_dataset_snapshot_orders_versions_oldest_first():
    snap = build_dataset_snapshot(
        name="ds",
        source="local",
        attrs=[],
        description=None,
        versions=[_version("1.0.10"), _version("1.0.2"), _version("1.0.1")],
        deps_provider=_no_deps,
    )
    assert [v["version"] for v in snap["versions"]] == ["1.0.1", "1.0.2", "1.0.10"]


def test_build_dataset_snapshot_caps_and_warns():
    versions = [
        _version(f"1.0.{i}", query_script=f"v{i}")
        for i in range(MAX_VERSION_ENTRIES + 5)
    ]
    snap = build_dataset_snapshot(
        name="ds",
        source="local",
        attrs=[],
        description=None,
        versions=versions,
        deps_provider=_no_deps,
    )
    assert len(snap["versions"]) == MAX_VERSION_ENTRIES
    assert any("truncated" in w for w in snap["warnings"])
    assert snap["versions"][0]["changes"] is not None


def test_build_dataset_snapshot_dangling_dependency_warns_and_filters():
    deps = {
        "1.0.0": [
            dep_entry("ns.proj.up", "1.0.0", "dataset"),
            dep_entry(None, None, None),  # deleted target
        ]
    }
    snap = build_dataset_snapshot(
        name="ds",
        source="studio",
        attrs=[],
        description=None,
        versions=[_version("1.0.0")],
        deps_provider=lambda v: deps.get(v.version, []),
    )
    rendered = snap["versions"][0]["dependencies"]
    assert [d["name"] for d in rendered] == ["ns.proj.up"]
    assert any("deleted dataset" in w for w in snap["warnings"])


def test_version_schema_from_feature_schema():
    feature_schema = SignalSchema(
        {"file": File, "score": float, "label": str}
    ).serialize()
    result = version_schema(SimpleNamespace(feature_schema=feature_schema, schema={}))
    assert result == {
        "file": {"type": "File", "fields": None},
        "score": {"type": "float", "fields": None},
        "label": {"type": "str", "fields": None},
    }


def test_version_schema_flat_fallback_filters_sys():
    result = version_schema(
        SimpleNamespace(
            feature_schema=None, schema={"file.path": "str", "sys.id": "int"}
        )
    )
    assert result == {"file.path": {"type": "str", "fields": None}}


def test_version_preview_pivots_list_of_dicts():
    rows = [{"a": 1, "b": 2, "sys__id": 9}, {"a": 3, "b": 4}]
    assert version_preview(SimpleNamespace(preview=rows)) == {
        "columns": ["a", "b"],
        "rows": [[1, 2], [3, 4]],
    }


def test_version_preview_dots_nested_signal_columns():
    rows = [{"file__source": "s3", "file__path": "a.jpg", "size": 10}]
    assert version_preview(SimpleNamespace(preview=rows)) == {
        "columns": ["file.source", "file.path", "size"],
        "rows": [["s3", "a.jpg", 10]],
    }


def test_version_preview_passthrough_dict():
    shaped = {"columns": ["a"], "rows": [[1]]}
    assert version_preview(SimpleNamespace(preview=shaped)) == shaped


def test_version_preview_empty_returns_none():
    assert version_preview(SimpleNamespace(preview=None)) is None
    assert version_preview(SimpleNamespace(preview=[])) is None


def test_version_preview_caps_large_cells():
    rows = [{"emb": "x" * 10_000, "id": 1}]
    result = version_preview(SimpleNamespace(preview=rows))
    assert result["columns"] == ["emb", "id"]
    cell = result["rows"][0][0]
    assert len(cell) <= MAX_PREVIEW_CELL_CHARS + 1
    assert cell.endswith("…")
    assert result["rows"][0][1] == 1


def test_version_preview_caps_large_cells_in_dict_shape():
    shaped = {"columns": ["emb"], "rows": [["x" * 10_000]]}
    result = version_preview(SimpleNamespace(preview=shaped))
    cell = result["rows"][0][0]
    assert len(cell) <= MAX_PREVIEW_CELL_CHARS + 1
    assert cell.endswith("…")
