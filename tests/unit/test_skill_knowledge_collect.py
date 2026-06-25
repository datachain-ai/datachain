"""Tests for the metastore-driven KB collectors."""

from datetime import datetime, timezone
from types import SimpleNamespace

from datachain.dataset import DatasetDependencyType
from datachain.skill.knowledge.collect import collect_dataset_snapshot


def _version(version, *, query_script=""):
    return SimpleNamespace(
        version=version,
        uuid=f"uuid-{version}",
        num_objects=10,
        size=100,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        query_script=query_script,
        schema={},
        feature_schema=None,
        preview=None,
    )


def _record(versions):
    return SimpleNamespace(
        name="pet_images",
        project=SimpleNamespace(name="proj", namespace=SimpleNamespace(name="ns")),
        attrs=["cast:l1"],
        description="pets",
        versions=versions,
    )


def _dep(type, name, version):
    return SimpleNamespace(
        type=type, namespace="ns", project="proj", name=name, version=version
    )


class _StubMetastore:
    def __init__(self, record, deps_by_version=None):
        self._record = record
        self._deps = deps_by_version or {}
        self.get_dataset_calls = []

    def get_dataset(self, name, namespace=None, project=None, **kwargs):
        self.get_dataset_calls.append((name, namespace, project, kwargs))
        return self._record

    def get_direct_dataset_dependencies(self, dataset, version):
        return self._deps.get(version, [])


def test_collect_dataset_snapshot_qualifies_name_and_fetches_complete():
    ms = _StubMetastore(_record([_version("1.0.0", query_script="x")]))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj")

    assert snap["name"] == "ns.proj.pet_images"
    assert snap["source"] == "studio"
    assert snap["attrs"] == ["cast:l1"]
    assert snap["description"] == "pets"
    assert [v["version"] for v in snap["versions"]] == ["1.0.0"]
    # COMPLETE-only, all versions, with preview — so the latest's preview is available.
    _, ns, proj, kwargs = ms.get_dataset_calls[0]
    assert (ns, proj) == ("ns", "proj")
    assert kwargs["include_incomplete"] is False
    assert kwargs["versions"] is None
    assert kwargs["include_preview"] is True


def test_collect_dataset_snapshot_project_less_uses_bare_name():
    record = _record([_version("1.0.0")])
    record.project = None
    ms = _StubMetastore(record)

    snap = collect_dataset_snapshot(ms, "pet_images")

    assert snap["name"] == "pet_images"


def test_collect_dataset_snapshot_dataset_dependency_is_qualified():
    deps = {"1.0.0": [_dep(DatasetDependencyType.DATASET, "upstream", "1.0.0")]}
    ms = _StubMetastore(_record([_version("1.0.0")]), deps)

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj")

    dep = snap["versions"][0]["dependencies"][0]
    assert dep["type"] == "dataset"
    assert dep["name"] == "ns.proj.upstream"
    assert dep["file_path"] == "datasets/ns/proj/upstream"


def test_collect_dataset_snapshot_storage_dependency_cleans_listing_name():
    deps = {
        "1.0.0": [
            _dep(DatasetDependencyType.STORAGE, "lst__s3://my_x2ebucket/data/", "1.0.0")
        ]
    }
    ms = _StubMetastore(_record([_version("1.0.0")]), deps)

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj")

    dep = snap["versions"][0]["dependencies"][0]
    assert dep["type"] == "storage"
    assert dep["name"] == "s3://my.bucket/data/"
    assert dep["file_path"] == "buckets/s3/my_bucket/data"


def test_collect_dataset_snapshot_deleted_dependency_dropped_and_warned():
    # A deleted target surfaces as a None edge; it must drop from the rendered
    # dependencies but still warn that the lineage is incomplete.
    ms = _StubMetastore(_record([_version("1.0.0")]), {"1.0.0": [None]})

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj")

    assert snap["versions"][0]["dependencies"] == []
    assert any("deleted dataset" in w for w in snap["warnings"])
