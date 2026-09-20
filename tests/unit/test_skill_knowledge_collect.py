"""Tests for the metastore-driven KB collectors."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.skill.knowledge.collect import collect_dataset_snapshot


def _version(version, *, query_script="", status=DatasetStatus.COMPLETE):
    return SimpleNamespace(
        version=version,
        status=status,
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
        attrs=["pets"],
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


def test_collect_dataset_snapshot_qualifies_name_and_loads_all_versions():
    ms = _StubMetastore(_record([_version("1.0.0", query_script="x")]))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0.0")

    assert snap["name"] == "ns.proj.pet_images"
    assert snap["source"] == "studio"
    assert snap["attrs"] == ["pets"]
    assert snap["description"] == "pets"
    assert [v["version"] for v in snap["versions"]] == ["1.0.0"]
    _, ns, proj, kwargs = ms.get_dataset_calls[0]
    assert (ns, proj) == ("ns", "proj")
    assert kwargs["include_incomplete"] is True
    assert kwargs["versions"] is None
    assert kwargs["include_preview"] is True


def test_collect_dataset_snapshot_project_less_uses_bare_name():
    record = _record([_version("1.0.0")])
    record.project = None
    ms = _StubMetastore(record)

    snap = collect_dataset_snapshot(ms, "pet_images", version="1.0.0")

    assert snap["name"] == "pet_images"


def test_collect_dataset_snapshot_dataset_dependency_is_qualified():
    deps = {"1.0.0": [_dep(DatasetDependencyType.DATASET, "upstream", "1.0.0")]}
    ms = _StubMetastore(_record([_version("1.0.0")]), deps)

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0.0")

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

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0.0")

    dep = snap["versions"][0]["dependencies"][0]
    assert dep["type"] == "storage"
    assert dep["name"] == "s3://my.bucket/data/"
    assert dep["file_path"] == "buckets/s3/my_bucket/data"


def test_collect_dataset_snapshot_deleted_dependency_dropped_and_warned():
    ms = _StubMetastore(_record([_version("1.0.0")]), {"1.0.0": [None]})

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0.0")

    assert snap["versions"][0]["dependencies"] == []
    assert any("deleted dataset" in w for w in snap["warnings"])


def test_collect_dataset_snapshot_describes_the_requested_version():
    versions = [_version("1.0.0"), _version("2.0.0"), _version("10.0.0")]
    ms = _StubMetastore(_record(versions))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="2.0.0")

    assert [v["version"] for v in snap["versions"]] == ["1.0.0", "2.0.0"]


def test_collect_dataset_snapshot_rejects_an_unknown_version():
    ms = _StubMetastore(_record([_version("1.0.0")]))

    with pytest.raises(ValueError, match=r"no completed version 9\.0\.0"):
        collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="9.0.0")


def test_collect_dataset_snapshot_keeps_the_exact_target_last_on_semver_ties():
    versions = [_version("1.0"), _version("1.0.0"), _version("2.0.0")]
    ms = _StubMetastore(_record(versions))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0")

    assert snap["versions"][-1]["version"] == "1.0"


def test_collect_dataset_snapshot_keeps_the_target_when_history_is_capped():
    versions = [_version(f"{i}.0.0") for i in range(1, 30)]
    ms = _StubMetastore(_record(versions))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="25.0.0")

    assert snap["versions"][-1]["version"] == "25.0.0"
    assert len(snap["versions"]) == 20
    assert any("truncated" in w for w in snap["warnings"])


def test_collect_dataset_snapshot_rejects_a_dataset_with_no_completed_versions():
    ms = _StubMetastore(_record([_version("1.0.0", status=DatasetStatus.CREATED)]))

    with pytest.raises(ValueError, match=r"no completed version 1\.0\.0"):
        collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="1.0.0")


def test_collect_dataset_snapshot_excludes_incomplete_versions_from_history():
    incomplete = _version("2.0.0", status=DatasetStatus.CREATED)
    ms = _StubMetastore(_record([_version("1.0.0"), incomplete, _version("3.0.0")]))

    snap = collect_dataset_snapshot(ms, "pet_images", "ns", "proj", version="3.0.0")

    assert [v["version"] for v in snap["versions"]] == ["1.0.0", "3.0.0"]
