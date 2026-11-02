import json

import pytest

import datachain as dc
from datachain.cli.commands.datasets import describe_dataset, group_dataset_versions
from datachain.client import Client
from datachain.dataset import DatasetDependency, DatasetStatus
from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError
from datachain.fs.utils import path_to_fsspec_uri


def test_group_dataset_versions_latest_only():
    """Test grouping datasets with latest_only=True returns only highest version."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "1.5.0"),
        ("dataset2", "0.1.0"),
        ("dataset2", "0.2.0"),
    ]

    result = group_dataset_versions(datasets, latest_only=True)

    assert result == {
        "dataset1": "2.0.0",
        "dataset2": "0.2.0",
    }


def test_group_dataset_versions_all_versions():
    """Test grouping datasets with latest_only=False returns all versions."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "1.5.0"),
        ("dataset2", "0.1.0"),
        ("dataset2", "0.2.0"),
    ]

    result = group_dataset_versions(datasets, latest_only=False)

    assert result == {
        "dataset1": ["1.0.0", "1.5.0", "2.0.0"],
        "dataset2": ["0.1.0", "0.2.0"],
    }


def test_group_dataset_versions_empty_input():
    """Test grouping empty dataset list."""
    result = group_dataset_versions([], latest_only=True)
    assert result == {}


def test_group_dataset_versions_single_dataset():
    """Test grouping single dataset with single version."""
    datasets = [("dataset1", "1.0.0")]

    result = group_dataset_versions(datasets, latest_only=True)
    assert result == {"dataset1": "1.0.0"}


def test_group_dataset_versions_semver_parsing():
    """Test that semver parsing is used correctly."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "10.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "9.9.9"),
    ]

    result = group_dataset_versions(datasets, latest_only=True)
    assert result == {"dataset1": "10.0.0"}


@pytest.fixture
def describe(capsys):
    def run(catalog, name, version=None):
        describe_dataset(catalog, name, version, json_output=True)
        return json.loads(capsys.readouterr().out)

    return run


@pytest.fixture
def prefix(test_session):
    metastore = test_session.catalog.metastore
    return f"{metastore.default_namespace_name}.{metastore.default_project_name}"


@pytest.fixture
def lineage(test_session):
    dc.read_values(value=[1, 2], session=test_session).save(
        "source", version="1.0.0", query_script="MODEL = 'v1'\n"
    )
    dc.read_dataset("source", version="1.0.0", session=test_session).save("derived")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    metastore = test_session.catalog.metastore
    source = metastore.get_dataset("source", versions=None)
    metastore.create_dataset_version(source, "3.0.0", status=DatasetStatus.CREATED)
    derived, _ = metastore.create_dataset_version(
        metastore.get_dataset("derived", versions=None),
        "2.0.0",
        status=DatasetStatus.FAILED,
    )
    metastore.add_dataset_dependency(derived, "2.0.0", source, "1.0.0")
    return test_session.catalog


@pytest.mark.parametrize(
    "name,version,qualified",
    [
        ("source@1.0.0", None, False),
        ("source", "1.0.0", False),
        ("source@1.0.0", None, True),
    ],
)
def test_describe_selects_version(lineage, describe, prefix, name, version, qualified):
    if qualified:
        name = f"{prefix}.{name}"
    info = describe(lineage, name, version)
    assert (info["name"], info["version"], info["rows"]) == (
        f"{prefix}.source",
        "1.0.0",
        2,
    )
    assert info["schema"]["value"] == "int"
    assert info["query_script"] == "MODEL = 'v1'\n"
    assert [(v["version"], v["rows"]) for v in info["versions"]] == [
        ("1.0.0", 2),
        ("2.0.0", 1),
    ]
    assert [(d["name"], d["version"], d["rows"]) for d in info["dependents"]] == [
        (f"{prefix}.derived", "1.0.0", 2)
    ]


def test_describe_only_complete_versions(lineage, describe):
    info = describe(lineage, "source")
    assert (info["version"], info["rows"], info["dependents"]) == ("2.0.0", 1, [])
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset(lineage, "source", "3.0.0")
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset(lineage, "source@99.0.0")
    with pytest.raises(DatasetNotFoundError):
        describe_dataset(lineage, "missing")


def test_describe_text(lineage, capsys, prefix):
    describe_dataset(lineage, "derived")
    output = capsys.readouterr().out
    assert f"{prefix}.derived@1.0.0\nRows: 2\n" in output
    assert "Versions: 1.0.0\nSchema:\n  value: int\n" in output
    assert (
        "Sources: none recorded\n"
        f"Depends on: {prefix}.source@1.0.0\n"
        "Lineage:\n"
        f"  {prefix}.source@1.0.0 (2 rows)\n"
        "Dependents: none recorded\n"
        "Script:\n"
    ) in output

    describe_dataset(lineage, "source@1.0.0")
    assert (
        "Depends on: none recorded\n"
        "Lineage: none recorded\n"
        "Dependents:\n"
        f"  {prefix}.derived@1.0.0 (2 rows)\n"
        "Script:\n"
        "MODEL = 'v1'\n"
    ) in capsys.readouterr().out


def test_describe_storage_lineage_reads_metadata_only(
    test_session, tmp_path, monkeypatch, capsys, describe, prefix
):
    path = tmp_path / "source with spaces"
    path.mkdir()
    (path / "sample.txt").write_text("hello")
    dc.read_storage(str(path), session=test_session).save("inventory")
    dc.read_dataset("inventory", session=test_session).save("report")

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must only read recorded metadata")

    monkeypatch.setattr(Client, "get_implementation", forbidden)
    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    uri = path_to_fsspec_uri(str(path)) + "/"
    listing = test_session.catalog.get_dataset_dependencies("inventory", "1.0.0")[0]
    assert listing is not None
    listing_ref = (
        f"{listing.namespace}.{listing.project}.{listing.name}@{listing.version}"
    )
    info = describe(test_session.catalog, "report")
    assert info["sources"] == [uri]
    assert info["depends_on"] == [f"{prefix}.inventory@1.0.0"]
    assert [
        (e["name"], e["version"], e["type"], e["rows"], e["removed"], e["depends_on"])
        for e in info["lineage"]
    ] == [
        (f"{prefix}.inventory", "1.0.0", "dataset", 1, False, [listing_ref]),
        (uri, "1.0.0", "storage", 1, False, []),
    ]

    describe_dataset(test_session.catalog, "report")
    output = capsys.readouterr().out
    assert (
        f"  {prefix}.inventory@1.0.0 (1 rows)\n"
        f"    depends on: {uri} (all versions)@1.0.0\n"
        f"  {uri} (all versions)@1.0.0 (1 rows)\n"
    ) in output
    assert "Schema:\n  file: File@v1\n    source: str\n" in output
    assert "lst__" not in output


def test_describe_lineage_lists_each_version_once(test_session, describe, prefix):
    def read(name, version=None):
        return dc.read_dataset(name, version=version, session=test_session)

    dc.read_values(value=[1, 2], session=test_session).save("source", version="1.0.0")
    read("source", "1.0.0").save("first")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    read("first").union(read("source", "2.0.0")).save("second")
    read("first").union(read("second")).save("top")

    info = describe(test_session.catalog, "top")
    assert sorted(info["depends_on"]) == [
        f"{prefix}.first@1.0.0",
        f"{prefix}.second@1.0.0",
    ]
    lineage = {
        f"{e['name']}@{e['version']}": (e["rows"], sorted(e["depends_on"]))
        for e in info["lineage"]
    }
    assert len(lineage) == len(info["lineage"])
    assert lineage == {
        f"{prefix}.first@1.0.0": (2, [f"{prefix}.source@1.0.0"]),
        f"{prefix}.second@1.0.0": (
            3,
            [f"{prefix}.first@1.0.0", f"{prefix}.source@2.0.0"],
        ),
        f"{prefix}.source@1.0.0": (2, []),
        f"{prefix}.source@2.0.0": (1, []),
    }


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_describe_removed_dependency(
    test_session, capsys, describe, prefix, keep_metadata
):
    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save("source")
    dc.read_dataset("source", session=test_session).save("derived", query_script="")
    catalog.remove_dataset("source", version="1.0.0", keep_metadata=keep_metadata)

    info = describe(catalog, "derived")
    if keep_metadata:
        assert info["depends_on"] == [f"{prefix}.source@1.0.0"]
        assert [(e["rows"], e["removed"]) for e in info["lineage"]] == [(2, True)]
    else:
        assert info["depends_on"] == [None]
        assert info["lineage"] == []

    describe_dataset(catalog, "derived")
    output = capsys.readouterr().out
    expected = (
        f"Lineage:\n  {prefix}.source@1.0.0 (2 rows) (removed)\n"
        if keep_metadata
        else "Depends on: unavailable dependency\n"
    )
    assert expected in output
    assert "Script:\nNo script recorded.\n" in output


def test_describe_text_includes_description(test_session, capsys):
    dc.read_values(value=[1], session=test_session).save(
        "source", description="Recorded labels"
    )
    describe_dataset(test_session.catalog, "source")
    assert "Description: Recorded labels\n" in capsys.readouterr().out


def test_describe_shared_lineage_populates_each_edge_once(
    test_session, describe, monkeypatch
):
    metastore = test_session.catalog.metastore
    previous = []
    edge_count = 0
    for layer in range(18):
        current = []
        for side in range(2):
            record = metastore.create_dataset(name=f"layer_{layer}_{side}")
            record, _ = metastore.create_dataset_version(
                record, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=1
            )
            for parent in previous:
                metastore.add_dataset_dependency(record, "1.0.0", parent, "1.0.0")
                edge_count += 1
            current.append(record)
        previous = current

    original = DatasetDependency.__setattr__
    writes = 0

    def bounded(self, name, value):
        nonlocal writes
        if name == "dependencies":
            writes += 1
            assert writes <= 4 * edge_count
        original(self, name, value)

    monkeypatch.setattr(DatasetDependency, "__setattr__", bounded)
    info = describe(test_session.catalog, "layer_17_0")
    assert len(info["lineage"]) == 34


def test_describe_keeps_listing_kinds_apart(test_session, describe, capsys):
    metastore = test_session.catalog.metastore

    def dataset(name, rows, project_id=None):
        record = metastore.create_dataset(name=name, project_id=project_id)
        return metastore.create_dataset_version(
            record, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=rows
        )[0]

    full = dataset("lst__s3://bucket/dir/", 10, metastore.listing_project.id)
    current = dataset("lst__current__s3://bucket/dir/", 3, metastore.listing_project.id)
    top = dataset("top", 13)
    metastore.add_dataset_dependency(top, "1.0.0", full, "1.0.0")
    metastore.add_dataset_dependency(top, "1.0.0", current, "1.0.0")

    info = describe(test_session.catalog, "top")
    assert len(set(info["depends_on"])) == 2
    assert set(info["depends_on"]) == {e["ref"] for e in info["lineage"]}
    assert sorted((e["name"], e["rows"]) for e in info["lineage"]) == [
        ("s3://bucket/dir/", 3),
        ("s3://bucket/dir/", 10),
    ]
    describe_dataset(test_session.catalog, "top")
    output = capsys.readouterr().out
    assert output.count("s3://bucket/dir/ (all versions)@1.0.0") == 2
    assert output.count("s3://bucket/dir/ (current only)@1.0.0") == 2
