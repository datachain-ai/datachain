import pytest

from datachain.cli.commands.datasets import group_dataset_versions


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


@pytest.mark.parametrize("extended_columns", [False, True])
def test_describe_saved_versions(test_session, capsys, monkeypatch, extended_columns):
    import json

    import datachain as dc
    from datachain.cli.commands.datasets import describe_dataset

    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save(
        "source", version="1.0.0", query_script="MODEL = 'v1'\n"
    )
    dc.read_dataset("source", version="1.0.0", session=test_session).save("derived")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    if extended_columns:
        from sqlalchemy import literal

        metastore = catalog.metastore
        columns = metastore._dataset_dependencies_select_columns()
        parse = metastore.dependency_class.parse

        def parse_extended(*row, **kwargs):
            if row:
                assert row[-1] == "backend-extra"
            return parse(*row[:-1], **kwargs)

        monkeypatch.setattr(
            type(metastore),
            "_dataset_dependencies_select_columns",
            lambda self: [*columns, literal("backend-extra")],
        )
        monkeypatch.setattr(
            metastore.dependency_class, "parse", staticmethod(parse_extended)
        )
    info = catalog.describe_dataset("source@1.0.0")
    assert info["rows"] == 2
    assert info["version"] == "1.0.0"
    assert [v["version"] for v in info["versions"]] == ["1.0.0", "2.0.0"]
    assert info["query_script"] == "MODEL = 'v1'\n"
    assert info["schema"]["value"] == "int"
    assert [d["name"] for d in info["dependents"]] == ["derived"]
    assert catalog.describe_dataset("source")["dependents"] == []
    derived = catalog.describe_dataset("derived")
    assert [(d["name"], d["version"]) for d in derived["dependencies"]] == [
        ("source", "1.0.0")
    ]
    assert derived["dependencies"][0]["rows"] == 2
    assert info["dependents"][0]["rows"] == 2
    assert derived["external_state"] == "not_checked"
    describe_dataset(catalog, "source@1.0.0", json_output=True)
    assert json.loads(capsys.readouterr().out)["query_script"] == info["query_script"]
    describe_dataset(catalog, "source@1.0.0")
    output = capsys.readouterr().out
    assert "MODEL = 'v1'" in output
    assert "Rows: 2" in output


def test_describe_storage_sources(test_session, tmp_path, monkeypatch):
    import datachain as dc
    from datachain.client import Client
    from datachain.fs.utils import path_to_fsspec_uri

    tmp_path = tmp_path / "source with spaces"
    tmp_path.mkdir()
    (tmp_path / "sample.txt").write_text("hello")
    dc.read_storage(str(tmp_path), session=test_session).save("inventory")
    dc.read_dataset("inventory", session=test_session).save("report")

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must only read recorded metadata")

    monkeypatch.setattr(Client, "get_implementation", forbidden)
    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    info = test_session.catalog.describe_dataset("report")
    assert any(path_to_fsspec_uri(str(tmp_path)) in uri for uri in info["sources"])
    assert all(not uri.startswith("lst__") for uri in info["sources"])
    assert info["dependencies"][0]["dependencies"][0]["type"] == "storage"
    assert info["dependencies"][0]["dependencies"][0]["rows"] == 1


def test_describe_missing_dataset_or_version(test_session):
    import datachain as dc
    from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError

    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.describe_dataset("missing")
    dc.read_values(value=[1], session=test_session).save("source")
    with pytest.raises(DatasetVersionNotFoundError):
        test_session.catalog.describe_dataset("source@99.0.0")


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_describe_deleted_source(test_session, capsys, keep_metadata):
    import datachain as dc
    from datachain.cli.commands.datasets import describe_dataset

    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save("source")
    dc.read_dataset("source", session=test_session).save("derived", query_script="")
    catalog.remove_dataset("source", version="1.0.0", keep_metadata=keep_metadata)

    info = catalog.describe_dataset("derived")
    if keep_metadata:
        dependency = info["dependencies"][0]
        assert (dependency["name"], dependency["removed"], dependency["rows"]) == (
            "source",
            True,
            2,
        )
    else:
        assert info["dependencies"] == [None]
    describe_dataset(catalog, "derived")
    output = capsys.readouterr().out
    expected_line = (
        "local.local.source@1.0.0 (2 rows) (removed)"
        if keep_metadata
        else "unavailable dependency"
    )
    assert expected_line in output
    assert "No script recorded." in output


def test_describe_text_includes_description(test_session, capsys):
    import datachain as dc
    from datachain.cli.commands.datasets import describe_dataset

    dc.read_values(value=[1], session=test_session).save(
        "source", description="Recorded labels"
    )
    describe_dataset(test_session.catalog, "source")
    output = capsys.readouterr().out
    assert "Description: Recorded labels" in output
    assert "Sources: none recorded" in output


@pytest.mark.parametrize("missing_version", [False, True])
def test_describe_source_removed_during_inspection(
    test_session, monkeypatch, missing_version
):
    import datachain as dc
    from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError

    catalog = test_session.catalog
    dc.read_values(value=[1], session=test_session).save("source")
    dc.read_dataset("source", session=test_session).save("derived")
    get_dataset = catalog.get_dataset
    error = DatasetVersionNotFoundError if missing_version else DatasetNotFoundError

    def lookup(name, **kwargs):
        if name == "source":
            raise error("Source was removed during inspection")
        return get_dataset(name, **kwargs)

    monkeypatch.setattr(catalog, "get_dataset", lookup)
    dependency = catalog.describe_dataset("derived")["dependencies"][0]
    assert dependency["name"] == "source"
    assert dependency["version"] == "1.0.0"
    assert dependency["rows"] is None


def test_describe_shared_source_versions(test_session):
    import datachain as dc

    dc.read_values(value=[1, 2], session=test_session).save("source", version="1.0.0")
    dc.read_dataset("source", session=test_session).save("first")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    dc.read_dataset("source", session=test_session).save("second")
    dc.read_dataset("first", session=test_session).union(
        dc.read_dataset("second", session=test_session)
    ).save("combined")

    info = test_session.catalog.describe_dataset("combined")
    ancestors = {
        ancestor["version"]: ancestor["rows"]
        for dependency in info["dependencies"]
        for ancestor in dependency["dependencies"]
    }
    assert ancestors == {"1.0.0": 2, "2.0.0": 1}
