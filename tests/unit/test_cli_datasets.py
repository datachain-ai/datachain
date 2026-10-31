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


def test_describe_saved_versions(test_session, capsys):
    import json

    import datachain as dc
    from datachain.cli.commands.datasets import describe_dataset

    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save(
        "source", version="1.0.0", query_script="MODEL = 'v1'\n"
    )
    dc.read_dataset("source", version="1.0.0", session=test_session).save("derived")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
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

    (tmp_path / "sample.txt").write_text("hello")
    dc.read_storage(str(tmp_path), session=test_session).save("inventory")
    dc.read_dataset("inventory", session=test_session).save("report")

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must only read recorded metadata")

    monkeypatch.setattr(Client, "get_implementation", forbidden)
    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    info = test_session.catalog.describe_dataset("report")
    assert any(str(tmp_path) in uri for uri in info["sources"])
    assert all(not uri.startswith("lst__") for uri in info["sources"])
    assert info["dependencies"][0]["dependencies"][0]["type"] == "storage"
    assert info["dependencies"][0]["dependencies"][0]["rows"] == 1


def test_describe_missing_dataset_or_version(test_session):
    import pytest

    import datachain as dc
    from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError

    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.describe_dataset("missing")
    dc.read_values(value=[1], session=test_session).save("source")
    with pytest.raises(DatasetVersionNotFoundError):
        test_session.catalog.describe_dataset("source@99.0.0")
