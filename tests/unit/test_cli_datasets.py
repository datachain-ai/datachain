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
