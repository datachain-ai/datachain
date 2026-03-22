"""
Functional tests for read_dataset with PEP 440 version specifiers.
"""

import pytest

import datachain as dc
from datachain.error import DatasetVersionNotFoundError


def test_read_dataset_version_specifiers(test_session):
    """Test read_dataset with various PEP 440 version specifiers."""
    # Create a dataset with multiple versions
    dataset_name = "test_version_specifiers"

    for version in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
        (
            dc.read_values(data=[1, 2], session=test_session)
            .mutate(dataset_version=version)
            .save(dataset_name, version=version)
        )

    # Test exact version specifier
    result = dc.read_dataset(dataset_name, version="==1.1.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.1.0"

    # Test greater than or equal specifier - should get latest (2.0.0)
    result = dc.read_dataset(dataset_name, version=">=1.1.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "2.0.0"

    # Test less than specifier - should get 1.2.0 (latest before 2.0.0)
    result = dc.read_dataset(dataset_name, version="<2.0.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.2.0"

    # Test compatible release specifier - should get latest 1.x (1.2.0)
    result = dc.read_dataset(dataset_name, version="~=1.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.2.0"

    # Test version pattern - should get latest 1.x (1.2.0)
    result = dc.read_dataset(dataset_name, version="==1.*", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.2.0"

    # Test complex specifier - should get 1.2.0
    result = dc.read_dataset(
        dataset_name, version=">=1.1.0,<2.0.0", session=test_session
    )
    assert result.to_values("dataset_version")[0] == "1.2.0"


def test_read_dataset_version_specifiers_no_match(test_session):
    """Test read_dataset with version specifiers that don't match any version."""
    # Create a dataset with a single version
    dataset_name = "test_no_match_specifiers"

    (
        dc.read_values(data=[1, 2], session=test_session)
        .mutate(dataset_version="1.0.0")
        .save(dataset_name, version="1.0.0")
    )

    # Test version specifier that doesn't match any existing version
    with pytest.raises(DatasetVersionNotFoundError) as exc_info:
        dc.read_dataset(dataset_name, version=">=2.0.0", session=test_session)

    assert (
        "No dataset test_no_match_specifiers version matching specifier >=2.0.0"
        in str(exc_info.value)
    )


def test_read_dataset_version_specifiers_exact_version(test_session):
    """Test that version specifiers work alongside with exact version reads."""
    # Create a dataset with multiple versions
    dataset_name = "test_backward_compatibility"

    (
        dc.read_values(data=[1, 2], session=test_session)
        .mutate(dataset_version="1.0.0")
        .save(dataset_name, version="1.0.0")
    )

    # Test reading by exact version
    result = dc.read_dataset(dataset_name, version="1.0.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.0.0"

    # Test reading by exact version int - backward compatibility
    result = dc.read_dataset(dataset_name, version=1, session=test_session)
    assert result.to_values("dataset_version")[0] == "1.0.0"


def test_read_dataset_version_in_name(test_session):
    """Test read_dataset with version embedded in name using the @ syntax."""
    dataset_name = "test_version_in_name"

    for version in ["1.0.0", "2.0.0"]:
        (
            dc.read_values(data=[1, 2], session=test_session)
            .mutate(dataset_version=version)
            .save(dataset_name, version=version)
        )

    # Test exact version embedded in name
    result = dc.read_dataset(f"{dataset_name}@1.0.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.0.0"

    # Test version specifier embedded in name
    result = dc.read_dataset(f"{dataset_name}@>=1.0.0,<2.0.0", session=test_session)
    assert result.to_values("dataset_version")[0] == "1.0.0"

    # Test that specifying version both in name and as parameter raises an error
    with pytest.raises(ValueError, match="Cannot specify version both"):
        dc.read_dataset(f"{dataset_name}@1.0.0", version="2.0.0", session=test_session)
