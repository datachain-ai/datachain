"""
Functional tests for read_dataset with PEP 440 version specifiers.
"""

import pytest

import datachain as dc
from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError
from tests.utils import skip_if_not_sqlite


def test_read_dataset_version_specifiers(test_session):
    """Test read_dataset with various PEP 440 version specifiers."""
    dataset_name = "test_version_specifiers"

    for version in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
        dc.read_values(data=[1, 2], session=test_session).save(
            dataset_name, version=version
        )

    assert (
        dc.read_dataset(dataset_name, version="==1.1.0", session=test_session).version
        == "1.1.0"
    )
    assert (
        dc.read_dataset(dataset_name, version=">=1.1.0", session=test_session).version
        == "2.0.0"
    )
    assert (
        dc.read_dataset(dataset_name, version="<2.0.0", session=test_session).version
        == "1.2.0"
    )
    assert (
        dc.read_dataset(dataset_name, version="~=1.0", session=test_session).version
        == "1.2.0"
    )
    assert (
        dc.read_dataset(dataset_name, version="==1.*", session=test_session).version
        == "1.2.0"
    )
    assert (
        dc.read_dataset(
            dataset_name, version=">=1.1.0,<2.0.0", session=test_session
        ).version
        == "1.2.0"
    )


def test_read_dataset_version_specifiers_no_match(test_session):
    """Test read_dataset with version specifiers that don't match any version."""
    dataset_name = "test_no_match_specifiers"

    dc.read_values(data=[1, 2], session=test_session).save(
        dataset_name, version="1.0.0"
    )

    with pytest.raises(DatasetVersionNotFoundError) as exc_info:
        dc.read_dataset(dataset_name, version=">=2.0.0", session=test_session)

    assert (
        "No dataset test_no_match_specifiers version matching specifier >=2.0.0"
        in str(exc_info.value)
    )


def test_read_dataset_version_specifiers_exact_version(test_session):
    dataset_name = "test_backward_compatibility"

    dc.read_values(data=[1, 2], session=test_session).save(
        dataset_name, version="1.0.0"
    )

    assert (
        dc.read_dataset(dataset_name, version="1.0.0", session=test_session).version
        == "1.0.0"
    )
    assert (
        dc.read_dataset(dataset_name, version=1, session=test_session).version
        == "1.0.0"
    )


def test_read_dataset_version_in_name(test_session):
    """Test read_dataset with version embedded in name using the @ syntax."""
    dataset_name = "test_version_in_name"

    for version in ["1.0.0", "2.0.0"]:
        dc.read_values(data=[1, 2], session=test_session).save(
            dataset_name, version=version
        )

    assert (
        dc.read_dataset(f"{dataset_name}@1.0.0", session=test_session).version
        == "1.0.0"
    )
    assert (
        dc.read_dataset(f"{dataset_name}@>=1.0.0,<2.0.0", session=test_session).version
        == "1.0.0"
    )

    assert (
        dc.read_dataset(
            f"{dataset_name}@1.0.0", version="2.0.0", session=test_session
        ).version
        == "2.0.0"
    )


def test_delete_dataset_version_in_name(test_session, no_studio_dataset):
    """Test delete_dataset with version embedded in name using the @ syntax."""
    dataset_name = "test_delete_version_in_name"

    for version in ["1.0.0", "2.0.0"]:
        dc.read_values(data=[1, 2], session=test_session).save(
            dataset_name, version=version
        )

    dc.delete_dataset(f"{dataset_name}@1.0.0", session=test_session)

    with pytest.raises(DatasetNotFoundError):
        list(dc.read_dataset(f"{dataset_name}@1.0.0", session=test_session).to_iter())

    assert (
        dc.read_dataset(f"{dataset_name}@2.0.0", session=test_session).version
        == "2.0.0"
    )


@skip_if_not_sqlite
def test_read_dataset_missing_version_raises_not_found(test_session, no_studio_dataset):
    dc.read_values(num=[1], session=test_session).save("selected", version="1.0.0")

    with pytest.raises(DatasetNotFoundError, match=r"9\.0\.0"):
        dc.read_dataset("selected", version="9.0.0", session=test_session)

    dataset = test_session.catalog.get_dataset("selected", versions=["1.0.0"])
    with pytest.raises(ValueError, match="Cannot pass 'dataset' without 'name'"):
        dc.query.DatasetQuery(name=None, dataset=dataset, session=test_session)
