from datetime import UTC, datetime, timedelta

import pytest

from datachain.lib.listing import LISTING_TTL
from datachain.lib.listing_info import ListingInfo


@pytest.mark.parametrize(
    "date,is_expired",
    [
        (datetime.now(UTC), False),
        (datetime.now(UTC) - timedelta(seconds=LISTING_TTL + 1), True),
    ],
)
def test_is_listing_expired(date, is_expired):
    listing_info = ListingInfo(
        name="lst_s3://whatever",
        namespace="dev",
        project="my-project",
        finished_at=date,
    )
    assert listing_info.is_expired is is_expired


@pytest.mark.parametrize(
    "ds1_name,ds2_name,contains",
    [
        ("lst__s3://my-bucket/animals/", "lst__s3://my-bucket/animals/dogs/", True),
        ("lst__s3://my-bucket/animals/", "lst__s3://my-bucket/animals/", True),
        ("lst__s3://my-bucket/", "lst__s3://my-bucket/", True),
        ("lst__s3://my-bucket/cats/", "lst__s3://my-bucket/animals/dogs/", False),
        ("lst__s3://my-bucket/dogs/", "lst__s3://my-bucket/animals/", False),
        ("lst__s3://my-bucket/animals/", "lst__s3://other-bucket/animals/", False),
    ],
)
def test_listing_subset(ds1_name, ds2_name, contains):
    listing_info = ListingInfo(name=ds1_name, namespace="dev", project="my-project")
    assert listing_info.contains(ds2_name) is contains


@pytest.mark.parametrize(
    "name,versions",
    [
        ("lst__s3://b/dir/", True),
        ("lst__current__s3://b/dir/", False),
        ("lst__all__s3://b/dir/", "all"),
    ],
)
def test_listing_versions_mode(name, versions):
    info = ListingInfo(name=name, namespace="dev", project="my-project")
    assert info.versions == versions
    assert info.uri == "s3://b/dir/"
