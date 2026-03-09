import pytest

import datachain as dc
from datachain import File
from datachain.sql.types import Int


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_merge_union(cloud_test_catalog, inner, cloud_type):
    catalog = cloud_test_catalog.catalog
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = dc.read_storage(f"{src}/dogs/", session=session)
    cats = dc.read_storage(f"{src}/cats/", session=session)

    signal_default_value = Int.default_value(catalog.warehouse.db.dialect)

    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    dogs2 = dogs.map(sig2=lambda: 2, output={"sig2": int})
    cats1 = cats.map(sig1=lambda: 1, output={"sig1": int})

    merged = (dogs1 | cats1).merge(dogs2, "file.path", inner=inner)
    signals = merged.select("file.path", "sig1", "sig2").order_by("file.path").results()

    if inner:
        assert signals == [
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]
    else:
        assert signals == [
            ("cats/cat1", 1, signal_default_value),
            ("cats/cat2", 1, signal_default_value),
            ("dogs/dog1", 1, 2),
            ("dogs/dog2", 1, 2),
            ("dogs/dog3", 1, 2),
            ("dogs/others/dog4", 1, 2),
        ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner1", [True, False])
@pytest.mark.parametrize("inner2", [True, False])
@pytest.mark.parametrize("inner3", [True, False])
def test_merge_multiple(cloud_test_catalog, inner1, inner2, inner3):
    catalog = cloud_test_catalog.catalog
    session = cloud_test_catalog.session

    src = cloud_test_catalog.src_uri

    dogs = dc.read_storage(f"{src}/dogs/", session=session)
    cats = dc.read_storage(f"{src}/cats/", session=session)

    signal_default_value = Int.default_value(catalog.warehouse.db.dialect)

    dogs_and_cats = dogs | cats
    dogs1 = dogs.map(sig1=lambda: 1, output={"sig1": int})
    cats1 = cats.map(sig2=lambda: 2, output={"sig2": int})
    dogs2 = dogs_and_cats.merge(dogs1, "file.path", inner=inner1)
    cats2 = dogs_and_cats.merge(cats1, "file.path", inner=inner2)
    merged = dogs2.merge(cats2, "file.path", inner=inner3)

    merged_signals = (
        merged.select("file.path", "sig1", "sig2").order_by("file.path").results()
    )

    if inner1 and inner2 and inner3:
        assert merged_signals == []
    elif inner1:
        assert merged_signals == [
            ("dogs/dog1", 1, signal_default_value),
            ("dogs/dog2", 1, signal_default_value),
            ("dogs/dog3", 1, signal_default_value),
            ("dogs/others/dog4", 1, signal_default_value),
        ]
    elif inner2 and inner3:
        assert merged_signals == [
            ("cats/cat1", signal_default_value, 2),
            ("cats/cat2", signal_default_value, 2),
        ]
    else:
        assert merged_signals == [
            ("cats/cat1", signal_default_value, 2),
            ("cats/cat2", signal_default_value, 2),
            ("dogs/dog1", 1, signal_default_value),
            ("dogs/dog2", 1, signal_default_value),
            ("dogs/dog3", 1, signal_default_value),
            ("dogs/others/dog4", 1, signal_default_value),
        ]


def test_full_outer_join_preserves_all_rows(test_session):
    """Test that full outer join correctly saves all rows including right-only rows.
    Checks also that sys columns are dropped on full outer join since it is not safe
    to carry them over.
    """
    # Create two datasets with no overlapping file paths
    ds1 = dc.read_values(
        id=[1, 2, 3],
        file=[File(path=str(i)) for i in [1, 2, 3]],
        session=test_session,
    ).persist()

    ds2 = dc.read_values(
        id=[5, 6, 7],
        file=[File(path=str(i)) for i in [5, 6, 7]],
        session=test_session,
    ).persist()

    merged = ds1.merge(ds2, on="file.path", full=True)

    chain_with_sys = merged.settings(sys=True)
    assert "sys__id" not in chain_with_sys.schema

    df = chain_with_sys.to_pandas()
    assert not any(str(col).startswith("sys__") for col in df.columns)

    merged.save("test_merge")
    count_after = dc.read_dataset("test_merge", session=test_session).count()

    assert count_after == 6
