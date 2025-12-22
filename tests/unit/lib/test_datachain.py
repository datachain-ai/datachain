import datetime
import json
import math
import os
import re
import uuid
from collections import Counter
from collections.abc import Generator, Iterator
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import Column, func
from datachain.error import (
    DatasetInvalidVersionError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    InvalidDatasetNameError,
    InvalidNamespaceNameError,
    InvalidProjectNameError,
    ProjectCreateNotAllowedError,
)
from datachain.lib.data_model import DataModel
from datachain.lib.dc import C, DatasetPrepareError, Sys
from datachain.lib.dc.listings import read_listing_dataset
from datachain.lib.file import File
from datachain.lib.listing import LISTING_PREFIX, parse_listing_uri
from datachain.lib.listing_info import ListingInfo
from datachain.lib.signal_schema import (
    SignalResolvingError,
    SignalResolvingTypeError,
    SignalSchema,
)
from datachain.lib.udf import UDFAdapter
from datachain.lib.udf_signature import UdfSignatureError
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.sql.types import Float, Int64, String
from datachain.utils import STUDIO_URL
from tests.utils import (
    ANY_VALUE,
    df_equal,
    skip_if_not_sqlite,
    sort_df,
    sorted_dicts,
)

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
}

DF_DATA_NESTED_NOT_NORMALIZED = {
    "nA-mE": [
        {"first-SELECT": "Ivan"},
        {"first-SELECT": "Alice", "l--as@t": "Smith"},
        {"l--as@t": "Jones", "first-SELECT": "Bob"},
        {"first-SELECT": "Charlie", "l--as@t": "Brown"},
        {"first-SELECT": "David", "l--as@t": "White"},
        {"first-SELECT": "Eva", "l--as@t": "Black"},
    ],
    "AgE": [41, 25, 30, 35, 40, 45],
    "citY": ["San Francisco", "New York", "Los Angeles", None, "Houston", "Phoenix"],
}

DF_OTHER_DATA = {
    "last_name": ["Smith", "Jones"],
    "country": ["USA", "Russia"],
}


class MyFr(BaseModel):
    nnn: str
    count: int


class MyNested(BaseModel):
    label: str
    fr: MyFr


features = sorted(
    [MyFr(nnn="n1", count=3), MyFr(nnn="n2", count=5), MyFr(nnn="n1", count=1)],
    key=lambda f: (f.nnn, f.count),
)

features_nested = [
    MyNested(fr=fr, label=f"label_{num}") for num, fr in enumerate(features)
]


def sort_files(files):
    return sorted(files, key=lambda f: (f.path, f.size))


def test_repr(test_session):
    chain = dc.read_values(
        sign1=features_nested, col1=["a", "b", "c"], session=test_session
    )
    assert (
        repr(chain)
        == """\
sign1: MyNested
  label: str
  fr: MyFr
    nnn: str
    count: int
col1: str
"""
    )

    # datachain without any columns
    assert repr(chain.select_except("col1", "sign1")) == "Empty DataChain"

    chain = chain.map(col2=lambda col1: col1 * 2)
    assert (
        repr(chain)
        == """\
sign1: MyNested
  label: str
  fr: MyFr
    nnn: str
    count: int
col1: str
col2: str
"""
    )

    chain = chain.mutate(countplusone=chain.column("sign1.fr.count") + 1)
    assert (
        repr(chain)
        == """\
sign1: MyNested
  label: str
  fr: MyFr
    nnn: str
    count: int
col1: str
col2: str
countplusone: int
"""
    )


def test_pandas_conversion(test_session):
    df = pd.DataFrame(DF_DATA)
    df1 = dc.read_pandas(df, session=test_session)
    df1 = df1.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_pandas_conversion_in_memory():
    df = pd.DataFrame(DF_DATA)
    chain = dc.read_pandas(df, in_memory=True)
    assert chain.session.catalog.in_memory is True
    assert chain.session.catalog.metastore.db.db_file == ":memory:"
    assert chain.session.catalog.warehouse.db.db_file == ":memory:"
    chain = chain.select("first_name", "age", "city").to_pandas()
    assert chain.equals(df)


def test_pandas_uppercase_columns(test_session):
    data = {
        "FirstName": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    }
    df = dc.read_pandas(pd.DataFrame(data), session=test_session).to_pandas()
    assert all(col not in df.columns for col in data)
    assert all(col.lower() in df.columns for col in data)


def test_pandas_incorrect_column_names(test_session):
    with pytest.raises(DataChainParamsError):
        dc.read_pandas(
            pd.DataFrame({"First Name": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )

    with pytest.raises(DataChainParamsError):
        dc.read_pandas(
            pd.DataFrame({"": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )

    with pytest.raises(DataChainParamsError):
        dc.read_pandas(
            pd.DataFrame({"First@Name": ["Alice", "Bob", "Charlie", "David", "Eva"]}),
            session=test_session,
        )


def test_from_features_basic(test_session):
    ds = dc.read_records(dc.DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(lambda prm: [File(path="")] * 5, params="path", output={"file": File})

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = dc.read_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"file"}
    assert set(ds.schema.values()) == {File}


@skip_if_not_sqlite
def test_from_features_basic_in_memory():
    ds = dc.read_records(dc.DataChain.DEFAULT_FILE_RECORD, in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    ds = ds.gen(lambda prm: [File(path="")] * 5, params="path", output={"file": File})

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = dc.read_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"file"}
    assert set(ds.schema.values()) == {File}


def test_from_features(test_session):
    ds = dc.read_records(dc.DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(
        lambda prm: list(zip([File(path="")] * len(features), features, strict=False)),
        params="path",
        output={"file": File, "t1": MyFr},
    )

    assert [r[1] for r in ds.order_by("t1.nnn", "t1.count").to_list()] == features


def test_read_record_empty_chain_with_schema(test_session):
    schema = {"my_file": File, "my_col": int}
    ds = dc.read_records([], schema=schema, session=test_session)

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = dc.read_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"my_file", "my_col"}
    assert set(ds.schema.values()) == {File, int}
    assert ds.count() == 0

    # check that columns have actually been created from schema
    catalog = test_session.catalog
    dr = catalog.warehouse.dataset_rows(catalog.get_dataset(ds_name))
    assert sorted([c.name for c in dr.columns]) == sorted(
        ds.signals_schema.db_signals()
    )


def test_read_record_empty_chain_without_schema(test_session):
    ds = dc.read_records([], schema=None, session=test_session)

    ds_name = "my_ds"
    ds.save(ds_name)
    ds = dc.read_dataset(name=ds_name)

    assert ds.schema.keys() == {
        "source",
        "path",
        "size",
        "version",
        "etag",
        "is_latest",
        "last_modified",
        "location",
    }
    assert ds.count() == 0

    # check that columns have actually been created from schema
    catalog = test_session.catalog
    dr = catalog.warehouse.dataset_rows(catalog.get_dataset(ds_name))
    assert sorted([c.name for c in dr.columns]) == sorted(
        ds.signals_schema.db_signals()
    )


def test_empty(test_session):
    assert dc.read_records([], schema=None, session=test_session).empty is True


def test_empty_chain_skip_udf_run(test_session):
    # Test that UDF is not called for empty chain
    with patch.object(UDFAdapter, "run") as mock_udf_run:
        (
            dc.read_records([], schema={"val": int}, session=test_session)
            .map(lambda val: val * 2, params="val", output={"val2": int})
            .exec()
        )
        mock_udf_run.assert_not_called()


def test_datasets(test_session):
    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save("fibonacci")

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    ds = dc.datasets(column="foo", session=test_session)
    datasets = [d for d in ds.to_values("foo") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6


def test_datasets_without_column_name(test_session):
    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save("fibonacci")
    ds = dc.datasets(session=test_session)
    names = [name for name in ds.to_values("name") if name == "fibonacci"]
    assert len(names) == 1


def test_datasets_studio(studio_datasets, test_session):
    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save("fibonacci")
    ds = dc.datasets(column="dataset", studio=True, session=test_session)
    # Local datasets are not included in the list
    datasets = [d for d in ds.to_values("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    # Studio datasets are included in the list
    datasets = [d for d in ds.to_values("dataset") if d.name == "cats"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    # Exclude studio datasets
    ds = dc.datasets(column="dataset", studio=False, session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == "cats"]
    assert len(datasets) == 0


@skip_if_not_sqlite
def test_datasets_in_memory():
    ds = dc.datasets(column="dataset", in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.to_values("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 0

    dc.read_values(fib=[1, 1, 2, 3, 5, 8]).save("fibonacci")

    ds = dc.datasets(column="dataset")
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.to_values("dataset") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6

    ds = dc.datasets(column="foo")
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"
    datasets = [d for d in ds.to_values("foo") if d.name == "fibonacci"]
    assert len(datasets) == 1
    assert datasets[0].num_objects == 6


@pytest.mark.parametrize(
    "attrs,result",
    [
        (["number"], ["evens", "primes"]),
        (["num=prime"], ["primes"]),
        (["num=even"], ["evens"]),
        (["num=*"], ["evens", "primes"]),
        (["num=*", "small"], ["primes"]),
        (["letter"], ["letters"]),
        (["missing"], []),
        (["num=*", "missing"], []),
        (None, ["evens", "letters", "primes"]),
        ([], ["evens", "letters", "primes"]),
    ],
)
def test_datasets_filtering(test_session, attrs, result):
    dc.read_values(num=[1, 2, 3], session=test_session).save(
        "primes", attrs=["number", "num=prime", "small"]
    )

    dc.read_values(num=[2, 4, 6], session=test_session).save(
        "evens", attrs=["number", "num=even"]
    )

    dc.read_values(letter=["a", "b", "c"], session=test_session).save(
        "letters", attrs=["letter"]
    )

    assert sorted(
        dc.datasets(attrs=attrs, session=test_session).to_values("name")
    ) == sorted(result)


def test_listings(test_session, tmp_dir):
    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")

    uri = tmp_dir.as_uri()
    dc.read_storage(uri, session=test_session).exec()

    # check that listing is not returned as normal dataset
    assert not any(
        n.startswith(LISTING_PREFIX)
        for n in dc.datasets(session=test_session).to_values("name")
    )

    listings = list(dc.listings(session=test_session).to_values("listing"))
    assert len(listings) == 1
    listing = listings[0]
    assert isinstance(listing, ListingInfo)
    assert listing.storage_uri == uri
    assert listing.is_expired is False
    assert listing.expires
    assert listing.version == "1.0.0"
    assert listing.num_objects == 1
    # Exact number if unreliable here since it depends on the PyArrow version
    assert listing.size > 1000
    assert listing.size < 5000
    assert listing.status == 4


def test_listings_reindex(test_session, tmp_dir):
    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")

    uri = tmp_dir.as_uri()

    dc.read_storage(uri, session=test_session).exec()
    assert len(list(dc.listings(session=test_session).to_values("listing"))) == 1

    dc.read_storage(uri, session=test_session).exec()
    assert len(list(dc.listings(session=test_session).to_values("listing"))) == 1

    dc.read_storage(uri, session=test_session, update=True).exec()
    listings = list(dc.listings(session=test_session).to_values("listing"))
    assert len(listings) == 2
    listings.sort(key=lambda lst: lst.version)
    assert listings[0].storage_uri == uri
    assert listings[0].version == "1.0.0"
    assert listings[1].storage_uri == uri
    assert listings[1].version == "2.0.0"


def test_listings_reindex_subpath_local_file_system(test_session, tmp_dir):
    subdir = tmp_dir / "subdir"
    os.mkdir(subdir)

    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")
    df.to_parquet(tmp_dir / "df2.parquet")
    df.to_parquet(subdir / "df3.parquet")

    assert dc.read_storage(tmp_dir.as_uri(), session=test_session).count() == 3
    assert dc.read_storage(subdir.as_uri(), session=test_session).count() == 1


def test_read_storage_plain_local_path_keeps_root(test_session, tmp_dir, monkeypatch):
    monkeypatch.chdir(tmp_dir)

    nested = tmp_dir / "nested"
    nested.mkdir()
    (nested / "file.txt").write_text("data")

    files = dc.read_storage(tmp_dir, session=test_session).to_values("file")

    assert len(files) == 1
    assert files[0].source == tmp_dir.as_uri()
    assert files[0].path == "nested/file.txt"


@pytest.mark.parametrize(
    "uri_factory",
    [
        lambda p: p,
        lambda p: p.as_uri(),
        lambda p: p.as_uri() + "/",
        lambda p: f"{p}{os.sep}",
    ],
)
def test_read_storage_local_variants_consistent(test_session, tmp_dir, uri_factory):
    nested = tmp_dir / "dir" / "file.bin"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"x")

    uri = uri_factory(tmp_dir)
    files = dc.read_storage(uri, session=test_session).to_values("file")

    assert len(files) == 1
    assert files[0].source == tmp_dir.as_uri()
    assert files[0].path == "dir/file.bin"


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_listings_read_listing_dataset(test_session, tmp_dir, version):
    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")
    uri = tmp_dir.as_uri()

    ds_name, _, _ = parse_listing_uri(uri)
    dc.read_storage(uri, session=test_session).exec()

    chain, listing_version = read_listing_dataset(
        ds_name, version=version, session=test_session
    )
    assert listing_version.num_objects == 1
    assert listing_version.size > 1000
    assert listing_version.size < 5000
    assert listing_version.status == 4

    assert chain.count() == 1
    files = chain.to_values("file")
    assert len(files) == 1
    assert files[0].path == "df.parquet"
    assert files[0].source == uri


def test_listings_read_listing_dataset_with_subpath(test_session, tmp_dir):
    subdir = tmp_dir / "subdir"
    os.mkdir(subdir)

    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")
    df.to_parquet(tmp_dir / "df2.parquet")
    df.to_parquet(subdir / "df3.parquet")

    ds_name, _, _ = parse_listing_uri(tmp_dir.as_uri())
    ds_name = ds_name.removeprefix(LISTING_PREFIX)
    dc.read_storage(tmp_dir.as_uri(), session=test_session).exec()

    chain, listing_version = read_listing_dataset(
        ds_name, path="subdir", session=test_session
    )
    assert listing_version.num_objects == 3

    # Chain is filtered for subdir
    assert chain.count() == 1
    files = chain.to_values("file")
    assert len(files) == 1
    assert files[0].path == "subdir/df3.parquet"
    assert files[0].source == tmp_dir.as_uri()


def test_preserve_feature_schema(test_session):
    ds = dc.read_records(dc.DataChain.DEFAULT_FILE_RECORD, session=test_session)
    ds = ds.gen(
        lambda prm: list(
            zip([File(path="")] * len(features), features, features, strict=False)
        ),
        params="path",
        output={"file": File, "t1": MyFr, "t2": MyFr},
    )

    ds_name = "my_ds1"
    ds.save(ds_name)
    ds = dc.read_dataset(name=ds_name)

    assert isinstance(ds._query.feature_schema, dict)
    assert isinstance(ds.signals_schema, SignalSchema)
    assert ds.schema.keys() == {"t1", "t2", "file"}
    assert set(ds.schema.values()) == {MyFr, File}


def test_from_features_simple_types(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    ds = dc.read_values(fib=fib, odds=values, session=test_session)

    df = sort_df(ds.to_pandas())
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


@skip_if_not_sqlite
def test_from_features_simple_types_in_memory():
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    ds = dc.read_values(fib=fib, odds=values, in_memory=True)
    assert ds.session.catalog.in_memory is True
    assert ds.session.catalog.metastore.db.db_file == ":memory:"
    assert ds.session.catalog.warehouse.db.db_file == ":memory:"

    df = ds.to_pandas()
    assert len(df) == len(fib)
    assert df["fib"].tolist() == fib
    assert df["odds"].tolist() == values


@skip_if_not_sqlite
def test_to_pandas_as_object_preserves_none(test_session):
    timestamp = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    chain = dc.read_values(
        id=[1, None],
        value=[3.14, None],
        ts=[timestamp, None],
        session=test_session,
    )

    df_default = chain.to_pandas()
    assert df_default["id"].dtype != object
    assert df_default["value"].dtype != object
    assert df_default["id"].isna().sum() == 1
    assert df_default["value"].isna().sum() == 1
    assert pd.isna(df_default.loc[df_default["id"].isna(), "ts"]).all()

    df_object = chain.to_pandas(as_object=True)
    assert df_object["id"].dtype == object
    assert df_object["value"].dtype == object
    assert df_object["ts"].dtype == object
    assert Counter(df_object["id"].tolist()) == Counter([1, None])
    assert Counter(df_object["value"].tolist()) == Counter([3.14, None])
    assert Counter(df_object["ts"].tolist()) == Counter([timestamp, None])


def test_from_features_more_simple_types(test_session):
    ds = dc.read_values(
        t1=features,
        num=range(len(features)),
        bb=[True, True, False],
        dd=[{}, {"ee": 3}, {"ww": 1, "qq": 2}],
        ll1=[[], [1, 2, 3], [3, 4, 5]],
        ll=[[None, 1, 2], [3, 4], []],
        dd2=[{"a": 1}, {"b": 2}, {"c": 3}],
        nn=[None, None, None],
        ss=["x", None, "y"],
        time=[
            datetime.datetime.now(),
            datetime.datetime.today(),
            datetime.datetime.today(),
        ],
        f=[3.14, 2.72, 1.62],
        session=test_session,
    )

    assert ds.schema.keys() == {
        "t1",
        "num",
        "bb",
        "dd",
        "ll1",
        "ll",
        "dd2",
        "nn",
        "ss",
        "time",
        "f",
    }
    assert set(ds.schema.values()) == {
        MyFr,
        int,
        bool,
        dict[str, str],  # from dd (starts with empty dict)
        dict[str, int],  # from dd2
        list[str],  # from ll1 (starts with empty list)
        list[int],  # from ll
        str,  # from nn and ss
        datetime.datetime,
        float,
    }


def test_file_list(test_session):
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = [
        File(path=name, size=size) for name, size in zip(names, sizes, strict=False)
    ]

    ds = dc.read_values(file=files, session=test_session)

    assert sort_files(files) == [
        r[0] for r in ds.order_by("file.path", "file.size").to_list()
    ]


def test_gen(test_session):
    class _TestFr(BaseModel):
        file: File
        sqrt: float
        my_name: str

    ds = dc.read_values(t1=features, session=test_session)
    ds = ds.gen(
        x=lambda m_fr: [
            _TestFr(
                file=File(path=""),
                sqrt=math.sqrt(m_fr.count),
                my_name=m_fr.nnn,
            )
        ],
        params="t1",
        output={"x": _TestFr},
    )

    for i, (x,) in enumerate(ds.order_by("x.my_name", "x.sqrt").to_list()):
        assert isinstance(x, _TestFr)

        fr = features[i]
        test_fr = _TestFr(file=File(path=""), sqrt=math.sqrt(fr.count), my_name=fr.nnn)
        assert x.file == test_fr.file
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_map(test_session):
    class _TestFr(BaseModel):
        sqrt: float
        my_name: str

    chain = dc.read_values(t1=features, session=test_session).map(
        x=lambda m_fr: _TestFr(
            sqrt=math.sqrt(m_fr.count),
            my_name=m_fr.nnn + "_suf",
        ),
        params="t1",
        output={"x": _TestFr},
    )

    x_list = chain.order_by("x.my_name", "x.sqrt").to_values("x")
    test_frs = [
        _TestFr(sqrt=math.sqrt(fr.count), my_name=fr.nnn + "_suf") for fr in features
    ]

    assert len(x_list) == len(test_frs)

    for x, test_fr in zip(x_list, test_frs, strict=False):
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


def test_map_existing_column_after_step(test_session):
    chain = dc.read_values(t1=features, session=test_session).map(
        x=lambda _: "test",
        params="t1",
        output={"x": str},
    )

    with pytest.raises(ValueError):
        chain.map(
            x=lambda _: "test",
            params="t1",
            output={"x": str},
        ).exec()


def test_agg(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    chain = dc.read_values(t1=features, session=test_session).agg(
        x=lambda frs: [
            _TestFr(
                f=File(path=""),
                cnt=sum(f.count for f in frs),
                my_name="-".join([fr.nnn for fr in frs]),
            )
        ],
        partition_by=C.t1.nnn,
        params="t1",
        output={"x": _TestFr},
    )

    assert chain.order_by("x.my_name").to_values("x") == [
        _TestFr(
            f=File(path=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n1"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n1"]),
        ),
        _TestFr(
            f=File(path=""),
            cnt=sum(fr.count for fr in features if fr.nnn == "n2"),
            my_name="-".join([fr.nnn for fr in features if fr.nnn == "n2"]),
        ),
    ]


def test_agg_two_params(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    ds = (
        dc.read_values(t1=features, t2=features2, session=test_session)
        .order_by("t1.nnn", "t2.nnn")
        .agg(
            x=lambda frs1, frs2: [
                _TestFr(
                    f=File(path=""),
                    cnt=sum(
                        f1.count + f2.count for f1, f2 in zip(frs1, frs2, strict=False)
                    ),
                    my_name="-".join([fr.nnn for fr in frs1]),
                )
            ],
            partition_by=C.t1.nnn,
            params=("t1", "t2"),
            output={"x": _TestFr},
        )
    )

    assert ds.order_by("x.my_name").to_values("x.my_name") == ["n1-n1", "n2"]
    assert ds.order_by("x.cnt").to_values("x.cnt") == [7, 20]


def test_agg_simple_iterator(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c"]
    values = [3, 1, 2]
    ds = dc.read_values(key=keys, val=values, session=test_session).gen(res=func)

    df = sort_df(ds.to_pandas())
    res = df["res_1"].tolist()
    assert res == ["a_0", "a_1", "a_2", "b_0", "c_0", "c_1"]


def test_agg_simple_iterator_error(test_session):
    chain = dc.read_values(key=["a", "b", "c"], session=test_session)

    with pytest.raises(UdfSignatureError):

        def func(key) -> int:
            return 1

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        class _MyCls(BaseModel):
            x: int

        def func(key) -> _MyCls:  # type: ignore[misc]
            return _MyCls(x=2)

        chain.gen(res=func)

    with pytest.raises(UdfSignatureError):

        def func(key) -> tuple[File, str]:  # type: ignore[misc]
            yield None, "qq"

        chain.gen(res=func)


def test_agg_tuple_result_iterator(test_session):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = "-".join(key)
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = dc.read_values(key=keys, val=values, session=test_session).agg(
        x=func, partition_by=C("key")
    )

    assert ds.order_by("x_1.name").to_values("x_1.name") == ["n1-n1", "n2"]
    assert ds.order_by("x_1.size").to_values("x_1.size") == [5, 10]


def test_agg_tuple_result_generator(test_session):
    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Generator[tuple[File, _ImageGroup], None, None]:
        n = "-".join(key)
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]
    ds = (
        dc.read_values(key=keys, val=values, session=test_session)
        .agg(x=func, partition_by=C("key"))
        .order_by("x_1.name")
    )

    assert ds.order_by("x_1.name").to_values("x_1.name") == ["n1-n1", "n2"]
    assert ds.order_by("x_1.size").to_values("x_1.size") == [5, 10]


_batch_mapper_warn = pytest.mark.filterwarnings("ignore:BatchMapper is deprecated.*")


@_batch_mapper_warn
def test_batch_map(test_session):
    class _TestFr(BaseModel):
        sqrt: float
        my_name: str

    with pytest.deprecated_call(match=r"batch_map\(\) is deprecated"):
        chain = dc.read_values(t1=features, session=test_session).batch_map(
            x=lambda m_frs: [
                _TestFr(
                    sqrt=math.sqrt(m_fr.count),
                    my_name=m_fr.nnn + "_suf",
                )
                for m_fr in m_frs
            ],
            params="t1",
            output={"x": _TestFr},
        )

    x_list = chain.order_by("x.my_name", "x.sqrt").to_values("x")
    test_frs = [
        _TestFr(sqrt=math.sqrt(fr.count), my_name=fr.nnn + "_suf") for fr in features
    ]

    assert len(x_list) == len(test_frs)

    for x, test_fr in zip(x_list, test_frs, strict=False):
        assert np.isclose(x.sqrt, test_fr.sqrt)
        assert x.my_name == test_fr.my_name


@_batch_mapper_warn
def test_batch_map_wrong_size(test_session):
    class _TestFr(BaseModel):
        total: int
        names: str

    with pytest.deprecated_call(match=r"batch_map\(\) is deprecated"):
        chain = dc.read_values(t1=features, session=test_session).batch_map(
            x=lambda m_frs: [
                _TestFr(
                    total=sum(m_fr.count for m_fr in m_frs),
                    names="-".join([m_fr.nnn for m_fr in m_frs]),
                )
            ],
            params="t1",
            output={"x": _TestFr},
        )

    with pytest.raises(AssertionError):
        chain.to_list()


@_batch_mapper_warn
def test_batch_map_two_params(test_session):
    class _TestFr(BaseModel):
        f: File
        cnt: int
        my_name: str

    features2 = [
        MyFr(nnn="n1", count=6),
        MyFr(nnn="n2", count=10),
        MyFr(nnn="n1", count=2),
    ]

    with pytest.deprecated_call(match=r"batch_map\(\) is deprecated"):
        ds = dc.read_values(t1=features, t2=features2, session=test_session).batch_map(
            x=lambda frs1, frs2: [
                _TestFr(
                    f=File(path=""),
                    cnt=f1.count + f2.count,
                    my_name=f"{f1.nnn}-{f2.nnn}",
                )
                for f1, f2 in zip(frs1, frs2, strict=False)
            ],
            params=("t1", "t2"),
            output={"x": _TestFr},
        )

    assert ds.order_by("x.my_name").to_values("x.my_name") == [
        "n1-n1",
        "n1-n2",
        "n2-n1",
    ]
    assert ds.order_by("x.cnt").to_values("x.cnt") == [7, 7, 13]


@_batch_mapper_warn
def test_batch_map_tuple_result_iterator(test_session):
    def sqrt(t1: list[int]) -> Iterator[float]:
        for val in t1:
            yield math.sqrt(val)

    with pytest.deprecated_call(match=r"batch_map\(\) is deprecated"):
        chain = dc.read_values(t1=[1, 4, 9], session=test_session).batch_map(x=sqrt)

    assert chain.order_by("x").to_values("x") == [1, 2, 3]


def test_read_dataset_name_version(test_session):
    name = "test-version"
    dc.read_values(
        first_name=["Alice", "Bob", "Charlie"],
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).save(name)

    chain = dc.read_dataset(name)
    assert chain.name == name
    assert chain.version


def test_chain_of_maps(test_session):
    chain = (
        dc.read_values(my_n=features_nested, session=test_session)
        .map(full_name=lambda my_n: my_n.label + "-" + my_n.fr.nnn, output=str)
        .map(square=lambda my_n: my_n.fr.count**2, output=int)
    )

    signals = ["my_n", "full_name", "square"]
    assert len(chain.schema) == len(signals)
    for signal in signals:
        assert signal in chain.schema

    preserved = chain.persist()
    for signal in signals:
        assert signal in preserved.schema


def test_vector(test_session):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[float]:
        return vector

    ds = dc.read_values(key=[123], session=test_session).map(emd=get_vector)

    df = ds.to_pandas()
    assert np.allclose(df["emd"].tolist()[0], vector)


def test_vector_of_vectors(test_session):
    vector = [[3.14, 2.72, 1.62], [1.0, 2.0, 3.0]]

    def get_vector(key) -> list[list[float]]:
        return vector

    ds = dc.read_values(key=[123], session=test_session).map(emd_list=get_vector)

    df = ds.to_pandas()
    actual = df["emd_list"].tolist()[0]
    assert len(actual) == 2
    assert np.allclose(actual[0], vector[0])
    assert np.allclose(actual[1], vector[1])


def test_persist_restores_sys_signals_after_merge(test_session):
    left = dc.read_values(ids=[1, 2], session=test_session)
    right = dc.read_values(ids=[1, 2], extra=["x", "y"], session=test_session)

    merged = left.merge(right, on="ids")

    with pytest.raises(SignalResolvingError):
        merged.signals_schema.resolve("sys.rand")

    persisted = merged.persist()

    sys_schema = persisted.signals_schema.resolve("sys.id", "sys.rand").values
    assert sys_schema["sys.id"] is int
    assert sys_schema["sys.rand"] is int


def test_shuffle_after_merge(test_session):
    left = dc.read_values(ids=[1, 2], session=test_session)
    right = dc.read_values(ids=[1, 2], extra=["x", "y"], session=test_session)

    shuffled = left.merge(right, on="ids").shuffle()

    sys_schema = shuffled.signals_schema.resolve("sys.id", "sys.rand").values
    assert sys_schema["sys.id"] is int
    assert sys_schema["sys.rand"] is int

    rows = set(shuffled.to_list("ids", "extra"))
    assert rows == {(1, "x"), (2, "y")}


def test_unsupported_output_type(test_session):
    vector = [3.14, 2.72, 1.62]

    def get_vector(key) -> list[np.float64]:
        return [vector]

    with pytest.raises(TypeError):
        dc.read_values(key=[123], session=test_session).map(emd=get_vector)


def test_default_output_type(test_session):
    names = sorted(["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"])
    suffix = "-new"

    chain = dc.read_values(name=names, session=test_session).map(
        res1=lambda name: name + suffix
    )

    assert chain.order_by("name").to_values("res1") == [t + suffix for t in names]


def test_parse_tabular(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular()
    df1 = chain.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_parse_tabular_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_storage(path.as_uri(), in_memory=True).parse_tabular()
    assert chain.session.catalog.in_memory is True
    assert chain.session.catalog.metastore.db.db_file == ":memory:"
    assert chain.session.catalog.warehouse.db.db_file == ":memory:"
    df1 = chain.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_parse_tabular_format(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json"
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_parse_nested_json(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA_NESTED_NOT_NORMALIZED)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json"
    )
    # Field names are normalized, values are preserved
    # E.g. nAmE -> name, l--as@t -> l_as_t, etc
    df1 = chain.select("na_me", "age", "city").to_pandas()

    # In CH we replace None with '' for peforance reasons,
    # have to handle it here
    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    assert sorted(df1["na_me"]["first_select"].to_list()) == sorted(
        d["first-SELECT"] for d in df["nA-mE"].to_list()
    )
    assert sorted(
        df1["na_me"]["l_as_t"].to_list(), key=lambda x: (x is None, x)
    ) == sorted(
        [d.get("l--as@t", string_default) for d in df["nA-mE"].to_list()],
        key=lambda x: (x is None, x),
    )


def test_parse_tabular_partitions(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    chain = (
        dc.read_storage(path.as_uri(), session=test_session)
        .filter(C("file.path").glob("*first_name=Alice*"))
        .parse_tabular(partitioning="hive")
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df_equal(df1, df.loc[:0])


def test_parse_tabular_no_files(test_session):
    chain = dc.read_values(
        f1=features, num=range(len(features)), session=test_session, in_memory=True
    )
    with pytest.raises(DatasetPrepareError):
        chain.parse_tabular()

    schema = {"file": File, "my_col": int}
    chain = dc.read_records([], schema=schema, session=test_session, in_memory=True)

    with pytest.raises(DatasetPrepareError):
        chain.parse_tabular()


def test_parse_tabular_unify_schema(tmp_dir, test_session):
    df1 = pd.DataFrame(DF_DATA)
    df2 = pd.DataFrame(DF_OTHER_DATA)
    path1 = tmp_dir / "df1.parquet"
    path2 = tmp_dir / "df2.parquet"
    df1.to_parquet(path1)
    df2.to_parquet(path2)

    df_combined = (
        pd.concat([df1, df2], ignore_index=True)
        .replace({"": None, 0: None, np.nan: None})
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    chain = (
        dc.read_storage(tmp_dir.as_uri(), session=test_session)
        .filter(C("file.path").glob("*.parquet"))
        .parse_tabular()
    )
    df = chain.select("first_name", "age", "city", "last_name", "country").to_pandas()
    df = (
        df.replace({"": None, 0: None, np.nan: None})
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    assert df_equal(df, df_combined)


def test_parse_tabular_output_dict(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = {"fname": str, "age": int, "loc": str}
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=output
    )
    df1 = chain.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_output_feature(tmp_dir, test_session):
    class Output(BaseModel):
        fname: str
        age: int
        loc: str

    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=Output
    )
    df1 = chain.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_output_list(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.jsonl"
    path.write_text(df.to_json(orient="records", lines=True))
    output = ["fname", "age", "loc"]
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        format="json", output=output
    )
    df1 = chain.select("fname", "age", "loc").to_pandas()
    df.columns = ["fname", "age", "loc"]
    assert df_equal(df1, df)


def test_parse_tabular_nrows(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_json(path, orient="records", lines=True)
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        nrows=2, format="json"
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df[:2])


def test_parse_tabular_nrows_invalid(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    with pytest.raises(DataChainParamsError):
        dc.read_storage(path.as_uri(), session=test_session).parse_tabular(nrows=2)


def test_read_csv(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(path.as_uri(), session=test_session)
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_to_read_csv(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.to_csv(path)
    dc_from = dc.read_csv(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_read_csv_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(path.as_uri(), in_memory=True)
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_to_read_csv_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, in_memory=True)
    path = tmp_dir / "test.csv"
    dc_to.to_csv(path)
    dc_from = dc.read_csv(path.as_uri(), in_memory=True)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


def test_read_csv_no_header_error(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    with pytest.raises(DataChainParamsError):
        dc.read_csv(path.as_uri(), header=False, session=test_session)


def test_read_csv_no_header_output_dict(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    chain = dc.read_csv(
        path.as_uri(),
        header=False,
        output={"first_name": str, "age": int, "city": str},
        session=test_session,
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_read_csv_no_header_output_feature(tmp_dir, test_session):
    class Output(BaseModel):
        first_name: str
        age: int
        city: str

    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    chain = dc.read_csv(
        path.as_uri(), header=False, output=Output, session=test_session
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_read_csv_no_header_output_list(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA.values()).transpose()
    path = tmp_dir / "test.csv"
    df.to_csv(path, header=False, index=False)
    chain = dc.read_csv(
        path.as_uri(),
        header=False,
        output=["first_name", "age", "city"],
        session=test_session,
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert (sort_df(df1).values != sort_df(df).values).sum() == 0


def test_read_csv_tab_delimited(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, sep="\t", index=False)
    chain = dc.read_csv(path.as_uri(), delimiter="\t", session=test_session)
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_read_csv_null_collect(tmp_dir, test_session):
    # Clickhouse requires setting type to Nullable(Type).
    # See https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/189.
    df = pd.DataFrame(DF_DATA)
    height = [70, 65, None, 72, 68]
    gender = ["f", "m", None, "m", "f"]
    df["height"] = height
    df["gender"] = gender
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(path.as_uri(), column="csv", session=test_session)
    for i, row in enumerate(chain.to_list()):
        # None value in numeric column will get converted to nan.
        if not height[i]:
            assert math.isnan(row[1].height)
        else:
            assert row[1].height == height[i]
        assert row[1].gender == gender[i]


def test_read_csv_nrows(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(path.as_uri(), nrows=2, session=test_session)
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df[:2])


def test_read_csv_column_types(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(
        path.as_uri(), column_types={"age": "str"}, session=test_session
    )
    df1 = chain.select("first_name", "age", "city").to_pandas()
    assert df1["age"].dtype == pd.StringDtype


def test_read_csv_parse_options(tmp_dir, test_session):
    def skip_comment(row):
        if row.text.startswith("# "):
            return "skip"
        return "error"

    s = (
        "animals;n_legs;entry\n"
        "Flamingo;2;2022-03-01\n"
        "# Comment here:\n"
        "Horse;4;2022-03-02\n"
        "Brittle stars;5;2022-03-03\n"
        "Centipede;100;2022-03-04"
    )

    path = tmp_dir / "test.csv"
    path.write_text(s)

    chain = dc.read_csv(
        path.as_uri(),
        session=test_session,
        parse_options={"invalid_row_handler": skip_comment, "delimiter": ";"},
    )

    df = chain.select("animals", "n_legs", "entry").to_pandas()
    assert set(df["animals"]) == {"Horse", "Centipede", "Brittle stars", "Flamingo"}


@pytest.mark.parametrize("column_type", (str, dict))
@pytest.mark.parametrize("column", (None, "test_column"))
@pytest.mark.parametrize("model_name", (None, "TestModelNameExploded"))
def test_explode(tmp_dir, test_session, column_type, column, model_name):
    df = pd.DataFrame(DF_DATA_NESTED_NOT_NORMALIZED)
    path = tmp_dir / "test.json"
    df.to_json(path, orient="records", lines=True)

    chain = (
        dc.read_storage(path.as_uri(), session=test_session)
        .gen(
            content=lambda file: (ln for ln in file.read_text().split("\n") if ln),
            output=column_type,
        )
        .explode(
            "content",
            column=column,
            model_name=model_name,
            schema_sample_size=2,
        )
    )

    column = column or "content_expl"
    model_name = model_name or "ContentExplodedModel"

    # In CH we have (atm at least) None converted to ''
    # for performance reasons, so we need to handle this case
    string_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    assert set(
        chain.to_list(
            f"{column}.na_me.first_select",
            f"{column}.age",
            f"{column}.city",
        )
    ) == {
        ("Alice", 25, "New York"),
        ("Bob", 30, "Los Angeles"),
        ("Charlie", 35, string_default),
        ("David", 40, "Houston"),
        ("Eva", 45, "Phoenix"),
        ("Ivan", 41, "San Francisco"),
    }

    assert chain.limit(1).to_values(column)[0].__class__.__name__ == model_name


def test_explode_raises_on_wrong_column_type(test_session):
    chain = dc.read_values(f1=features, session=test_session)

    with pytest.raises(TypeError):
        chain.explode("f1.count")


def test_to_json_features(tmp_dir, test_session):
    dc_to = dc.read_values(f1=features, num=range(len(features)), session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("f1.nnn", "f1.count").to_json(path)
    with open(path) as f:
        values = json.load(f)
    assert values == [
        {"f1": {"nnn": f.nnn, "count": f.count}, "num": n}
        for n, f in enumerate(features)
    ]


def test_to_json_features_nested(tmp_dir, test_session):
    dc_to = dc.read_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_json(path)
    with open(path) as f:
        values = json.load(f)
    assert values == [
        {"sign1": {"label": f"label_{n}", "fr": {"nnn": f.nnn, "count": f.count}}}
        for n, f in enumerate(features)
    ]


def test_to_json_datetime(tmp_dir, test_session):
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    chain = dc.read_values(ts=[timestamp], session=test_session)

    path = tmp_dir / "dt.json"
    chain.to_json(path)

    with path.open() as fh:
        values = json.load(fh)

    expected_ts = timestamp.isoformat().replace("+00:00", "Z")
    assert values == [{"ts": expected_ts}]
    assert (
        datetime.datetime.fromisoformat(values[0]["ts"].replace("Z", "+00:00"))
        == timestamp
    )


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_to_read_jsonl(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=test_session)
    path = tmp_dir / "test.jsonl"
    dc_to.order_by("first_name", "age").to_jsonl(path)

    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(
            DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"], strict=False
        )
    ]

    dc_from = dc.read_json(path.as_uri(), format="jsonl", session=test_session)
    df1 = dc_from.select("jsonl.first_name", "jsonl.age", "jsonl.city").to_pandas()
    df1 = df1["jsonl"]
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_read_jsonl_jmespath(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    values = [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(
            DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"], strict=False
        )
    ]
    path = tmp_dir / "test.jsonl"
    with open(path, "w") as f:
        for v in values:
            f.write(
                json.dumps({"data": "Contained Within", "row_version": 5, "value": v})
            )
            f.write("\n")

    dc_from = dc.read_json(
        path.as_uri(), format="jsonl", jmespath="value", session=test_session
    )
    df1 = dc_from.select("value.first_name", "value.age", "value.city").to_pandas()
    df1 = df1["value"]
    assert df_equal(df1, df)


def test_to_jsonl_features(tmp_dir, test_session):
    dc_to = dc.read_values(f1=features, num=range(len(features)), session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("f1.nnn", "f1.count").to_jsonl(path)
    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"f1": {"nnn": f.nnn, "count": f.count}, "num": n}
        for n, f in enumerate(features)
    ]


def test_to_jsonl_features_nested(tmp_dir, test_session):
    dc_to = dc.read_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_jsonl(path)
    with open(path) as f:
        values = [json.loads(line) for line in f.read().split("\n")]
    assert values == [
        {"sign1": {"label": f"label_{n}", "fr": {"nnn": f.nnn, "count": f.count}}}
        for n, f in enumerate(features)
    ]


def test_read_parquet(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    df1 = chain.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_read_parquet_exported_with_source(test_session, tmp_dir):
    path = tmp_dir / "df.parquet"
    path2 = tmp_dir / "df2.parquet"
    df = pd.DataFrame(DF_DATA)

    df.to_parquet(path)
    dc.read_parquet(path, source=True).to_parquet(path2)
    df1 = (
        dc.read_parquet(path2, source=True)
        .select("first_name", "age", "city")
        .to_pandas()
    )

    assert df_equal(df1, df)


@skip_if_not_sqlite
def test_read_parquet_in_memory(tmp_dir):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_parquet(path.as_uri(), in_memory=True)
    df1 = chain.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_read_parquet_partitioned(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path, partition_cols=["first_name"])
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    df1 = chain.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df_equal(df1, df)


def test_to_parquet(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    chain = dc.read_pandas(df, session=test_session)

    path = tmp_dir / "test.parquet"
    chain.to_parquet(path)

    assert path.is_file()
    pd.testing.assert_frame_equal(sort_df(pd.read_parquet(path)), sort_df(df))


def test_to_parquet_partitioned(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    chain = dc.read_pandas(df, session=test_session)

    path = tmp_dir / "parquets"
    chain.to_parquet(path, partition_cols=["first_name"])

    assert set(path.iterdir()) == {
        path / f"first_name={name}" for name in df["first_name"]
    }
    df1 = pd.read_parquet(path)
    df1 = df1.reindex(columns=df.columns)
    df1["first_name"] = df1["first_name"].astype("str")
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
@pytest.mark.parametrize("kwargs", ({}, {"compression": "gzip"}))
def test_to_read_parquet(tmp_dir, test_session, chunk_size, kwargs):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size, **kwargs)

    assert path.is_file()
    pd.testing.assert_frame_equal(sort_df(pd.read_parquet(path)), sort_df(df))

    dc_from = dc.read_parquet(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_read_parquet_partitioned(tmp_dir, test_session, chunk_size):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=test_session)

    path = tmp_dir / "parquets"
    dc_to.to_parquet(path, partition_cols=["first_name"], chunk_size=chunk_size)

    assert set(path.iterdir()) == {
        path / f"first_name={name}" for name in df["first_name"]
    }
    df1 = pd.read_parquet(path)
    df1 = df1.reindex(columns=df.columns)
    df1["first_name"] = df1["first_name"].astype("str")
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df)

    dc_from = dc.read_parquet(path.as_uri(), session=test_session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df1.equals(df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_read_parquet_features(tmp_dir, test_session, chunk_size):
    dc_to = dc.read_values(f1=features, num=range(len(features)), session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = dc.read_parquet(path.as_uri(), session=test_session)

    n = 0
    for sample in dc_from.order_by("f1.nnn", "f1.count").select("f1", "num").to_list():
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_read_parquet_nested_features(tmp_dir, test_session, chunk_size):
    dc_to = dc.read_values(sign1=features_nested, session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = dc.read_parquet(path.as_uri(), session=test_session)

    for n, sample in enumerate(
        dc_from.order_by("sign1.fr.nnn", "sign1.fr.count").select("sign1").to_list()
    ):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


@pytest.mark.parametrize("chunk_size", (1000, 2))
def test_to_read_parquet_two_top_level_features(tmp_dir, test_session, chunk_size):
    dc_to = dc.read_values(f1=features, nest1=features_nested, session=test_session)

    path = tmp_dir / "test.parquet"
    dc_to.to_parquet(path, chunk_size=chunk_size)

    assert path.is_file()

    dc_from = dc.read_parquet(path.as_uri(), session=test_session)

    for n, sample in enumerate(
        dc_from.order_by("f1.nnn", "f1.count").select("f1", "nest1").to_list()
    ):
        assert len(sample) == 2
        fr, nested = sample

        assert isinstance(fr, MyFr)
        assert fr == features[n]
        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


@skip_if_not_sqlite
def test_parallel_in_memory():
    prefix = "t & "
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    with pytest.raises(RuntimeError):
        (
            dc.read_values(key=vals, in_memory=True)
            .settings(parallel=True)
            .map(res=lambda key: prefix + key)
        ).to_values("res")


def test_exec(test_session, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    names = ("f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json")
    all_names = set()

    chain = (
        dc.read_values(name=names, session=test_session)
        .map(nop=lambda name: all_names.add(name))
        .exec()
    )
    assert isinstance(chain, dc.DataChain)
    assert all_names == set(names)


def test_extend_features(test_session):
    chain = dc.read_values(f1=features, num=range(len(features)), session=test_session)

    res = chain._extend_to_data_model("sum", "num")
    assert res == sum(range(len(features)))


def test_read_storage_column(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_storage(path.as_uri(), column="custom", session=test_session)
    assert chain.schema["custom"] == File


def test_from_features_column(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    chain = dc.read_values(fib=fib, odds=values, column="custom", session=test_session)
    assert "custom.fib" in chain.to_pandas(flatten=True).columns


def test_parse_tabular_column(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    df.to_parquet(path)
    chain = dc.read_storage(path.as_uri(), session=test_session).parse_tabular(
        column="tbl"
    )
    assert "tbl.first_name" in chain.to_pandas(flatten=True).columns


def test_sys_feature(test_session, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    ds = dc.read_values(t1=features, session=test_session).order_by(
        "t1.nnn", "t1.count"
    )
    ds_sys = ds.settings(sys=True)
    assert not ds._sys
    assert ds_sys._sys

    args = []
    ds_sys.map(res=lambda sys, t1: args.append((sys, t1))).save("ds_sys")

    sys_cls = Sys.model_construct
    assert args == [
        (sys_cls(id=1, rand=ANY), MyFr(nnn="n1", count=1)),
        (sys_cls(id=2, rand=ANY), MyFr(nnn="n1", count=3)),
        (sys_cls(id=3, rand=ANY), MyFr(nnn="n2", count=5)),
    ]
    assert "sys" not in test_session.catalog.get_dataset("ds_sys").feature_schema

    ds_no_sys = ds_sys.settings(sys=False)
    assert not ds_no_sys._sys

    args = []
    ds_no_sys.map(res=lambda t1: args.append(t1)).save("ds_no_sys")
    assert args == [
        MyFr(nnn="n1", count=1),
        MyFr(nnn="n1", count=3),
        MyFr(nnn="n2", count=5),
    ]
    assert "sys" not in test_session.catalog.get_dataset("ds_no_sys").feature_schema


def test_to_pandas_multi_level(test_session):
    df = dc.read_values(t1=features, session=test_session).to_pandas()

    assert "t1" in df.columns
    assert "nnn" in df["t1"].columns
    assert "count" in df["t1"].columns
    assert sort_df(df)["t1"]["count"].tolist() == [1, 3, 5]


def test_to_pandas_multi_level_flatten(test_session):
    df = dc.read_values(t1=features, session=test_session).to_pandas(flatten=True)

    assert "t1.nnn" in df.columns
    assert "t1.count" in df.columns
    assert len(df.columns) == 2
    assert sort_df(df)["t1.count"].tolist() == [1, 3, 5]


def test_to_pandas_empty(test_session):
    df = (
        dc.read_values(t1=[1, 2, 3], session=test_session)
        .limit(0)
        .to_pandas(flatten=True)
    )

    assert df.empty
    assert "t1" in df.columns
    assert df["t1"].tolist() == []

    df = (
        dc.read_values(my_n=features_nested, session=test_session)
        .limit(0)
        .to_pandas(flatten=False)
    )

    assert df.empty
    assert df["my_n"].empty
    assert list(df.columns) == [
        ("my_n", "label", ""),
        ("my_n", "fr", "nnn"),
        ("my_n", "fr", "count"),
    ]

    df = (
        dc.read_values(my_n=features_nested, session=test_session)
        .limit(0)
        .to_pandas(flatten=True)
    )

    assert df.empty
    assert df["my_n.fr.nnn"].tolist() == []
    assert list(df.columns) == ["my_n.label", "my_n.fr.nnn", "my_n.fr.count"]


def test_mutate(test_session):
    chain = (
        dc.read_values(t1=features, session=test_session)
        .order_by("t1.nnn", "t1.count")
        .mutate(circle=2 * 3.14 * Column("t1.count"), place="pref_" + Column("t1.nnn"))
    )

    assert chain.signals_schema.values["circle"] is float
    assert chain.signals_schema.values["place"] is str

    expected = [fr.count * 2 * 3.14 for fr in features]
    np.testing.assert_allclose(chain.to_values("circle"), expected)


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_with_nested_columns(test_session, with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    chain = dc.read_values(
        file=[File(path=name) for name in names], session=test_session
    )
    if with_function:
        from datachain.sql.functions.random import rand

        chain = chain.order_by("file.path", rand())
    else:
        chain = chain.order_by("file.path")

    assert chain.to_values("file.path") == [
        "a.txt",
        "a.txt",
        "b.txt",
        "c.txt",
        "d.txt",
    ]


def test_order_by_to_list(test_session):
    numbers = [6, 2, 3, 1, 5, 7, 4]
    letters = ["u", "y", "x", "z", "v", "t", "w"]

    chain = dc.read_values(number=numbers, letter=letters, session=test_session)
    assert chain.order_by("number").to_list() == [
        (1, "z"),
        (2, "y"),
        (3, "x"),
        (4, "w"),
        (5, "v"),
        (6, "u"),
        (7, "t"),
    ]

    assert chain.order_by("letter").to_list() == [
        (7, "t"),
        (6, "u"),
        (5, "v"),
        (4, "w"),
        (3, "x"),
        (2, "y"),
        (1, "z"),
    ]


@pytest.mark.parametrize("with_function", [True, False])
def test_order_by_descending(test_session, with_function):
    names = ["a.txt", "c.txt", "d.txt", "a.txt", "b.txt"]

    chain = dc.read_values(
        file=[File(path=name) for name in names], session=test_session
    )
    if with_function:
        from datachain.sql.functions.random import rand

        chain = chain.order_by("file.path", rand(), descending=True)
    else:
        chain = chain.order_by("file.path", descending=True)

    assert chain.to_values("file.path") == [
        "d.txt",
        "c.txt",
        "b.txt",
        "a.txt",
        "a.txt",
    ]


def test_union(test_session):
    chain1 = dc.read_values(value=[1, 2], session=test_session)
    chain2 = dc.read_values(value=[3, 4], session=test_session)
    chain3 = chain1 | chain2
    assert chain3.count() == 4
    assert chain3.order_by("value").to_values("value") == [1, 2, 3, 4]


def test_union_different_columns(test_session):
    chain1 = dc.read_values(value=[1, 2], name=["chain", "more"], session=test_session)
    chain2 = dc.read_values(value=[3, 4], session=test_session)
    chain3 = dc.read_values(other=["a", "different", "thing"], session=test_session)
    with pytest.raises(
        ValueError, match=r"Cannot perform union. name only present in left"
    ):
        chain1.union(chain2).show()
    with pytest.raises(
        ValueError, match=r"Cannot perform union. name only present in right"
    ):
        chain2.union(chain1).show()
    with pytest.raises(
        ValueError,
        match=r"Cannot perform union. "
        r"other only present in left. "
        r"name, value only present in right",
    ):
        chain3.union(chain1).show()


def test_union_different_column_order(test_session):
    chain1 = dc.read_values(value=[1, 2], name=["chain", "more"], session=test_session)
    chain2 = dc.read_values(
        name=["different", "order"], value=[9, 10], session=test_session
    )
    assert chain1.union(chain2).order_by("value").to_list() == [
        (1, "chain"),
        (2, "more"),
        (9, "different"),
        (10, "order"),
    ]


def test_subtract(test_session):
    chain1 = dc.read_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)
    assert set(chain1.subtract(chain2, on=["a", "b"]).to_list()) == {(1, "y"), (2, "z")}
    assert set(chain1.subtract(chain2, on=["b"]).to_list()) == {(2, "z")}
    assert not set(chain1.subtract(chain2, on=["a"]).to_list())
    assert set(chain1.subtract(chain2).to_list()) == {(1, "y"), (2, "z")}
    assert chain1.subtract(chain1).count() == 0

    chain3 = dc.read_values(a=[1, 3], c=["foo", "bar"], session=test_session)
    assert set(chain1.subtract(chain3, on="a").to_list()) == {(2, "z")}
    assert set(chain1.subtract(chain3).to_list()) == {(2, "z")}

    chain4 = dc.read_values(d=[1, 2, 3], e=["x", "y", "z"], session=test_session)
    chain5 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)

    assert set(chain4.subtract(chain5, on="d", right_on="a").to_list()) == {(3, "z")}


def test_subtract_duplicated_rows(test_session):
    chain1 = dc.read_values(id=[1, 1], name=["1", "1"], session=test_session)
    chain2 = dc.read_values(id=[2], name=["2"], session=test_session)
    sub = chain1.subtract(chain2, on="id")
    assert set(sub.to_list()) == {(1, "1"), (1, "1")}


def test_subtract_error(test_session):
    chain1 = dc.read_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=[])
    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="a", right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["c", ""])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["d"])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=["a"])

    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42, right_on=42)

    chain3 = dc.read_values(c=["foo", "bar"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain3)


def test_column_math(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    chain = dc.read_values(num=fib, session=test_session).order_by("num")

    ch = chain.mutate(add2=chain.column("num") + 2)
    assert ch.to_values("add2") == [x + 2 for x in fib]

    ch2 = ch.mutate(x=1 - ch.column("add2"))
    assert ch2.to_values("x") == [1 - (x + 2.0) for x in fib]


@skip_if_not_sqlite
def test_column_math_division(test_session):
    fib = [1, 1, 2, 3, 5, 8]
    chain = dc.read_values(num=fib, session=test_session)

    ch = chain.mutate(div2=chain.column("num") / 2.0)
    assert ch.to_values("div2") == [x / 2.0 for x in fib]


def test_read_values_array_of_floats(test_session):
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    chain = dc.read_values(emd=embeddings, session=test_session)

    expected_embeddings = {tuple(emb) for emb in embeddings}
    actual_embeddings = {tuple(emb) for emb in chain.to_values("emd")}
    assert actual_embeddings == expected_embeddings


def test_custom_model_with_nested_lists(test_session):
    class Trace(BaseModel):
        x: float
        y: float

    class Nested(BaseModel):
        values: list[list[float]]
        traces_single: list[Trace]
        traces_double: list[list[Trace]]

    DataModel.register(Nested)

    ds = dc.read_values(
        nested=[
            Nested(
                values=[[0.5, 0.5], [0.5, 0.5]],
                traces_single=[{"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}],
                traces_double=[[{"x": 0.5, "y": 0.5}], [{"x": 0.5, "y": 0.5}]],
            )
        ],
        nums=[1],
        session=test_session,
    )

    assert ds.to_values("nested") == [
        Nested(
            values=[[0.5, 0.5], [0.5, 0.5]],
            traces_single=[{"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}],
            traces_double=[[{"x": 0.5, "y": 0.5}], [{"x": 0.5, "y": 0.5}]],
        )
    ]


def test_min_limit(test_session):
    chain = dc.read_values(a=[1, 2, 3, 4, 5], session=test_session)
    assert chain.count() == 5
    assert chain.limit(4).count() == 4
    assert chain.count() == 5
    assert chain.limit(1).count() == 1
    assert chain.count() == 5
    assert chain.limit(2).limit(3).count() == 2
    assert chain.count() == 5
    assert chain.limit(3).limit(2).count() == 2
    assert chain.count() == 5


def test_show_limit(test_session):
    chain = dc.read_values(a=[1, 2, 3, 4, 5], session=test_session)
    assert chain.count() == 5
    assert chain.limit(4).count() == 4
    chain.show(1)
    assert chain.count() == 5
    assert chain.limit(1).count() == 1
    chain.show(1)
    assert chain.count() == 5
    assert chain.limit(2).limit(3).count() == 2
    chain.show(1)
    assert chain.count() == 5
    assert chain.limit(3).limit(2).count() == 2
    chain.show(1)
    assert chain.count() == 5


def test_gen_limit(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c", "d"]
    values = [3, 3, 3, 3]

    ds = dc.read_values(key=keys, val=values, session=test_session)

    assert ds.count() == 4
    assert ds.gen(res=func).count() == 12
    assert ds.limit(2).gen(res=func).count() == 6
    assert ds.limit(2).gen(res=func).limit(1).count() == 1
    assert ds.limit(3).gen(res=func).limit(2).count() == 2
    assert ds.limit(2).gen(res=func).limit(3).count() == 3
    assert ds.limit(3).gen(res=func).limit(10).count() == 9


def test_gen_filter(test_session):
    def func(key, val) -> Iterator[tuple[File, str]]:
        for i in range(val):
            yield File(path=""), f"{key}_{i}"

    keys = ["a", "b", "c", "d"]
    values = [3, 3, 3, 3]

    ds = dc.read_values(key=keys, val=values, session=test_session)

    assert ds.count() == 4
    assert ds.gen(res=func).count() == 12
    assert ds.gen(res=func).filter(C("res_1").glob("a_*")).count() == 3


def test_rename_non_object_column_name_with_mutate(test_session):
    ds = dc.read_values(ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(my_ids=Column("ids"))

    assert ds.schema == {"my_ids": int}
    assert ds.order_by("my_ids").to_values("my_ids") == [1, 2, 3]

    assert ds.signals_schema.values.get("my_ids") is int
    assert "ids" not in ds.signals_schema.values
    assert ds.order_by("my_ids").to_values("my_ids") == [1, 2, 3]


def test_rename_object_column_name_with_mutate(test_session):
    names = ["a", "b", "c"]
    sizes = [1, 2, 3]
    files = [
        File(path=name, size=size) for name, size in zip(names, sizes, strict=False)
    ]

    ds = dc.read_values(file=files, ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(fname=Column("file.path"))

    assert ds.order_by("fname").to_values("fname") == ["a", "b", "c"]
    assert ds.schema == {"file": File, "ids": int, "fname": str}

    # check that persist after saving
    ds.save("mutated")

    ds = dc.read_dataset(name="mutated", session=test_session)
    assert ds.signals_schema.values.get("file") is File
    assert ds.signals_schema.values.get("ids") is int
    assert ds.signals_schema.values.get("fname") is str
    assert ds.order_by("fname").to_values("fname") == ["a", "b", "c"]


def test_count_basic(test_session):
    """Test basic count functionality with different data types."""
    # Test with simple values
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)
    assert chain.count() == 5

    # Test with strings
    chain = dc.read_values(names=["Alice", "Bob", "Charlie"], session=test_session)
    assert chain.count() == 3

    # Test with empty chain
    chain = dc.read_values(numbers=[], session=test_session)
    assert chain.count() == 0

    # Test with single item
    chain = dc.read_values(numbers=[42], session=test_session)
    assert chain.count() == 1


def test_count_with_complex_objects(test_session):
    """Test count with complex objects like File and custom models."""
    files = [File(path=f"file_{i}.txt", size=i * 100) for i in range(3)]
    chain = dc.read_values(files=files, session=test_session)
    assert chain.count() == 3

    # Test with nested objects
    chain = dc.read_values(features=features_nested, session=test_session)
    assert chain.count() == 3


def test_count_after_operations(test_session):
    """Test count after various chain operations."""
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)
    assert chain.count() == 5

    # Test after limit
    limited_chain = chain.limit(3)
    assert limited_chain.count() == 3
    assert chain.count() == 5  # Original chain unchanged

    # Test after filter
    filtered_chain = chain.filter(C("numbers") > 3)
    assert filtered_chain.count() == 2
    assert chain.count() == 5  # Original chain unchanged

    # Test after select
    selected_chain = chain.select("numbers")
    assert selected_chain.count() == 5
    assert chain.count() == 5  # Original chain unchanged

    # Test after map
    mapped_chain = chain.map(doubled=lambda numbers: numbers * 2, output=int)
    assert mapped_chain.count() == 5
    assert chain.count() == 5  # Original chain unchanged


def test_count_with_generation(test_session):
    """Test count with generation operations."""

    def generate_items(keys, counts) -> Iterator[tuple[File, str]]:
        for i in range(counts):
            yield File(path=f"{keys}_{i}.txt"), f"item_{i}"

    chain = dc.read_values(keys=["a", "b"], counts=[2, 3], session=test_session)
    assert chain.count() == 2

    # Test after gen operation
    generated_chain = chain.gen(generate_items, output={"file": File, "item": str})
    assert generated_chain.count() == 5  # 2 + 3 = 5 total generated items
    assert chain.count() == 2  # Original chain unchanged


def test_count_with_aggregation(test_session):
    """Test count with aggregation operations."""
    chain = dc.read_values(
        category=["A", "A", "B", "B", "C"], value=[1, 2, 3, 4, 5], session=test_session
    )
    assert chain.count() == 5

    # Test after group_by
    grouped_chain = chain.group_by(total=dc.func.sum("value"), partition_by="category")
    assert grouped_chain.count() == 3  # 3 categories: A, B, C
    assert chain.count() == 5  # Original chain unchanged


def test_count_with_union(test_session):
    """Test count with union operations."""
    chain1 = dc.read_values(numbers=[1, 2, 3], session=test_session)
    chain2 = dc.read_values(numbers=[4, 5], session=test_session)

    assert chain1.count() == 3
    assert chain2.count() == 2

    union_chain = chain1.union(chain2)
    assert union_chain.count() == 5
    # Original chains should remain unchanged
    assert chain1.count() == 3
    assert chain2.count() == 2


def test_count_with_subtract(test_session):
    """Test count with subtract operations."""
    chain1 = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)
    chain2 = dc.read_values(numbers=[2, 4], session=test_session)

    assert chain1.count() == 5
    assert chain2.count() == 2

    subtracted_chain = chain1.subtract(chain2, on="numbers")
    assert subtracted_chain.count() == 3  # 1, 3, 5 remain
    # Original chains should remain unchanged
    assert chain1.count() == 5
    assert chain2.count() == 2


def test_count_persistence(test_session):
    """Test that count persists correctly after operations."""
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)
    assert chain.count() == 5

    # Apply various operations
    chain1 = chain.limit(3).map(doubled=lambda numbers: numbers * 2, output=int)
    assert chain1.count() == 3

    # Test that count is consistent
    assert chain1.count() == 3
    assert chain1.count() == 3  # Should be idempotent
    assert chain.count() == 5  # Original chain unchanged


def test_count_with_empty_results(test_session):
    """Test count with operations that result in empty chains."""
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)

    # Filter to empty result
    empty_chain = chain.filter(C("numbers") > 10)
    assert empty_chain.count() == 0

    # Limit to 0
    assert chain.limit(0).count() == 0
    assert empty_chain.limit(0).count() == 0
    assert chain.count() == 5


@skip_if_not_sqlite
def test_count_in_memory(test_session):
    """Test count functionality with in-memory database."""
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], in_memory=True)
    assert chain.count() == 5

    # Test with operations
    limited_chain = chain.limit(3)
    assert limited_chain.count() == 3
    assert chain.count() == 5  # Original chain unchanged

    filtered_chain = chain.filter(C("numbers") > 3)
    assert filtered_chain.count() == 2
    assert chain.count() == 5  # Original chain unchanged


def test_distinct_basic(test_session):
    """Test basic distinct functionality with simple data types."""
    # Test with simple values - no duplicates
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], session=test_session)
    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 5
    assert sorted(distinct_chain.to_values("numbers")) == [1, 2, 3, 4, 5]

    # Test with duplicates
    chain = dc.read_values(numbers=[1, 2, 2, 3, 3, 3, 4], session=test_session)
    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 4
    assert sorted(distinct_chain.to_values("numbers")) == [1, 2, 3, 4]

    # Test with strings
    chain = dc.read_values(
        names=["Alice", "Bob", "Alice", "Charlie", "Bob"], session=test_session
    )
    distinct_chain = chain.distinct("names")
    assert distinct_chain.count() == 3
    assert sorted(distinct_chain.to_values("names")) == ["Alice", "Bob", "Charlie"]


def test_distinct_multiple_columns(test_session):
    """Test distinct with multiple columns."""
    chain = dc.read_values(
        category=["A", "A", "B", "B", "C"], value=[1, 2, 1, 2, 2], session=test_session
    )

    # Test distinct on single column
    distinct_category = chain.distinct("category")
    assert distinct_category.count() == 3
    assert sorted(distinct_category.to_values("category")) == ["A", "B", "C"]
    distinct_value = chain.distinct("value")
    assert distinct_value.count() == 2
    assert sorted(distinct_value.to_values("value")) == [1, 2]

    # Test distinct on multiple columns
    distinct_both = chain.distinct("category", "value")
    assert distinct_both.count() == 5  # All combinations are unique
    assert sorted(distinct_both.to_list("category", "value")) == [
        ("A", 1),
        ("A", 2),
        ("B", 1),
        ("B", 2),
        ("C", 2),
    ]


def test_distinct_with_complex_objects(test_session):
    """Test distinct with complex objects like File and custom models."""
    files = [
        File(path="file1.txt", size=100),
        File(path="file2.txt", size=200),
        File(path="file1.txt", size=100),  # Duplicate
        File(path="file3.txt", size=300),
    ]

    chain = dc.read_values(files=files, session=test_session)
    distinct_chain = chain.distinct("files")
    assert distinct_chain.count() == 3  # Removes duplicate file1.txt

    # Test with nested objects
    chain = dc.read_values(features=features_nested, session=test_session)
    distinct_chain = chain.distinct("features")
    assert distinct_chain.count() == 3  # All features are unique


def test_distinct_with_nested_fields(test_session):
    """Test distinct with nested field references."""
    files = [
        File(path="dir1/file1.txt", size=100),
        File(path="dir1/file2.txt", size=200),
        File(path="dir2/file1.txt", size=100),
        File(path="dir1/file1.txt", size=150),  # Different size, same path
    ]

    chain = dc.read_values(files=files, session=test_session)

    # Test distinct on nested field
    distinct_path = chain.distinct("files.path")
    assert distinct_path.count() == 3  # 3 unique paths
    assert sorted(distinct_path.to_values("files.path")) == [
        "dir1/file1.txt",
        "dir1/file2.txt",
        "dir2/file1.txt",
    ]

    # Test distinct on multiple nested fields
    distinct_path_size = chain.distinct("files.path", "files.size")
    assert distinct_path_size.count() == 4  # All combinations are unique


def test_distinct_after_operations(test_session):
    """Test distinct after various chain operations."""
    chain = dc.read_values(
        numbers=[1, 2, 2, 3, 3, 3, 4],
        categories=["A", "A", "B", "B", "C", "C", "D"],
        session=test_session,
    )

    # Test distinct after filter
    filtered_chain = chain.filter(C("numbers") > 2)
    distinct_filtered = filtered_chain.distinct("numbers")
    assert distinct_filtered.count() == 2  # Only 3 and 4
    assert sorted(distinct_filtered.to_values("numbers")) == [3, 4]

    # Test distinct after map
    mapped_chain = chain.map(doubled=lambda numbers: numbers * 2, output=int)
    distinct_mapped = mapped_chain.distinct("doubled")
    assert distinct_mapped.count() == 4  # 2, 4, 6, 8
    assert sorted(distinct_mapped.to_values("doubled")) == [2, 4, 6, 8]

    # Test distinct after select
    selected_chain = chain.select("numbers")
    distinct_selected = selected_chain.distinct("numbers")
    assert distinct_selected.count() == 4
    assert sorted(distinct_selected.to_values("numbers")) == [1, 2, 3, 4]


def test_distinct_with_empty_chain(test_session):
    """Test distinct with empty chain."""
    chain = dc.read_values(numbers=[], session=test_session)
    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 0


def test_distinct_with_single_item(test_session):
    """Test distinct with single item."""
    chain = dc.read_values(numbers=[42], session=test_session)
    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 1
    assert distinct_chain.to_values("numbers") == [42]


def test_distinct_with_all_duplicates(test_session):
    """Test distinct when all values are duplicates."""
    chain = dc.read_values(
        numbers=[1, 1, 1, 1, 1],
        categories=["A", "A", "A", "A", "A"],
        session=test_session,
    )

    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 1
    assert distinct_chain.to_values("numbers") == [1]

    distinct_both = chain.distinct("numbers", "categories")
    assert distinct_both.count() == 1
    assert distinct_both.to_values("numbers") == [1]
    assert distinct_both.to_values("categories") == ["A"]


@skip_if_not_sqlite
def test_distinct_in_memory(test_session):
    """Test distinct functionality with in-memory database."""
    chain = dc.read_values(numbers=[1, 2, 2, 3, 3, 3], in_memory=True)
    distinct_chain = chain.distinct("numbers")
    assert distinct_chain.count() == 3
    assert sorted(distinct_chain.to_values("numbers")) == [1, 2, 3]


def test_distinct_error_handling(test_session):
    """Test distinct with invalid column names."""
    chain = dc.read_values(numbers=[1, 2, 3], session=test_session)

    # Test with non-existent column
    with pytest.raises(SignalResolvingError):
        chain.distinct("non_existent_column")

    # Test with empty string
    with pytest.raises(SignalResolvingError):
        chain.distinct("")

    # Test with invalid type
    with pytest.raises(SignalResolvingTypeError):
        chain.distinct(42)

    # Test with invalid nested field
    with pytest.raises(SignalResolvingError):
        chain.distinct("numbers.invalid_field")


def test_filter_basic(test_session):
    """Test basic filter functionality with simple conditions."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Test basic comparison operators
    filtered_chain = chain.filter(C("numbers") > 5)
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [6, 7, 8, 9, 10]

    filtered_chain = chain.filter(C("numbers") >= 5)
    assert filtered_chain.count() == 6
    assert sorted(filtered_chain.to_values("numbers")) == [5, 6, 7, 8, 9, 10]

    filtered_chain = chain.filter(C("numbers") < 5)
    assert filtered_chain.count() == 4
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 3, 4]

    filtered_chain = chain.filter(C("numbers") <= 5)
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 3, 4, 5]

    filtered_chain = chain.filter(C("numbers") == 5)
    assert filtered_chain.count() == 1
    assert filtered_chain.to_values("numbers") == [5]

    filtered_chain = chain.filter(C("numbers") != 5)
    assert filtered_chain.count() == 9
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 3, 4, 6, 7, 8, 9, 10]


def test_filter_with_booleans_read_values(test_session):
    """Ensure boolean columns from read_values work with filters."""
    chain = dc.read_values(
        flags=[True, False, True],
        labels=["keep", "skip", "keep"],
        session=test_session,
    )

    # Direct column filter defaults to truthy rows
    true_rows = chain.filter(C("flags"))
    assert true_rows.count() == 2
    assert true_rows.to_values("labels") == ["keep", "keep"]

    # Equality predicate
    true_rows_eq = chain.filter(C("flags") == True)  # noqa: E712
    assert true_rows_eq.to_values("labels") == ["keep", "keep"]

    # Inclusion predicate
    true_rows_in = chain.filter(C("flags").in_([True]))
    assert true_rows_in.to_values("labels") == ["keep", "keep"]

    # False branch
    false_rows = chain.filter(C("flags") == False)  # noqa: E712
    assert false_rows.to_values("labels") == ["skip"]

    false_rows_in = chain.filter(C("flags").in_([False]))
    assert false_rows_in.to_values("labels") == ["skip"]


def test_filter_with_strings(test_session):
    """Test filter with string conditions."""
    chain = dc.read_values(
        names=["Alice", "Bob", "Charlie", "David", "Eva", "Alice"],
        ages=[25, 30, 35, 40, 45, 50],
        cities=["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Denver"],
        session=test_session,
    )

    # Test string equality
    filtered_chain = chain.filter(C("names") == "Alice")
    assert filtered_chain.count() == 2
    assert filtered_chain.to_values("names") == ["Alice", "Alice"]

    # Test string inequality
    filtered_chain = chain.filter(C("names") != "Alice")
    assert filtered_chain.count() == 4
    assert sorted(filtered_chain.to_values("names")) == [
        "Bob",
        "Charlie",
        "David",
        "Eva",
    ]


def test_filter_with_glob_patterns(test_session):
    """Test filter with glob patterns."""
    files = [
        File(path="image1.jpg", size=100),
        File(path="image2.png", size=200),
        File(path="document1.pdf", size=300),
        File(path="image3.jpg", size=400),
        File(path="document2.txt", size=500),
    ]

    chain = dc.read_values(files=files, session=test_session)

    # Test glob pattern for file extensions
    filtered_chain = chain.filter(C("files.path").glob("*.jpg"))
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("files.path")) == [
        "image1.jpg",
        "image3.jpg",
    ]

    # Test glob pattern for file names
    filtered_chain = chain.filter(C("files.path").glob("image*"))
    assert filtered_chain.count() == 3
    assert sorted(filtered_chain.to_values("files.path")) == [
        "image1.jpg",
        "image2.png",
        "image3.jpg",
    ]

    # Test glob pattern for document files
    filtered_chain = chain.filter(C("files.path").glob("document*"))
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("files.path")) == [
        "document1.pdf",
        "document2.txt",
    ]


def test_filter_with_in_operator(test_session):
    """Test filter with 'in' operator."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Test in operator with list
    filtered_chain = chain.filter(C("numbers").in_([1, 3, 5, 7, 9]))
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [1, 3, 5, 7, 9]

    # Test in operator with categories
    filtered_chain = chain.filter(C("categories").in_(["A", "C"]))
    assert filtered_chain.count() == 7
    assert sorted(filtered_chain.to_values("categories")) == [
        "A",
        "A",
        "A",
        "A",
        "C",
        "C",
        "C",
    ]


def test_filter_with_and_operator(test_session):
    """Test filter with multiple conditions using AND."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Test multiple conditions (implicit AND)
    filtered_chain = chain.filter(C("numbers") > 5, C("categories") == "A")
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("numbers")) == [6, 9]

    # Test with explicit AND operator
    filtered_chain = chain.filter((C("numbers") > 5) & (C("categories") == "A"))
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("numbers")) == [6, 9]

    # Test with func.and_
    filtered_chain = chain.filter(func.and_(C("numbers") > 5, C("categories") == "A"))
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("numbers")) == [6, 9]


def test_filter_with_or_operator(test_session):
    """Test filter with OR operator."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Test OR operator
    filtered_chain = chain.filter((C("numbers") < 3) | (C("numbers") > 8))
    assert filtered_chain.count() == 4
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 9, 10]

    filtered_chain = chain.filter((C("categories") == "A") | (C("numbers") == 5))
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [1, 3, 5, 6, 9]

    # Test with func.or_
    filtered_chain = chain.filter(func.or_(C("numbers") < 3, C("numbers") > 8))
    assert filtered_chain.count() == 4
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 9, 10]


def test_filter_with_not_operator(test_session):
    """Test filter with NOT operator."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Test NOT operator
    filtered_chain = chain.filter(~(C("numbers") > 5))
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 3, 4, 5]

    filtered_chain = chain.filter(~(C("categories") == "A"))
    assert filtered_chain.count() == 6
    assert sorted(filtered_chain.to_values("categories")) == [
        "B",
        "B",
        "B",
        "C",
        "C",
        "C",
    ]

    # Test with func.not_
    filtered_chain = chain.filter(func.not_(C("numbers") > 5))
    assert filtered_chain.count() == 5
    assert sorted(filtered_chain.to_values("numbers")) == [1, 2, 3, 4, 5]


def test_filter_with_complex_objects(test_session):
    """Test filter with complex objects."""
    files = [
        File(path="image1.jpg", size=100),
        File(path="image2.png", size=200),
        File(path="document1.pdf", size=300),
        File(path="image3.jpg", size=400),
        File(path="document2.txt", size=500),
    ]

    chain = dc.read_values(files=files, session=test_session)

    # Test filter on nested field
    filtered_chain = chain.filter(C("files.size") > 250)
    assert filtered_chain.count() == 3
    assert sorted(filtered_chain.to_values("files.size")) == [300, 400, 500]

    # Test filter on nested field with glob
    filtered_chain = chain.filter(C("files.path").glob("*.jpg"))
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("files.path")) == [
        "image1.jpg",
        "image3.jpg",
    ]


def test_filter_with_empty_results(test_session):
    """Test filter that results in empty chain."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5],
        categories=["A", "B", "A", "C", "B"],
        session=test_session,
    )

    # Test filter that returns no results
    filtered_chain = chain.filter(C("numbers") > 10)
    assert filtered_chain.count() == 0
    assert filtered_chain.to_values("numbers") == []

    # Test filter with impossible condition
    filtered_chain = chain.filter(C("numbers") == 10)
    assert filtered_chain.count() == 0
    assert filtered_chain.to_values("numbers") == []


def test_filter_with_booleans_round_trip_dataset(test_session):
    """Boolean columns remain filterable after save/read round trip."""
    ds_name = f"bool-filter-{uuid.uuid4().hex}"
    chain = dc.read_values(
        published=[True, False, True],
        identifiers=[1, 2, 3],
        session=test_session,
    )

    chain.save(ds_name)
    reloaded = dc.read_dataset(name=ds_name, session=test_session)

    published_rows = reloaded.filter(C("published"))
    assert sorted(published_rows.to_values("identifiers")) == [1, 3]

    published_rows_eq = reloaded.filter(C("published") == True)  # noqa: E712
    assert sorted(published_rows_eq.to_values("identifiers")) == [1, 3]

    published_rows_in = reloaded.filter(C("published").in_([True]))
    assert sorted(published_rows_in.to_values("identifiers")) == [1, 3]

    draft_rows = reloaded.filter(C("published") == False)  # noqa: E712
    assert draft_rows.to_values("identifiers") == [2]

    draft_rows_in = reloaded.filter(C("published").in_([False]))
    assert draft_rows_in.to_values("identifiers") == [2]


def test_filter_chaining(test_session):
    """Test chaining multiple filter operations."""
    chain = dc.read_values(
        numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        categories=["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        session=test_session,
    )

    # Chain multiple filters
    filtered_chain = (
        chain.filter(C("numbers") >= 3)
        .filter(C("numbers") < 8)
        .filter(C("categories") == "A")
    )
    assert filtered_chain.count() == 2
    assert sorted(filtered_chain.to_values("numbers")) == [3, 6]

    # Test that original chain is unchanged
    assert chain.count() == 10
    assert sorted(chain.to_values("numbers")) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_filter_with_func_operations(test_session):
    """Test filter with datachain.func operations."""
    from datachain.func import string

    chain = dc.read_values(
        names=["Alice", "Bob", "Charlie", "David", "Eva"],
        ages=[25, 30, 35, 40, 45],
        session=test_session,
    )

    # Test string length filter
    filtered_chain = chain.filter(string.length(C("names")) > 4)
    assert filtered_chain.count() == 3
    assert sorted(filtered_chain.to_values("names")) == ["Alice", "Charlie", "David"]


def test_filter_with_boolean_nested_model(test_session):
    """Boolean fields inside nested DataModels are filterable."""

    class ReviewMeta(DataModel):
        passed: bool
        notes: str

    class Item(DataModel):
        name: str
        meta: ReviewMeta

    items = [
        Item(name="doc-1", meta=ReviewMeta(passed=True, notes="looks good")),
        Item(name="doc-2", meta=ReviewMeta(passed=False, notes="needs work")),
        Item(name="doc-3", meta=ReviewMeta(passed=True, notes="approved")),
    ]

    chain = dc.read_values(records=items, session=test_session)

    approved = chain.filter(C("records.meta.passed"))
    assert sorted(approved.to_values("records.name")) == ["doc-1", "doc-3"]  # type: ignore[arg-type]

    approved_eq = chain.filter(C("records.meta.passed") == True)  # noqa: E712
    assert sorted(approved_eq.to_values("records.name")) == ["doc-1", "doc-3"]  # type: ignore[arg-type]

    approved_in = chain.filter(C("records.meta.passed").in_([True]))
    assert sorted(approved_in.to_values("records.name")) == ["doc-1", "doc-3"]  # type: ignore[arg-type]

    rejected = chain.filter(C("records.meta.passed") == False)  # noqa: E712
    assert rejected.to_values("records.name") == ["doc-2"]

    rejected_in = chain.filter(C("records.meta.passed").in_([False]))
    assert rejected_in.to_values("records.name") == ["doc-2"]


@skip_if_not_sqlite
def test_filter_in_memory(test_session):
    """Test filter functionality with in-memory database."""
    chain = dc.read_values(numbers=[1, 2, 3, 4, 5], in_memory=True)

    filtered_chain = chain.filter(C("numbers") > 3)
    assert filtered_chain.count() == 2
    assert filtered_chain.to_values("numbers") == [4, 5]

    # Test that original chain is unchanged
    assert chain.count() == 5
    assert chain.to_values("numbers") == [1, 2, 3, 4, 5]


def test_rename_column_with_mutate(test_session):
    names = ["a", "b", "c"]
    sizes = [1, 2, 3]
    files = [
        File(path=name, size=size) for name, size in zip(names, sizes, strict=False)
    ]

    ds = dc.read_values(file=files, ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(my_file=Column("file"))

    assert ds.order_by("my_file.path").to_values("my_file.path") == ["a", "b", "c"]
    assert ds.schema == {"my_file": File, "ids": int}

    # check that persist after saving
    ds.save("mutated")

    ds = dc.read_dataset(name="mutated", session=test_session)
    schema = ds.schema
    assert schema.get("my_file") is File
    assert schema.get("ids") is int
    assert "file" not in schema
    assert ds.order_by("my_file.path").to_values("my_file.path") == ["a", "b", "c"]


def test_column(test_session):
    ds = dc.read_values(
        ints=[1, 2],
        floats=[0.5, 0.5],
        file=[File(path="a"), File(path="b")],
        session=test_session,
    )

    c = ds.column("ints")
    assert isinstance(c, Column)
    assert c.name == "ints"
    assert isinstance(c.type, Int64)

    c = ds.column("floats")
    assert isinstance(c, Column)
    assert c.name == "floats"
    assert isinstance(c.type, Float)

    c = ds.column("file.path")
    assert isinstance(c, Column)
    assert c.name == "file__path"
    assert isinstance(c.type, String)

    with pytest.raises(ValueError):
        c = ds.column("missing")


def test_mutate_with_subtraction(test_session):
    ds = dc.read_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") - 1).signals_schema.values["new"] is int


def test_mutate_with_addition(test_session):
    ds = dc.read_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") + 1).signals_schema.values["new"] is int


def test_mutate_with_division(test_session):
    ds = dc.read_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") / 10).signals_schema.values["new"] is float


def test_mutate_with_multiplication(test_session):
    ds = dc.read_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=ds.column("id") * 10).signals_schema.values["new"] is int


def test_mutate_with_sql_func(test_session):
    from datachain import func

    ds = dc.read_values(id=[1, 2], session=test_session)
    assert ds.mutate(new=func.avg("id")).signals_schema.values["new"] is float


def test_mutate_with_complex_expression(test_session):
    from datachain import func

    ds = dc.read_values(id=[1, 2], name=["Jim", "Jon"], session=test_session)
    assert (
        ds.mutate(new=func.sum("id") * (5 - func.min("id"))).signals_schema.values[
            "new"
        ]
        is int
    )


@skip_if_not_sqlite
def test_mutate_with_saving(test_session):
    ds = dc.read_values(id=[1, 2], session=test_session)
    ds = ds.mutate(new=ds.column("id") / 2).save("mutated")

    ds = dc.read_dataset(name="mutated", session=test_session)
    assert ds.signals_schema.values["new"] is float
    assert ds.to_values("new") == [0.5, 1.0]


def test_mutate_with_expression_without_type(test_session):
    with pytest.raises(DataChainColumnError) as excinfo:
        dc.read_values(id=[1, 2], session=test_session).mutate(
            new=(Column("id") - 1)
        ).persist()

    assert str(excinfo.value) == (
        "Error for column new: Cannot infer type with expression id - :id_1"
    )


def test_read_values_nan_inf(test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    chain = dc.read_values(vals=vals, session=test_session)
    res = chain.to_values("vals")
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_read_pandas_nan_inf(test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    df = pd.DataFrame({"vals": vals})
    chain = dc.read_pandas(df, session=test_session)
    res = chain.to_values("vals")
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_read_parquet_nan_inf(tmp_dir, test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    tbl = pa.table({"vals": vals})
    path = tmp_dir / "test.parquet"
    pq.write_table(tbl, path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)

    res = list(chain.to_values("vals"))
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_read_csv_nan_inf(tmp_dir, test_session):
    vals = [float("nan"), float("inf"), float("-inf")]
    df = pd.DataFrame({"vals": vals})
    path = tmp_dir / "test.csv"
    df.to_csv(path, index=False)
    chain = dc.read_csv(path.as_uri(), session=test_session)

    res = chain.to_values("vals")
    assert len(res) == 3
    assert any(r for r in res if np.isnan(r))
    assert any(r for r in res if np.isposinf(r))
    assert any(r for r in res if np.isneginf(r))


def test_dicts_nan_inf(test_session):
    metrics_data = [
        {"accuracy": 0.95, "loss": 0.1, "precision": 0.92},
        {"accuracy": float("nan"), "loss": float("inf"), "precision": 0.88},
        {"accuracy": 0.87, "loss": float("-inf"), "precision": float("nan")},
    ]

    dc.read_values(
        id=[1, 2, 3],
        metrics=metrics_data,
        session=test_session,
    ).save("test_dicts_nan_inf")

    res = dc.read_dataset("test_dicts_nan_inf").order_by("id").to_values("metrics")
    assert len(res) == 3

    assert res[0]["accuracy"] == 0.95
    assert res[0]["loss"] == 0.1
    assert res[0]["precision"] == 0.92

    assert math.isnan(res[1]["accuracy"])
    assert math.isinf(res[1]["loss"]) and res[1]["loss"] > 0
    assert res[1]["precision"] == 0.88

    assert res[2]["accuracy"] == 0.87
    assert math.isinf(res[2]["loss"]) and res[2]["loss"] < 0
    assert math.isnan(res[2]["precision"])


def test_group_by_int(test_session):
    from datachain import func

    ds = (
        dc.read_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 3, 4, 5, 6],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col2"),
            avg=func.avg("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "int",
        "avg": "float",
        "min": "int",
        "max": "int",
        "value": "int",
        "collect": "list[int]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "sum": 3,
                "avg": 1.5,
                "min": 1,
                "max": 2,
                "value": ANY_VALUE(1, 2),
                "collect": [1, 2],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "sum": 12,
                "avg": 4.0,
                "min": 3,
                "max": 5,
                "value": ANY_VALUE(3, 4, 5),
                "collect": [3, 4, 5],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6,
                "avg": 6.0,
                "min": 6,
                "max": 6,
                "value": ANY_VALUE(6),
                "collect": [6],
            },
        ],
        "col1",
    )


def test_group_by_float(test_session):
    from datachain import func

    ds = (
        dc.read_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col2"),
            avg=func.avg("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "avg": "float",
        "min": "float",
        "max": "float",
        "value": "float",
        "collect": "list[float]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "sum": 4.0,
                "avg": 2.0,
                "min": 1.5,
                "max": 2.5,
                "value": ANY_VALUE(1.5, 2.5),
                "collect": [1.5, 2.5],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "sum": 13.5,
                "avg": 4.5,
                "min": 3.5,
                "max": 5.5,
                "value": ANY_VALUE(3.5, 4.5, 5.5),
                "collect": [3.5, 4.5, 5.5],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6.5,
                "avg": 6.5,
                "min": 6.5,
                "max": 6.5,
                "value": ANY_VALUE(6.5),
                "collect": [6.5],
            },
        ],
        "col1",
    )


def test_group_by_str(test_session):
    from datachain import func

    ds = (
        dc.read_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col1", "col2")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            min=func.min("col2"),
            max=func.max("col2"),
            concat=func.concat("col2"),
            concat_sep=func.concat("col2", separator=","),
            value=func.any_value("col2"),
            collect=func.collect("col2"),
            partition_by="col1",
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "cnt": "int",
        "cnt_col": "int",
        "min": "str",
        "max": "str",
        "concat": "str",
        "concat_sep": "str",
        "value": "str",
        "collect": "list[str]",
    }
    assert sorted_dicts(ds.to_records(), "col1") == sorted_dicts(
        [
            {
                "col1": "a",
                "cnt": 2,
                "cnt_col": 2,
                "min": "1",
                "max": "2",
                "concat": "12",
                "concat_sep": "1,2",
                "value": ANY_VALUE("1", "2"),
                "collect": ["1", "2"],
            },
            {
                "col1": "b",
                "cnt": 3,
                "cnt_col": 3,
                "min": "3",
                "max": "5",
                "concat": "345",
                "concat_sep": "3,4,5",
                "value": ANY_VALUE("3", "4", "5"),
                "collect": ["3", "4", "5"],
            },
            {
                "col1": "c",
                "cnt": 1,
                "cnt_col": 1,
                "min": "6",
                "max": "6",
                "concat": "6",
                "concat_sep": "6",
                "value": ANY_VALUE("6"),
                "collect": ["6"],
            },
        ],
        "col1",
    )


def test_group_by_multiple_partition_by(test_session):
    from datachain import func

    ds = (
        dc.read_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 1, 2, 1, 2],
            col3=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col4=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col1", "col2", "col3", "col4")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col3"),
            concat=func.concat("col4"),
            value=func.any_value("col3"),
            collect=func.collect("col3"),
            partition_by=("col1", "col2"),
        )
    )

    assert ds.signals_schema.serialize() == {
        "col1": "str",
        "col2": "int",
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "concat": "str",
        "value": "float",
        "collect": "list[float]",
    }
    assert sorted_dicts(ds.to_records(), "col1", "col2") == sorted_dicts(
        [
            {
                "col1": "a",
                "col2": 1,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 1.0,
                "concat": "1",
                "value": ANY_VALUE(1.0),
                "collect": [1.0],
            },
            {
                "col1": "a",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 2.0,
                "concat": "2",
                "value": ANY_VALUE(2.0),
                "collect": [2.0],
            },
            {
                "col1": "b",
                "col2": 1,
                "cnt": 2,
                "cnt_col": 2,
                "sum": 8.0,
                "concat": "35",
                "value": ANY_VALUE(3.0, 5.0),
                "collect": [3.0, 5.0],
            },
            {
                "col1": "b",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 4.0,
                "concat": "4",
                "value": ANY_VALUE(4.0),
                "collect": [4.0],
            },
            {
                "col1": "c",
                "col2": 2,
                "cnt": 1,
                "cnt_col": 1,
                "sum": 6.0,
                "concat": "6",
                "value": ANY_VALUE(6.0),
                "collect": [6.0],
            },
        ],
        "col1",
        "col2",
    )


def test_group_by_no_partition_by(test_session):
    from datachain import func

    ds = (
        dc.read_values(
            col1=["a", "a", "b", "b", "b", "c"],
            col2=[1, 2, 1, 2, 1, 2],
            col3=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col4=["1", "2", "3", "4", "5", "6"],
            session=test_session,
        )
        .order_by("col4")
        .group_by(
            cnt=func.count(),
            cnt_col=func.count("col2"),
            sum=func.sum("col3"),
            concat=func.concat("col4"),
            value=func.any_value("col3"),
            collect=func.collect("col3"),
        )
    )

    assert ds.signals_schema.serialize() == {
        "cnt": "int",
        "cnt_col": "int",
        "sum": "float",
        "concat": "str",
        "value": "float",
        "collect": "list[float]",
    }
    assert ds.to_records() == [
        {
            "cnt": 6,
            "cnt_col": 6,
            "sum": 21.0,
            "concat": "123456",
            "value": 1.0,
            "collect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    ]


def test_group_by_schema(test_session):
    from datachain import func
    from datachain.lib.model_store import ModelStore

    class Signal(DataModel):
        name: str
        value: float

    class Parent(DataModel):
        signal: Signal

    def multiplier(name: str, val: float) -> Signal:
        return Signal(name=name, value=val * 2)

    def to_parent(signal: Signal) -> Parent:
        return Parent(signal=signal)

    chain = (
        dc.read_values(
            name=["a", "a", "b", "b", "b", "c"],
            val=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .map(signal=multiplier)
        .map(parent=to_parent)
        .order_by("signal.name")
        .group_by(
            cnt=func.count(),
            sum=func.sum("signal.value"),
            partition_by=["signal.name", "parent.signal.name"],
        )
        .select("signal.name", "parent.signal.name", "cnt", "sum")
    )

    serialized = chain.signals_schema.serialize()
    assert serialized["cnt"] == "int"
    assert serialized["sum"] == "float"
    assert "signal" in serialized
    assert "parent" in serialized
    assert "_custom_types" in serialized

    assert ModelStore.is_partial(chain.schema["signal"])
    assert ModelStore.is_partial(chain.schema["parent"])

    signal_type_name = serialized["signal"]
    parent_type_name = serialized["parent"]
    assert serialized["_custom_types"][signal_type_name]["fields"] == {"name": "str"}
    assert serialized["_custom_types"][parent_type_name]["fields"] == {
        "signal": signal_type_name
    }
    assert sorted(
        chain.to_records(),
        key=lambda row: (row["signal__name"], row["parent__signal__name"]),
    ) == [
        {
            "signal__name": "a",
            "parent__signal__name": "a",
            "cnt": 2,
            "sum": 6.0,
        },
        {
            "signal__name": "b",
            "parent__signal__name": "b",
            "cnt": 3,
            "sum": 24.0,
        },
        {
            "signal__name": "c",
            "parent__signal__name": "c",
            "cnt": 1,
            "sum": 12.0,
        },
    ]


def test_group_by_error(test_session):
    from datachain import func

    chain = dc.read_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    )

    with pytest.raises(
        ValueError, match="At least one column should be provided for group_by"
    ):
        chain.group_by(partition_by="col1")

    with pytest.raises(
        DataChainColumnError,
        match="Column foo has type <class 'str'> but expected Func object",
    ):
        chain.group_by(foo="col2", partition_by="col1")

    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'col3': is not found"
    ):
        chain.group_by(foo=func.sum("col3"), partition_by="col1")

    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'col3': is not found"
    ):
        chain.group_by(foo=func.sum("col2"), partition_by="col3")


def test_group_by_case(test_session):
    from datachain import func

    ds = dc.read_values(
        col1=[1.0, 0.0, 3.2, 0.1, 5.9, -1.0],
        col2=[0.0, 6.1, -0.05, 3.7, 0.1, -3.0],
        session=test_session,
    ).group_by(
        col1=func.sum(func.case((C("col1") > 0.1, 1), else_=0)),
        col2=func.sum(func.case((C("col2") < 0.0, 1), else_=0)),
    )

    assert ds.signals_schema.serialize() == {
        "col1": "int",
        "col2": "int",
    }
    assert ds.to_records() == [
        {
            "col1": 3,
            "col2": 2,
        }
    ]


@pytest.mark.parametrize("desc", [True, False])
def test_window_functions(test_session, desc):
    from datachain import func

    window = func.window(partition_by="col1", order_by="col2", desc=desc)

    ds = dc.read_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    ).mutate(
        row_number=func.row_number().over(window),
        rank=func.rank().over(window),
        dense_rank=func.dense_rank().over(window),
        first=func.first("col2").over(window),
    )

    assert ds.schema == {
        "col1": str,
        "col2": int,
        "row_number": int,
        "rank": int,
        "dense_rank": int,
        "first": int,
    }

    assert sorted_dicts(ds.to_records(), "col1", "col2") == sorted_dicts(
        [
            {
                "col1": "a",
                "col2": 1,
                "row_number": 2 if desc else 1,
                "rank": 2 if desc else 1,
                "dense_rank": 2 if desc else 1,
                "first": 2 if desc else 1,
            },
            {
                "col1": "a",
                "col2": 2,
                "row_number": 1 if desc else 2,
                "rank": 1 if desc else 2,
                "dense_rank": 1 if desc else 2,
                "first": 2 if desc else 1,
            },
            {
                "col1": "b",
                "col2": 3,
                "row_number": 3 if desc else 1,
                "rank": 3 if desc else 1,
                "dense_rank": 3 if desc else 1,
                "first": 5 if desc else 3,
            },
            {
                "col1": "b",
                "col2": 4,
                "row_number": 2,
                "rank": 2,
                "dense_rank": 2,
                "first": 5 if desc else 3,
            },
            {
                "col1": "b",
                "col2": 5,
                "row_number": 1 if desc else 3,
                "rank": 1 if desc else 3,
                "dense_rank": 1 if desc else 3,
                "first": 5 if desc else 3,
            },
            {
                "col1": "c",
                "col2": 6,
                "row_number": 1,
                "rank": 1,
                "dense_rank": 1,
                "first": 6,
            },
        ],
        "col1",
        "col2",
    )


def test_window_error(test_session):
    from datachain import func

    window = func.window(partition_by="col1", order_by="col2")

    chain = dc.read_values(
        col1=["a", "a", "b", "b", "b", "c"],
        col2=[1, 2, 3, 4, 5, 6],
        session=test_session,
    )

    with pytest.raises(
        DataChainParamsError,
        match=re.escape(
            "Window function first() requires over() clause with a window spec",
        ),
    ):
        chain.mutate(first=func.first("col2"))

    with pytest.raises(
        DataChainParamsError,
        match=re.escape(
            "sum() doesn't support window (over())",
        ),
    ):
        chain.mutate(first=func.sum("col2").over(window))


def test_delete_dataset_version(test_session):
    name = "numbers"
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="1.0.0")
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="2.0.0")

    dc.delete_dataset(name, version="1.0.0", session=test_session)

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == name]
    assert len(datasets) == 1
    assert datasets[0].version == "2.0.0"


def test_delete_dataset_latest_version(test_session):
    name = "numbers"
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="1.0.0")
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="2.0.0")

    dc.delete_dataset(name, session=test_session)

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == name]
    assert len(datasets) == 1
    assert datasets[0].version == "1.0.0"


def test_delete_dataset_only_version(test_session):
    name = "numbers"
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="1.0.0")

    dc.delete_dataset(name, session=test_session)

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == name]
    assert len(datasets) == 0


def test_delete_dataset_missing_version(test_session):
    name = "numbers"
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="1.0.0")
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="2.0.0")

    with pytest.raises(DatasetInvalidVersionError):
        dc.delete_dataset(name, version="5.0.0", session=test_session)


def test_delete_dataset_versions_all(test_session):
    name = "numbers"
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="1.0.0")
    dc.read_values(num=[1, 2, 3], session=test_session).save(name, version="2.0.0")

    dc.delete_dataset(name, force=True, session=test_session)

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [d for d in ds.to_values("dataset") if d.name == name]
    assert len(datasets) == 0


@pytest.mark.parametrize("force", (True, False))
@pytest.mark.parametrize("is_studio", (False,))
@skip_if_not_sqlite
def test_delete_dataset_from_studio(test_session, studio_token, requests_mock, force):
    requests_mock.delete(f"{STUDIO_URL}/api/datachain/datasets", json={"ok": True})
    dc.delete_dataset(
        "dev.animals.cats",
        version="1.0.0",
        force=force,
        studio=True,
        session=test_session,
    )


@pytest.mark.parametrize("is_studio", (False,))
@skip_if_not_sqlite
def test_delete_dataset_from_studio_not_found(
    test_session, studio_token, requests_mock
):
    error_message = "Dataset cats not found"
    requests_mock.delete(
        f"{STUDIO_URL}/api/datachain/datasets",
        json={"message": error_message},
        status_code=404,
    )
    with pytest.raises(Exception) as exc_info:
        dc.delete_dataset(
            "dev.animals.cats", version="1.0.0", studio=True, session=test_session
        )

    assert str(exc_info.value) == error_message


def test_delete_dataset_cached_from_studio(
    test_session, project, studio_token, requests_mock
):
    ds_full_name = f"{project.namespace.name}.{project.name}.fibonacci"
    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save(ds_full_name)

    error_message = f"Dataset {ds_full_name} not found"
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/info",
        json={"message": error_message},
        status_code=404,
    )

    dc.delete_dataset(ds_full_name)

    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset(name=ds_full_name)


@pytest.mark.parametrize(
    "update_version,versions",
    [
        ("patch", ["1.0.0", "1.0.1", "1.0.2"]),
        ("minor", ["1.0.0", "1.1.0", "1.2.0"]),
        ("major", ["1.0.0", "2.0.0", "3.0.0"]),
    ],
)
def test_update_versions(test_session, update_version, versions):
    ds_name = "fibonacci"
    chain = dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session)
    chain.save(ds_name, update_version=update_version)
    chain.save(ds_name, update_version=update_version)
    chain.save(ds_name, update_version=update_version)
    assert sorted(
        [
            ds.version
            for ds in dc.datasets(column="dataset", session=test_session).to_values(
                "dataset"
            )
        ]
    ) == sorted(versions)


def test_update_versions_mix_major_minor_patch(test_session):
    ds_name = "fibonacci"
    chain = dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session)
    chain.save(ds_name)
    chain.save(ds_name, update_version="patch")
    chain.save(ds_name, update_version="minor")
    chain.save(ds_name, update_version="major")
    chain.save(ds_name, update_version="minor")
    chain.save(ds_name, update_version="patch")
    chain.save(ds_name)
    chain.save(ds_name, version="3.0.0")
    assert sorted(
        [
            ds.version
            for ds in dc.datasets(column="dataset", session=test_session).to_values(
                "dataset"
            )
        ]
    ) == sorted(
        [
            "1.0.0",
            "1.0.1",
            "1.1.0",
            "2.0.0",
            "2.1.0",
            "2.1.1",
            "2.1.2",
            "3.0.0",
        ]
    )


def test_update_versions_wrong_value(test_session):
    ds_name = "fibonacci"
    chain = dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session)
    chain.save(ds_name)
    with pytest.raises(ValueError) as excinfo:
        chain.save(ds_name, update_version="wrong")

    assert str(excinfo.value) == (
        "update_version must be one of ['major', 'minor', 'patch']"
    )


def test_from_dataset_version_int_backward_compatible(test_session):
    ds_name = "numbers"
    dc.read_values(nums=[1], session=test_session).save(ds_name, version="1.0.0")
    dc.read_values(nums=[2], session=test_session).save(ds_name, version="1.0.1")
    dc.read_values(nums=[3], session=test_session).save(ds_name, version="2.0.0")
    dc.read_values(nums=[4], session=test_session).save(ds_name, version="2.1.0")
    dc.read_values(nums=[5], session=test_session).save(ds_name, version="2.1.2")
    dc.read_values(nums=[6], session=test_session).save(ds_name, version="3.0.0")

    assert dc.read_dataset(ds_name, version=1).to_values("nums") == [2]
    assert dc.read_dataset(ds_name, version=2).to_values("nums") == [5]
    assert dc.read_dataset(ds_name, version=3).to_values("nums") == [6]
    assert dc.read_dataset(ds_name, version="1.0.0").to_values("nums") == [1]
    with pytest.raises(DatasetVersionNotFoundError):
        dc.read_dataset(ds_name, version=5)


def test_wrong_semver_format(test_session):
    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save("fibonacci")
    with pytest.raises(ValueError) as excinfo:
        dc.read_dataset("fibonacci").save("fibonacci", version="1.0")
    assert str(excinfo.value) == (
        "Invalid version. It should be in format: <major>.<minor>.<patch> where"
        " each version part is positive integer"
    )


def test_semver_preview_ok(test_session):
    ds_name = "numbers"
    dc.read_values(num=[1, 2], session=test_session).save(ds_name)
    dc.read_values(num=[3, 4], session=test_session).save(ds_name)

    dataset = test_session.catalog.get_dataset(ds_name)
    assert sorted([p["num"] for p in dataset.get_version("1.0.0").preview]) == [1, 2]
    assert sorted([p["num"] for p in dataset.get_version("1.0.1").preview]) == [3, 4]


@pytest.mark.parametrize("is_studio", [True, False])
def test_save_to_default_project(test_session, is_studio):
    catalog = test_session.catalog
    ds_name = "fibonacci"
    dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save(ds_name)
    ds = dc.read_dataset(name=ds_name)
    assert ds.dataset.project == catalog.metastore.default_project


@pytest.mark.parametrize("is_studio", [True, False])
def test_save_to_default_project_with_read_storage(tmp_dir, test_session, is_studio):
    catalog = test_session.catalog
    ds_name = "parquet_ds"

    df = pd.DataFrame(DF_DATA)
    df.to_parquet(tmp_dir / "df.parquet")

    uri = tmp_dir.as_uri()
    dc.read_storage(uri, session=test_session).save(ds_name)

    ds = dc.read_dataset(name=ds_name)
    assert ds.dataset.project == catalog.metastore.default_project


@pytest.mark.parametrize("use_settings", (False,))
@pytest.mark.parametrize("project_created_upfront", (False,))
def test_save_to_non_default_namespace_and_project(
    test_session, use_settings, project_created_upfront
):
    catalog = test_session.catalog
    if project_created_upfront:
        catalog.metastore.create_project("dev", "numbers")

    ds = dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session)
    if use_settings:
        ds = ds.settings(namespace="dev", project="numbers").save("fibonacci")
    else:
        ds = ds.save("dev.numbers.fibonacci")

    ds = dc.read_dataset(name="dev.numbers.fibonacci")
    assert ds.dataset.project == catalog.metastore.get_project("numbers", "dev")
    assert ds.dataset.name == "fibonacci"
    assert ds.dataset.full_name == "dev.numbers.fibonacci"

    with pytest.raises(DatasetNotFoundError):
        # dataset is not in default namespace / project
        dc.read_dataset(name="fibonacci")


def test_dataset_not_found_in_default_project(test_session):
    metastore = test_session.catalog.metastore
    with pytest.raises(DatasetNotFoundError) as excinfo:
        dc.read_dataset("fibonacci")
    assert str(excinfo.value) == (
        f"Dataset fibonacci not found in namespace {metastore.default_namespace_name}"
        f" and project {metastore.default_project_name}"
    )


@pytest.mark.parametrize("project_created", (True, False))
def test_dataset_not_found_in_non_default_project(test_session, project_created):
    if project_created:
        dc.create_project("dev", "numbers")
    with pytest.raises(DatasetNotFoundError) as excinfo:
        dc.read_dataset("dev.numbers.fibonacci")
    assert str(excinfo.value) == (
        "Dataset fibonacci not found in namespace dev and project numbers"
    )


@pytest.mark.parametrize("use_settings", (True, False))
@pytest.mark.parametrize("project_created_upfront", (True, False))
def test_save_specify_only_non_default_project(
    test_session, use_settings, project_created_upfront
):
    catalog = test_session.catalog
    default_namespace_name = catalog.metastore.default_namespace_name

    if project_created_upfront:
        catalog.metastore.create_project(
            default_namespace_name, "numbers", validate=False
        )

    ds = dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session)
    if use_settings:
        ds = ds.settings(project="numbers").save("fibonacci")
    else:
        ds = ds.save("numbers.fibonacci")

    ds = dc.read_dataset(name="numbers.fibonacci")
    assert ds.dataset.project == catalog.metastore.get_project(
        "numbers", default_namespace_name
    )
    assert ds.dataset.name == "fibonacci"
    assert ds.dataset.full_name == (f"{default_namespace_name}.numbers.fibonacci")

    with pytest.raises(DatasetNotFoundError):
        # dataset is not in default namespace / project
        dc.read_dataset(name="fibonacci")


@pytest.mark.parametrize(
    (
        "ds_name_namespace,ds_name_project,"
        "settings_namespace,settings_project,"
        "env_namespace,env_project,"
        "result_ds_namespace,result_ds_project"
    ),
    [
        ("n3", "p3", "n2", "p2", "n1", "p1", "n3", "p3"),
        ("", "", "n2", "p2", "n1", "p1", "n2", "p2"),
        ("", "", "", "", "n1", "p1", "n1", "p1"),
        ("", "", "", "", "n5", "n1.p1", "n1", "p1"),
        ("", "", "", "", "", "n1.p1", "n1", "p1"),
        ("", "", "", "", "", "n5.p5", "n5", "p5"),
        ("n3", "p3", "n2", "p2", "", "", "n3", "p3"),
        ("n3", "p3", "", "", "", "", "n3", "p3"),
        ("n3", "p3", "", "", "n1", "p1", "n3", "p3"),
        ("", "", "", "", "", "", "", ""),
    ],
)
def test_save_all_ways_to_set_project(
    test_session,
    monkeypatch,
    ds_name_namespace,
    ds_name_project,
    settings_namespace,
    settings_project,
    env_namespace,
    env_project,
    result_ds_namespace,
    result_ds_project,
):
    def _full_name(namespace, project, name) -> str:
        if namespace and project:
            return f"{namespace}.{project}.{name}"
        return name

    metastore = test_session.catalog.metastore
    ds_name = "numbers"

    monkeypatch.setenv("DATACHAIN_NAMESPACE", env_namespace)
    monkeypatch.setenv("DATACHAIN_PROJECT", env_project)

    if not result_ds_namespace and not result_ds_project:
        # special case when nothing is defined - we set default ones
        result_ds_namespace = metastore.default_namespace_name
        result_ds_project = metastore.default_project_name

    ds = (
        dc.read_values(num=[1, 2, 3, 4], session=test_session)
        .settings(namespace=settings_namespace, project=settings_project)
        .save(_full_name(ds_name_namespace, ds_name_project, ds_name))
    )

    assert ds.dataset.project == metastore.get_project(
        result_ds_project, result_ds_namespace
    )
    dc.read_dataset(_full_name(result_ds_namespace, result_ds_project, ds_name))


@pytest.mark.parametrize(
    (
        "ds_name_namespace,ds_name_project,"
        "settings_namespace,settings_project,"
        "env_namespace,env_project,"
        "error"
    ),
    [
        ("n3.n3", "p3", "n2", "p2", "n1", "p1", InvalidDatasetNameError),
        ("n3", "p3.p3", "n2", "p2", "n1", "p1", InvalidDatasetNameError),
        ("", "", "n2.n2", "p2", "n1", "p1", InvalidNamespaceNameError),
        ("", "", "n2", "p2.p2", "n1", "p1", InvalidProjectNameError),
        ("", "", "", "", "n1.n1", "p1", InvalidNamespaceNameError),
        ("", "", "", "", "n1", "p1.p1.p1", InvalidProjectNameError),
    ],
)
def test_save_all_ways_to_set_project_invalid_name(
    test_session,
    monkeypatch,
    ds_name_namespace,
    ds_name_project,
    settings_namespace,
    settings_project,
    env_namespace,
    env_project,
    error,
):
    def _full_name(namespace, project, name) -> str:
        if namespace and project:
            return f"{namespace}.{project}.{name}"
        return name

    ds_name = "numbers"

    monkeypatch.setenv("DATACHAIN_NAMESPACE", env_namespace)
    monkeypatch.setenv("DATACHAIN_PROJECT", env_project)

    with pytest.raises(error):
        (
            dc.read_values(num=[1, 2, 3, 4], session=test_session)
            .settings(namespace=settings_namespace, project=settings_project)
            .save(_full_name(ds_name_namespace, ds_name_project, ds_name))
        )


@pytest.mark.parametrize("is_studio", [False])
@skip_if_not_sqlite
def test_save_create_project_not_allowed(test_session, is_studio):
    with pytest.raises(ProjectCreateNotAllowedError):
        dc.read_values(fib=[1, 1, 2, 3, 5, 8], session=test_session).save(
            "dev.numbers.fibonacci"
        )


def test_agg_partition_by_string_notation(test_session):
    """Test that agg method supports string notation for partition_by."""

    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = "-".join(key)
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    keys = ["n1", "n2", "n1"]
    values = [1, 5, 9]

    # Test using string notation (NEW functionality)
    ds = dc.read_values(key=keys, val=values, session=test_session).agg(
        x=func,
        partition_by="key",  # String notation instead of C("key")
    )

    assert ds.order_by("x_1.name").to_values("x_1.name") == ["n1-n1", "n2"]
    assert ds.order_by("x_1.size").to_values("x_1.size") == [5, 10]


def test_agg_partition_by_string_sequence(test_session):
    """Test that agg method supports sequence of strings for partition_by."""

    class _ImageGroup(BaseModel):
        name: str
        size: int

    def func(key1, key2, val) -> Iterator[tuple[File, _ImageGroup]]:
        n = f"{key1[0]}-{key2[0]}"
        v = sum(val)
        yield File(path=n), _ImageGroup(name=n, size=v)

    key1_values = ["a", "a", "b"]
    key2_values = ["x", "y", "x"]
    values = [1, 5, 9]

    # Test using sequence of strings (NEW functionality)
    ds = dc.read_values(
        key1=key1_values, key2=key2_values, val=values, session=test_session
    ).agg(
        x=func,
        partition_by=["key1", "key2"],  # Sequence of strings
    )

    result_names = ds.order_by("x_1.name").to_values("x_1.name")
    result_sizes = ds.order_by("x_1.size").to_values("x_1.size")

    # Should have 3 partitions: (a,x), (a,y), (b,x)
    assert len(result_names) == 3
    assert len(result_sizes) == 3


def test_column_compute(test_session):
    """Test that sum, avg, min and max chain functions works correctly."""

    class Signal1(DataModel):
        i3: int
        f3: float
        s3: str

    class Signal2(DataModel):
        i2: int
        f2: float
        s2: str
        signal: Signal1

    i1 = [1, 2, 3, 4, 5]
    f1 = [0.5, 1.0, 1.5, 2.0, 2.5]
    s1 = ["a", "b", "c", "d", "e"]
    signals = [
        Signal2(
            i2=i * 2,
            f2=f * 2,
            s2=s * 2,
            signal=Signal1(
                i3=i * 3,
                f3=f * 3.0,
                s3=s * 3,
            ),
        )
        for i, f, s in zip(i1, f1, s1, strict=False)
    ]

    chain = dc.read_values(
        i1=i1,
        f1=f1,
        s1=s1,
        signals=signals,
        session=test_session,
    )

    assert chain.sum("i1") == 15
    assert chain.sum("f1") == 7.5
    assert chain.sum("signals.i2") == 30
    assert chain.sum("signals.f2") == 15.0
    assert chain.sum("signals.signal.i3") == 45
    assert chain.sum("signals.signal.f3") == 22.5

    assert chain.avg("i1") == 3
    assert chain.avg("f1") == 1.5
    assert chain.avg("signals.i2") == 6
    assert chain.avg("signals.f2") == 3.0
    assert chain.avg("signals.signal.i3") == 9
    assert chain.avg("signals.signal.f3") == 4.5

    assert chain.min("i1") == 1
    assert chain.min("f1") == 0.5
    assert chain.min("s1") == "a"
    assert chain.min("signals.i2") == 2
    assert chain.min("signals.f2") == 1.0
    assert chain.min("signals.s2") == "aa"
    assert chain.min("signals.signal.i3") == 3
    assert chain.min("signals.signal.f3") == 1.5
    assert chain.min("signals.signal.s3") == "aaa"

    assert chain.max("i1") == 5
    assert chain.max("f1") == 2.5
    assert chain.max("s1") == "e"
    assert chain.max("signals.i2") == 10
    assert chain.max("signals.f2") == 5.0
    assert chain.max("signals.s2") == "ee"
    assert chain.max("signals.signal.i3") == 15
    assert chain.max("signals.signal.f3") == 7.5
    assert chain.max("signals.signal.s3") == "eee"


def test_union_does_not_break_schema_order(test_session):
    class Meta(BaseModel):
        name: str
        count: int

    def add_file(key) -> File:
        return File(path="")

    def add_meta(file) -> Meta:
        return Meta(name="meta", count=10)

    keys = ["a", "b", "c", "d"]
    values = [3, 3, 3, 3]

    (
        dc.read_values(key=keys, val=values, session=test_session)
        .map(file=add_file)
        .map(meta=add_meta)
        .save("ds1")
    )
    (
        dc.read_values(key=keys, val=values, session=test_session)
        .map(file=add_file)
        .map(meta=add_meta)
        .save("ds2")
    )

    (
        dc.read_dataset("ds1", session=test_session)
        .union(dc.read_dataset("ds2", session=test_session))
        .save("union")
    )

    dat = test_session.catalog.get_dataset("union")
    assert list(dat.versions[0].schema.keys()) == [
        "key",
        "val",
        "file__source",
        "file__path",
        "file__size",
        "file__version",
        "file__etag",
        "file__is_latest",
        "file__last_modified",
        "file__location",
        "meta__name",
        "meta__count",
    ]
