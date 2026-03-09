import csv
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel

import datachain as dc
from datachain.lib.file import File


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


def test_to_csv_features(tmp_dir, test_session):
    dc_to = dc.read_values(f1=features, num=range(len(features)), session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.order_by("f1.nnn", "f1.count").to_csv(path)
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == ["f1.nnn,f1.count,num", "n1,1,0", "n1,3,1", "n2,5,2", ""]


def test_to_tsv_features(tmp_dir, test_session):
    dc_to = dc.read_values(f1=features, num=range(len(features)), session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.order_by("f1.nnn", "f1.count").to_csv(path, delimiter="\t")
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == ["f1.nnn\tf1.count\tnum", "n1\t1\t0", "n1\t3\t1", "n2\t5\t2", ""]


def test_to_csv_features_nested(tmp_dir, test_session):
    dc_to = dc.read_values(sign1=features_nested, session=test_session)
    path = tmp_dir / "test.csv"
    dc_to.order_by("sign1.fr.nnn", "sign1.fr.count").to_csv(path)
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines == [
        "sign1.label,sign1.fr.nnn,sign1.fr.count",
        "label_0,n1,1",
        "label_1,n1,3",
        "label_2,n2,5",
        "",
    ]


def test_to_csv_returns_file(tmp_dir, test_session):
    chain = dc.read_values(value=[1, 2], session=test_session)
    dest = tmp_dir / "values.csv"

    stored = chain.order_by("value").to_csv(dest)

    assert isinstance(stored, File)
    assert stored.source.startswith("file://")
    assert stored.path == dest.name
    assert stored.size == dest.stat().st_size

    with open(dest, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["value"]
    assert rows[1:] == [["1"], ["2"]]


def test_to_csv_relative_path(tmp_dir, test_session, monkeypatch):
    monkeypatch.chdir(tmp_dir)

    chain = dc.read_values(value=[3, 4], session=test_session)
    dest = Path("rel.csv")

    stored = chain.to_csv(dest)

    assert stored.source == tmp_dir.as_uri()
    assert stored.path == dest.name

    output_path = tmp_dir / dest
    assert output_path.exists()
    assert stored.size == output_path.stat().st_size

    with open(output_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["value"]
    assert sorted(rows[1:]) == [["3"], ["4"]]


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_to_csv_returns_file_with_version(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload

    chain = dc.read_values(value=[5, 6], session=ctc.session)
    dest = f"{ctc.src_uri}/to-csv-{uuid.uuid4().hex}.csv"

    stored = chain.order_by("value").to_csv(dest)

    assert isinstance(stored, File)
    assert stored.version
    assert stored.source == ctc.src_uri
    assert stored.path.endswith(".csv")

    parsed = list(csv.reader(stored.read_text().splitlines()))
    assert parsed[0] == ["value"]
    assert parsed[1:] == [["5"], ["6"]]


def test_to_csv_custom_delimiter(tmp_dir, test_session):
    chain = dc.read_values(a=["x", "y"], b=[1, 2], session=test_session)
    dest = tmp_dir / "values.tsv"

    stored = chain.order_by("a").to_csv(dest, delimiter="\t")

    assert isinstance(stored, File)
    with open(dest, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    assert rows[0] == ["a", "b"]
    assert rows[1:] == [["x", "1"], ["y", "2"]]
