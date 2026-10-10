"""Picking, typing and naming the columns of tabular files."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel

import datachain as dc
from datachain.lib.dc import DatasetPrepareError

MESSY = (
    "Customer First Name,Unit Price (USD),ZIP Code,Notes\n"
    "Alice,19.99,02134,first\n"
    "Bob,5,10001,\n"
)
NO_HEADER = "Alice,19.99,02134,first\nBob,5,10001,\n"


@pytest.fixture
def messy(tmp_dir):
    path = tmp_dir / "messy.csv"
    path.write_text(MESSY)
    return path.as_uri()


@pytest.fixture
def no_header(tmp_dir):
    path = tmp_dir / "no_header.csv"
    path.write_text(NO_HEADER)
    return path.as_uri()


@pytest.fixture
def abc_parquet(tmp_dir):
    path = tmp_dir / "abc.parquet"
    pq.write_table(
        pa.table({"a": [1.0, 2.0], "b": [10.0, 20.0], "c": [100.0, 200.0]}), path
    )
    return path.as_uri()


def signals(chain):
    return [name for name in chain.schema if name != "source"]


def test_read_csv_columns(messy, test_session):
    chain = dc.read_csv(
        messy, columns=["zip_code", "Customer First Name"], session=test_session
    )
    assert signals(chain) == ["zip_code", "customer_first_name"]
    assert sorted(chain.to_list("customer_first_name", "zip_code")) == [
        ("Alice", 2134),
        ("Bob", 10001),
    ]


def test_read_csv_columns_not_found(messy, test_session):
    with pytest.raises(DatasetPrepareError, match=r"Columns \['zip'\] not found"):
        dc.read_csv(messy, columns=["zip"], session=test_session)


def test_read_csv_output_and_columns(messy, test_session):
    with pytest.raises(DatasetPrepareError, match="either output or columns"):
        dc.read_csv(
            messy, output=["zip_code"], columns=["zip_code"], session=test_session
        )


def test_read_csv_column_types(messy, test_session):
    chain = dc.read_csv(messy, column_types={"zip_code": str}, session=test_session)
    assert signals(chain) == [
        "customer_first_name",
        "unit_price_usd",
        "zip_code",
        "notes",
    ]
    assert sorted(chain.to_values("zip_code")) == ["02134", "10001"]


def test_read_csv_column_types_with_columns(messy, test_session):
    chain = dc.read_csv(
        messy,
        columns=["ZIP Code"],
        column_types={"ZIP Code": "string"},
        session=test_session,
    )
    assert signals(chain) == ["zip_code"]
    assert sorted(chain.to_values("zip_code")) == ["02134", "10001"]


def test_read_csv_column_types_not_found(messy, test_session):
    with pytest.warns(UserWarning, match=r"column_types names \['zip'\]"):
        chain = dc.read_csv(messy, column_types={"zip": str}, session=test_session)
    assert sorted(chain.to_values("zip_code")) == [2134, 10001]


def test_read_csv_output_types_while_parsing(messy, test_session):
    chain = dc.read_csv(
        messy,
        output={"zip_code": str, "Unit Price (USD)": float},
        session=test_session,
    )
    assert signals(chain) == ["zip_code", "unit_price_usd"]
    assert sorted(chain.to_list("zip_code", "unit_price_usd")) == [
        ("02134", 19.99),
        ("10001", 5.0),
    ]


def test_read_csv_output_model(messy, test_session):
    class Order(BaseModel):
        zip_code: str
        customer_first_name: str

    chain = dc.read_csv(messy, output=Order, session=test_session)
    assert sorted(chain.to_list("customer_first_name", "zip_code")) == [
        ("Alice", "02134"),
        ("Bob", "10001"),
    ]


def test_read_csv_output_with_column_types(messy, test_session):
    chain = dc.read_csv(
        messy,
        output={"ZIP Code": str},
        column_types={"ZIP Code": "string"},
        session=test_session,
    )
    assert sorted(chain.to_values("zip_code")) == ["02134", "10001"]


def test_read_csv_output_same_column_twice(messy, test_session):
    with pytest.raises(DatasetPrepareError, match="same column twice"):
        dc.read_csv(
            messy, output={"ZIP Code": str, "zip_code": str}, session=test_session
        )


def test_read_csv_output_empty_header_name(tmp_dir, test_session):
    # pandas writes its index under an empty header, which DataChain names c0.
    path = tmp_dir / "indexed.csv"
    path.write_text(",a\n0,1\n1,2\n")
    chain = dc.read_csv(
        path.as_uri(), output={"c0": int, "a": int}, session=test_session
    )
    assert sorted(chain.to_list("c0", "a")) == [(0, 1), (1, 2)]


def test_read_csv_column_names(messy, test_session):
    chain = dc.read_csv(
        messy, column_names=["name", "price", "zip", "notes"], session=test_session
    )
    assert signals(chain) == ["name", "price", "zip", "notes"]
    assert sorted(chain.to_list("name", "price")) == [("Alice", 19.99), ("Bob", 5.0)]


def test_read_csv_column_names_wrong_length(messy, test_session):
    with pytest.raises(DatasetPrepareError, match="Expected 2 columns, got 4"):
        dc.read_csv(messy, column_names=["name", "price"], session=test_session)


def test_read_csv_no_header_column_names(no_header, test_session):
    chain = dc.read_csv(
        no_header,
        header=False,
        column_names=["name", "price", "zip", "notes"],
        columns=["zip", "name"],
        column_types={"zip": str},
        session=test_session,
    )
    assert signals(chain) == ["zip", "name"]
    assert sorted(chain.to_list("zip", "name")) == [
        ("02134", "Alice"),
        ("10001", "Bob"),
    ]


def test_read_csv_no_header_generated_names_output(no_header, test_session):
    chain = dc.read_csv(
        no_header, header=False, output={"f2": str, "f0": str}, session=test_session
    )
    assert sorted(chain.to_list("f0", "f2")) == [("Alice", "02134"), ("Bob", "10001")]


def test_read_parquet_columns_and_column_types(abc_parquet, test_session):
    chain = dc.read_parquet(
        abc_parquet, columns=["c", "a"], column_types={"a": int}, session=test_session
    )
    assert signals(chain) == ["c", "a"]
    assert sorted(chain.to_list("c", "a")) == [(100.0, 1), (200.0, 2)]


def test_read_parquet_output_with_column_types(abc_parquet, test_session):
    chain = dc.read_parquet(
        abc_parquet,
        output={"c": float, "a": int},
        column_types={"a": "int64"},
        session=test_session,
    )
    assert sorted(chain.to_list("c", "a")) == [(100.0, 1), (200.0, 2)]


def test_read_parquet_written_by_datachain_output_list(tmp_dir, test_session):
    path = tmp_dir / "exported.parquet"
    dc.read_values(x=[1, 2], y=["a", "b"], session=test_session).to_parquet(path)
    chain = dc.read_parquet(path.as_uri(), output=["y"], session=test_session)
    assert signals(chain) == ["y"]
    assert sorted(chain.to_values("y")) == ["a", "b"]


def test_read_csv_output_list_renames_every_column_deprecated(tmp_dir, test_session):
    path = tmp_dir / "ab.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.warns(FutureWarning, match="by position"):
        chain = dc.read_csv(path.as_uri(), output=["b", "a"], session=test_session)
    assert chain.to_list("b", "a") == [(1, 2)]


@pytest.mark.parametrize("output", [["a", "b"], ["b"]])
def test_read_csv_output_list_selects(tmp_dir, test_session, output):
    path = tmp_dir / "ab.csv"
    path.write_text("a,b\n1,2\n")
    chain = dc.read_csv(path.as_uri(), output=output, session=test_session)
    assert signals(chain) == output
    assert chain.to_list(*output) == [tuple({"a": 1, "b": 2}[n] for n in output)]


def test_read_parquet_written_by_datachain_output_list_unknown(tmp_dir, test_session):
    path = tmp_dir / "exported.parquet"
    dc.read_values(x=[1], y=[2], session=test_session).to_parquet(path)
    with pytest.raises(DatasetPrepareError, match="not signals of the file"):
        dc.read_parquet(path.as_uri(), output=["a", "b"], session=test_session)


def test_read_parquet_column_types_file_lacks_column(tmp_dir, test_session):
    pq.write_table(pa.table({"a": [1]}), tmp_dir / "1.parquet")
    pq.write_table(pa.table({"b": [2, 3]}), tmp_dir / "2.parquet")
    chain = dc.read_parquet(
        tmp_dir.as_uri(),
        output={"a": float},
        column_types={"a": "float64"},
        session=test_session,
    )
    assert sorted(chain.to_values("a"), key=str) == [1.0, None, None]


def test_read_csv_column_types_files_infer_differently(tmp_dir, test_session):
    (tmp_dir / "1.csv").write_text("id\n001\n")
    (tmp_dir / "2.csv").write_text("id\nABC\n")
    chain = dc.read_csv(
        tmp_dir.as_uri(), column_types={"id": "string"}, session=test_session
    )
    assert sorted(chain.to_values("id")) == ["001", "ABC"]


def test_read_parquet_columns_ignore_unselected_types(tmp_dir, test_session):
    pq.write_table(
        pa.table({"a": [7], "d": pa.array([1], pa.duration("s"))}),
        tmp_dir / "1.parquet",
    )
    pq.write_table(pa.table({"a": [8], "d": ["text"]}), tmp_dir / "2.parquet")
    chain = dc.read_parquet(tmp_dir.as_uri(), columns=["a"], session=test_session)
    assert sorted(chain.to_values("a")) == [7, 8]


@pytest.mark.parametrize("writer", ["datachain", "huggingface"])
def test_read_parquet_columns_ignore_unselected_types_with_metadata(
    tmp_dir, test_session, writer
):
    rows = [{"a": 1, "other": 123}, {"a": 2, "other": "text"}]
    for i, row in enumerate(rows):
        path = tmp_dir / f"{i}.parquet"
        if writer == "datachain":
            values = {k: [v] for k, v in row.items()}
            dc.read_values(**values, session=test_session).to_parquet(path)
        else:
            from datasets import Dataset

            Dataset.from_dict({k: [v] for k, v in row.items()}).to_parquet(path)
    chain = dc.read_parquet(tmp_dir.as_uri(), columns=["a"], session=test_session)
    assert signals(chain) == ["a"]
    assert sorted(chain.to_values("a")) == [1, 2]
