import pytest
from datasets import (
    Array2D,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    List,
    Sequence,
    Value,
)

import datachain as dc
from datachain.lib.data_model import DataModel, dict_to_data_model
from datachain.lib.hf import (
    HFClassLabel,
    HFGenerator,
    get_output_schema,
    stream_splits,
)


def test_hf():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    schema, norm_names = get_output_schema(ds.features)
    assert schema["pokemon"] is str

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon == "bulbasaur"


def test_hf_split():
    # Space in the column name should be normalized
    ds_train = Dataset.from_dict({"pok emon": ["bulbasaur", "squirtle"]})
    ds_test = Dataset.from_dict({"pok emon": ["charizard", "pikachu"]})
    ds_dict = DatasetDict({"train": ds_train, "test": ds_test})
    ds_dict = stream_splits(ds_dict)
    hf_schema, norm_names = get_output_schema(ds_dict["train"].features, ["split"])
    schema = {"split": str} | hf_schema

    gen = HFGenerator(
        ds_dict, dict_to_data_model("", schema, list(norm_names.values()))
    )
    gen.setup()
    row = next(iter(gen.process("train")))

    assert row.split == "train"
    assert row.pok_emon == "bulbasaur"


def test_hf_class_label():
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds = ds.class_encode_column("pokemon")
    schema, norm_names = get_output_schema(ds.features)
    assert schema["pokemon"] is HFClassLabel

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.string == "bulbasaur"
    assert row.pokemon.integer == 0


def test_hf_sequence_list():
    ds = Dataset.from_dict({"seq": [[0, 1], [2, 3]]})
    schema, norm_names = get_output_schema(ds.features)
    assert schema["seq"] == list[int]

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.seq == [0, 1]


def test_hf_sequence_dict():
    # ? in the column name should be normalized
    # Check if even nested names are not normalized we handle it correctly
    ds = Dataset.from_dict(
        {"pokemon": [{"name?": ["bulbasaur"]}, {"name?": ["squirtle"]}]}
    )
    new_features = ds.features.copy()
    new_features["pokemon"] = Sequence(feature={"name?": Value(dtype="string")})
    ds = ds.cast(new_features)
    schema, norm_names = get_output_schema(ds.features)
    assert schema["pokemon"].model_fields["name"].annotation == list[str] | None

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.pokemon.name == ["bulbasaur"]


def _first_row(ds):
    schema, norm_names = get_output_schema(ds.features)
    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    return next(iter(gen.process()))


def test_hf_list_of_class_labels():
    features = Features({"labels": List(ClassLabel(names=["cat", "dog"]))})
    ds = Dataset.from_dict({"labels": [[0, 1]]}, features=features)
    row = _first_row(ds)
    assert [(c.string, c.integer) for c in row.labels] == [("cat", 0), ("dog", 1)]


def test_read_hf_list_of_class_labels(test_session):
    labels = List(ClassLabel(names=["cat", "dog"]))
    features = Features({"id": Value("int64"), "labels": labels})
    ds = Dataset.from_dict({"id": [1], "labels": [[0, 1]]}, features=features)
    (labels,) = dc.read_hf(ds, session=test_session).to_values("labels")
    assert [c.string for c in labels] == ["cat", "dog"]


class Tag(DataModel):
    name: str


TAGS = ({"tags": List({"name": Value("string")})}, {"tags": [[{"name": "x"}]]})
LABELS = ({"labels": List(ClassLabel(names=["a", "b"]))}, {"labels": [[0, 1]]})
LABEL = ({"label": ClassLabel(names=["a", "b"])}, {"label": [1]})
META = ({"meta": {"name": Value("string")}}, {"meta": [{"name": "x"}]})


@pytest.mark.parametrize(
    "features,data,output,expected",
    [
        (*TAGS, list[dict], [{"name": "x"}]),
        (*TAGS, list[dict[str, str]], [{"name": "x"}]),
        (*TAGS, list[Tag], [Tag(name="x")]),
        (*LABELS, list[int], [0, 1]),
        (*LABELS, list[int] | None, [0, 1]),
        (*LABEL, int, 1),
        (*META, dict, {"name": "x"}),
    ],
)
def test_read_parquet_hf_explicit_output(
    tmp_path, test_session, features, data, output, expected
):
    (name,) = features
    path = tmp_path / "hf.parquet"
    Dataset.from_dict(data, features=Features(features)).to_parquet(path)
    chain = dc.read_parquet(path.as_uri(), output={name: output}, session=test_session)
    assert chain.to_values(name) == [expected]


def test_hf_list_of_class_labels_missing():
    features = Features({"labels": List(ClassLabel(names=["cat", "dog"]))})
    ds = Dataset.from_dict({"labels": [None]}, features=features)
    assert _first_row(ds).labels is None


def test_hf_list_of_dicts_with_class_label():
    tag = {"name": Value("string"), "kind": ClassLabel(names=["a", "b"])}
    features = Features({"tags": List(tag)})
    ds = Dataset.from_dict({"tags": [[{"name": "x", "kind": 1}]]}, features=features)
    row = _first_row(ds)
    assert [(t.name, t.kind.string) for t in row.tags] == [("x", "b")]


def test_hf_sequence_dict_with_class_label():
    tag = {"kind": ClassLabel(names=["a", "b"])}
    features = Features({"tags": Sequence(tag)})
    ds = Dataset.from_dict({"tags": [{"kind": [1, 0]}]}, features=features)
    row = _first_row(ds)
    assert [k.string for k in row.tags.kind] == ["b", "a"]


def test_read_parquet_hf_list_of_class_labels(tmp_path, test_session):
    features = Features({"labels": List(ClassLabel(names=["cat", "dog"]))})
    path = tmp_path / "labels.parquet"
    Dataset.from_dict({"labels": [[1, 0]]}, features=features).to_parquet(path)
    (labels,) = dc.read_parquet(path.as_uri(), session=test_session).to_values("labels")
    assert [c.string for c in labels] == ["dog", "cat"]


def test_hf_array():
    ds = Dataset.from_dict({"arr": [[[0, 1], [2, 3]]]})
    new_features = ds.features.copy()
    new_features["arr"] = Array2D(shape=(2, 2), dtype="int32")
    ds = ds.cast(new_features)
    schema, norm_names = get_output_schema(ds.features)
    assert schema["arr"] == list[list[int]]

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process()))
    assert row.arr == [[0, 1], [2, 3]]
