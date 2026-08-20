import pytest
from datasets import Array2D, Dataset, DatasetDict, Sequence, Value

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.hf import (
    HFClassLabel,
    HFGenerator,
    get_output_schema,
    stream_splits,
)


@pytest.mark.parametrize("as_dict", [False, True])
def test_hf_generator_constructor_hash(as_dict):
    first_ds = Dataset.from_dict({"value": [1]})
    second_ds = Dataset.from_dict({"value": [1]})
    if as_dict:
        first_ds = DatasetDict({"train": first_ds})
        second_ds = DatasetDict({"train": second_ds})
    first_schema = dict_to_data_model("", {"value": int})
    second_schema = dict_to_data_model("", {"value": int})

    first = HFGenerator(first_ds, first_schema)
    second = HFGenerator(second_ds, second_schema)
    limited = HFGenerator(second_ds, second_schema, limit=1)

    assert first._constructor_state_hash == second._constructor_state_hash
    assert first._constructor_state_hash != limited._constructor_state_hash


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
