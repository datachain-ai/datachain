import importlib

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, load_dataset
from datasets.features.image import image_to_bytes
from PIL import Image
from scipy.io.wavfile import write

import datachain as dc
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.hf import HFGenerator, HFImage, get_output_schema
from tests.utils import df_equal

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
}


def require_torchcodec(test_case):
    """
    Decorator marking a test that requires torchcodec (not available on Windows).
    These tests are skipped when torchcodec isn't installed.
    """
    if not importlib.util.find_spec("torchcodec"):
        test_case = pytest.mark.skip(
            "test requires torchcodec, not available on Windows yet"
        )(test_case)
    return test_case


def test_read_hf(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = dc.read_hf(ds, session=test_session).to_pandas()
    assert df_equal(df, pd.DataFrame(DF_DATA))


def test_read_hf_column(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = dc.read_hf(ds, session=test_session, column="obj").to_pandas()
    assert df_equal(df["obj"], pd.DataFrame(DF_DATA))


def test_read_hf_invalid(test_session):
    with pytest.raises(FileNotFoundError):
        dc.read_hf("invalid_dataset", session=test_session)


def test_read_hf_limit(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = dc.read_hf(ds, session=test_session, limit=1).to_pandas()
    assert len(df) == 1


def test_read_hf_nested_data(test_session):
    ds = Dataset.from_dict(
        {
            "user": [
                {
                    "name": "Alice",
                    "middle name": "Aurora",
                    "age": None,
                    "location": {"city": "New York", "state": "NY"},
                },
                {
                    "name": "Bob",
                    "middle name": None,
                    "age": 30,
                    "location": {"city": "Los Angeles", "state": "CA"},
                },
            ]
        }
    )
    schema = dc.read_hf(ds, session=test_session).schema
    anno = schema["user"]
    fields = anno.model_fields
    assert len(fields) == 4
    assert fields["name"].annotation == str | None
    assert fields["middle_name"].annotation == str | None
    assert fields["age"].annotation == int | None

    # Middle name fields is normalized and has an alias
    assert len(anno._model_fields_by_aliases()) == 5

    location_fields = fields["location"].annotation.model_fields
    assert len(location_fields) == 2
    assert location_fields["city"].annotation == str | None
    assert location_fields["state"].annotation == str | None


def test_read_hf_streaming(test_session):
    ds = Dataset.from_dict(DF_DATA)
    df = dc.read_hf(ds, session=test_session, streaming=True).to_pandas()
    assert df_equal(df, pd.DataFrame(DF_DATA))


def test_hf_image(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    img = Image.new(mode="RGB", size=(64, 64))
    img.save(train_dir / "img1.png")

    ds = load_dataset("imagefolder", data_dir=tmp_path)
    hf_schema, norm_names = get_output_schema(ds["train"].features, ["split"])
    schema = {"split": str} | hf_schema
    assert schema["image"] is HFImage

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process("train")))
    assert row.image.img == image_to_bytes(img)


@require_torchcodec
def test_hf_audio(tmp_path):
    # See https://stackoverflow.com/questions/66191480/how-to-convert-a-numpy-array-to-a-mp3-file
    samplerate = 44100
    fs = 100
    t = np.linspace(0.0, 1.0, samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np.sin(2.0 * np.pi * fs * t)
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    write(train_dir / "example.wav", samplerate, data.astype(np.int16))

    ds = load_dataset("audiofolder", data_dir=tmp_path)
    hf_schema, norm_names = get_output_schema(ds["train"].features, ["split"])
    schema = {"split": str} | hf_schema

    gen = HFGenerator(ds, dict_to_data_model("", schema, list(norm_names.values())))
    gen.setup()
    row = next(iter(gen.process("train")))
    assert np.allclose(row.audio.array, data / amplitude, atol=1e-4)
    assert row.audio.sampling_rate == samplerate
