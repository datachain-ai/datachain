import json

import pytest

from datachain.lib.file import TextFile
from datachain.lib.meta_formats import gen_datamodel_code, read_meta

example = {
    "id": "1",
    "split": "test",
    "image_id": {
        "author": "author",
        "title": "title",
        "size": 5090109,
        "md5": "md5",
        "url": "https://example.org/image.jpg",
        "rotation": 0.0,
    },
    "classifications": [
        {"Source": "verification", "LabelName": "label", "Confidence": 0}
    ],
}


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_gen_datamodel_code(tmp_dir, catalog):
    (tmp_dir / "valid.json").write_text(json.dumps(example), encoding="utf8")
    file = TextFile(path=tmp_dir / "valid.json")
    file._set_stream(catalog)

    expected = """\
from datachain.lib.data_model import DataModel
from datachain.lib.meta_formats import UserModel


class ImageId(UserModel):
    author: str
    title: str
    size: int
    md5: str
    url: str
    rotation: float


class Classification(UserModel):
    Source: str
    LabelName: str
    Confidence: int


class Image(UserModel):
    id: str
    split: str
    image_id: ImageId
    classifications: list[Classification]

DataModel.register(Image)
spec = Image"""

    actual = gen_datamodel_code(file, format="json", model_name="Image")
    actual = "\n".join(actual.splitlines()[4:])  # remove header
    assert actual == expected


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_read_meta(tmp_dir, catalog):
    (tmp_dir / "valid.json").write_text(json.dumps(example), encoding="utf8")
    file = TextFile(path=tmp_dir / "valid.json")
    file._set_stream(catalog)

    parser = read_meta(
        schema_from=str(tmp_dir / "valid.json"),
        format="jsonl",
        model_name="Image",
    )
    rows = list(parser(file))
    assert len(rows) == 1
    assert rows[0].model_dump() == example

    (tmp_dir / "invalid.json").write_text(
        json.dumps({"hello": "world"}), encoding="utf8"
    )
    invalid_file = TextFile(path=tmp_dir / "invalid.json")
    invalid_file._set_stream(catalog)
    assert not list(parser(invalid_file))
