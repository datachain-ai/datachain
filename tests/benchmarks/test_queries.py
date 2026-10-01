import itertools
import json

import pytest
from PIL import Image

import datachain as dc
from datachain import C, File
from datachain.func import array, path, string

NUM_FILES = 1000
NUM_IMAGES = 100
NUM_JSON_OBJECTS = 2000

pytestmark = pytest.mark.usefixtures("ignore_checkpoints")


@pytest.fixture
def files_uri(tmp_path, test_session):
    root = tmp_path / "files"
    root.mkdir()
    for i in range(NUM_FILES):
        kind = "dog" if i % 2 else "cat"
        ext = "jpg" if i % 4 else "txt"
        (root / f"{kind}.{i}.{ext}").write_text(str(i))

    uri = root.as_uri()
    dc.read_storage(uri, session=test_session).exec()
    return uri


@pytest.fixture
def images_uri(tmp_path, test_session):
    root = tmp_path / "images"
    root.mkdir()
    for i in range(NUM_IMAGES):
        Image.new("RGB", (16 + i % 8, 16), color=(i, i, i)).save(root / f"{i}.jpg")

    uri = root.as_uri()
    dc.read_storage(uri, session=test_session).exec()
    return uri


@pytest.fixture
def json_uri(tmp_path):
    images = [
        {
            "id": i,
            "file_name": f"{i:012}.jpg",
            "height": 480,
            "width": 640,
            "coco_url": f"http://images.cocodataset.org/val2017/{i:012}.jpg",
        }
        for i in range(NUM_JSON_OBJECTS)
    ]
    json_path = tmp_path / "captions.json"
    json_path.write_text(json.dumps({"images": images}))
    return json_path.as_uri()


def test_storage_listing(benchmark, files_uri, test_session):
    def run():
        return dc.read_storage(files_uri, update=True, session=test_session).count()

    assert benchmark(run) == NUM_FILES


def test_metadata_filter_to_pandas(benchmark, files_uri, test_session):
    def run():
        return (
            dc.read_storage(files_uri, session=test_session)
            .filter(C("file.path").glob("*.jpg"))
            .limit(500)
            .to_pandas()
        )

    assert len(benchmark(run)) == 500


def test_read_json(benchmark, json_uri, test_session):
    def run():
        return dc.read_json(json_uri, jmespath="images", session=test_session).count()

    assert benchmark(run) == NUM_JSON_OBJECTS


def test_sql_mutate(benchmark, files_uri, test_session):
    def run():
        parts = string.split(path.name(C("file.path")), ".")
        return (
            dc.read_storage(files_uri, session=test_session)
            .mutate(
                stem=path.file_stem(C("file.path")),
                ext=path.file_ext(C("file.path")),
                is_dog=array.contains(parts, "dog"),
            )
            .select("file.path", "stem", "ext", "is_dog")
            .count()
        )

    assert benchmark(run) == NUM_FILES


def test_map_save_read(benchmark, files_uri, test_session):
    def run():
        (
            dc.read_storage(files_uri, session=test_session)
            .filter(C("file.path").glob("*.jpg"))
            .map(path_len=len, params=["file.path"], output=int)
            .save("bench_map_save")
        )
        return dc.read_dataset("bench_map_save", session=test_session).count()

    assert benchmark(run) == NUM_FILES * 3 // 4


def test_image_udf(benchmark, images_uri, test_session):
    def pixel_count(file: File) -> int:
        with file.open() as fd, Image.open(fd) as img:
            width, height = img.size
            return width * height

    def run():
        return (
            dc.read_storage(images_uri, session=test_session)
            .map(pixels=pixel_count)
            .sum("pixels")
        )

    assert benchmark(run) > 0


def test_delta_cold(benchmark, files_uri, test_session):
    names = (f"bench_delta_cold_{i}" for i in itertools.count())

    def run():
        return (
            dc.read_storage(
                files_uri, delta=True, delta_on="file.path", session=test_session
            )
            .map(name_len=len, params=["file.path"], output=int)
            .save(next(names))
            .count()
        )

    assert benchmark(run) == NUM_FILES


def test_delta_warm(benchmark, files_uri, test_session):
    def run():
        return (
            dc.read_storage(
                files_uri, delta=True, delta_on="file.path", session=test_session
            )
            .map(name_len=len, params=["file.path"], output=int)
            .save("bench_delta_warm")
            .count()
        )

    run()
    assert benchmark(run) == NUM_FILES
