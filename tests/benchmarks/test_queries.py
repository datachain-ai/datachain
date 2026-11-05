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
            .to_list("stem", "ext", "is_dog")
        )

    assert len(benchmark(run)) == NUM_FILES


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


@pytest.mark.parametrize("retry", [False, True, "error"])
def test_delta_changed(benchmark, test_session, retry):
    processed = []

    def generate(id, value):
        processed.append(id)
        if retry is True and id == 2:
            return
        yield id, value * 2, "retry" if retry == "error" and id == 2 else ""

    def build(delta):
        return dc.read_dataset(
            "bench_delta_source",
            delta=delta,
            delta_on="id",
            delta_retry=retry,
            session=test_session,
        ).gen(generate, output={"id": int, "score": int, "error": str})

    def setup():
        dc.read_values(
            id=list(range(NUM_FILES)),
            value=list(range(NUM_FILES)),
            session=test_session,
        ).save("bench_delta_source")
        build(False).save("bench_delta_changed")
        dc.read_values(
            id=list(range(1, NUM_FILES + 10)),
            value=[-1, *range(2, NUM_FILES + 10)],
            session=test_session,
        ).save("bench_delta_source")
        processed.clear()

    result = benchmark.pedantic(
        lambda: build(True).save("bench_delta_changed"),
        setup=setup,
    )
    expected_ids = [1, *range(NUM_FILES, NUM_FILES + 10)] + ([2] if retry else [])
    assert sorted(processed) == sorted(expected_ids)
    assert sorted(result.to_list("id", "score", "error")) == [
        (id, -2 if id == 1 else id * 2, "retry" if retry == "error" and id == 2 else "")
        for id in range(1, NUM_FILES + 10)
        if not (retry is True and id == 2)
    ]
