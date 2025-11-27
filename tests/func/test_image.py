import os

import pytest
from PIL import Image as PILImage
from torch import Tensor
from torchvision.transforms import ToTensor

from datachain.lib.file import File, FileError, ImageFile
from datachain.lib.image import convert_image


@pytest.fixture(autouse=True)
def image_file(catalog) -> File:
    data_path = os.path.join(os.path.dirname(__file__), "data")
    file_name = "lena.jpg"

    with open(os.path.join(data_path, file_name), "rb") as f:
        file = File.upload(f.read(), file_name)

    file.ensure_cached()
    return file


def test_image_file(image_file):
    assert not isinstance(image_file.read(), PILImage.Image)
    assert isinstance(image_file.as_image_file().read(), PILImage.Image)

    with open(os.path.join(os.path.dirname(__file__), "data", "lena.jpg"), "rb") as f:
        content = f.read()
        assert image_file.read_bytes() == content
        assert image_file.as_image_file().read_bytes() == content


@pytest.mark.parametrize("format", [None, "JPEG", "PNG"])
def test_image_save(tmp_path, image_file, format):
    image_file = image_file.as_image_file()
    filename = f"{tmp_path}/test.jpg"
    image_file.save(filename, format=format)

    with PILImage.open(filename) as img:
        assert img.format == (format or "JPEG")
        assert img.size == (256, 256)


def test_image_save_no_extension(tmp_path, image_file):
    image_file = image_file.as_image_file()
    filename = f"{tmp_path}/test"
    with pytest.raises(FileError):
        image_file.save(filename)


@pytest.mark.parametrize("format", [None, "JPEG", "PNG"])
def test_image_save_cloud(cloud_test_catalog_upload, image_file, format):
    """Test saving ImageFile to different cloud filesystems (S3, GCS, Azure)."""
    ctc = cloud_test_catalog_upload
    image_file = image_file.as_image_file()
    image_file._set_stream(ctc.catalog)

    # Save to cloud storage with the specified format
    cloud_filename = f"{ctc.src_uri}/test_image.jpg"
    image_file.save(cloud_filename, format=format, client_config=ctc.client_config)

    # Verify the saved file by reading it back
    saved_image_file = ImageFile(path="test_image.jpg", source=ctc.src_uri)
    saved_image_file._set_stream(ctc.catalog)

    # Read the saved image and verify it
    saved_img = saved_image_file.read()
    assert saved_img.format == (format or "JPEG")
    assert saved_img.size == (256, 256)


def test_get_info(image_file):
    info = image_file.as_image_file().get_info()
    assert info.model_dump() == {"width": 256, "height": 256, "format": "JPEG"}


def test_get_info_error():
    # upload current Python file as image file to get an error while getting image meta
    with open(__file__, "rb") as f:
        file = ImageFile.upload(f.read(), "test.jpg")

    file.ensure_cached()
    with pytest.raises(FileError):
        file.get_info()


def test_convert_image(image_file):
    converted_img = convert_image(
        image_file.as_image_file().read(),
        mode="RGBA",
        size=(32, 32),
        transform=ToTensor(),
    )
    assert isinstance(converted_img, Tensor)
    assert converted_img.size() == (4, 32, 32)
