import pytest

from datachain.client import bucket_status


@pytest.mark.parametrize(
    "uri",
    [
        "s3://my-bucket/some/path",
        "gs://my-bucket/dir",
        "az://my-container/blob",
    ],
)
def test_bucket_status_rejects_path_component(uri):
    with pytest.raises(ValueError, match="path in a bucket is not allowed"):
        bucket_status(uri)
