import pytest
from botocore.utils import parse_timestamp

from datachain.client.s3 import ClientS3, _parse_timestamp
from datachain.client.writeconfig import WriteConfig
from datachain.node import DirType, Node
from datachain.nodes_thread_pool import NodeChunk


@pytest.mark.parametrize("streaming", [True, False])
def test_write_kwargs_maps_all_fields(streaming):
    # s3fs forwards extra kwargs into s3_additional_kwargs (boto3 put_object /
    # multipart args), including the raw write_options escape hatch.
    cfg = WriteConfig(
        content_type="application/pdf",
        content_disposition="attachment",
        cache_control="max-age=3600",
        content_encoding="gzip",
        metadata={"a": "b"},
        write_options={"ACL": "public-read"},
    )
    assert ClientS3._write_kwargs(cfg, streaming=streaming) == {
        "ContentType": "application/pdf",
        "ContentDisposition": "attachment",
        "CacheControl": "max-age=3600",
        "ContentEncoding": "gzip",
        "Metadata": {"a": "b"},
        "ACL": "public-read",
    }


def test_write_kwargs_empty_config_maps_nothing():
    assert ClientS3._write_kwargs(WriteConfig(), streaming=False) == {}


@pytest.fixture
def nodes():
    return iter(
        [
            make_size_node(11, DirType.DIR, "a/f1", 100),
            make_size_node(12, DirType.FILE, "b/f2", 100),
            make_size_node(13, DirType.FILE, "c/f3", 100),
            make_size_node(14, DirType.FILE, "d/", 100),
            make_size_node(15, DirType.FILE, "e/f5", 100),
            make_size_node(16, DirType.DIR, "f/f6", 100),
            make_size_node(17, DirType.FILE, "g/f7", 100),
        ]
    )


def make_size_node(node_id, dir_type, path, size):
    return Node(
        node_id,
        dir_type=dir_type,
        path=path,
        size=size,
    )


class FakeCache:
    def contains(self, _):
        return False


def make_chunks(nodes, *args, **kwargs):
    return NodeChunk(FakeCache(), "s3://foo", nodes, *args, **kwargs)


def test_node_bucket_the_only_item():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 201)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_only_item_over_limit():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_last_one():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    next(bkt)
    with pytest.raises(StopIteration):
        next(bkt)


def test_node_bucket_basic(nodes):
    bkt = list(make_chunks(nodes, 201))

    assert len(bkt) == 2
    assert len(bkt[0]) == 3
    assert len(bkt[1]) == 1


def test_node_bucket_full_split(nodes):
    bkt = list(make_chunks(nodes, 0))

    assert len(bkt) == 4
    assert len(bkt[0]) == 1
    assert len(bkt[1]) == 1
    assert len(bkt[2]) == 1
    assert len(bkt[3]) == 1


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-30T23:00:00.000Z",
        "2026-09-30T23:00:00Z",
        "Wed, 30 Sep 2026 23:00:00 GMT",
        1790809200,
    ],
)
def test_parse_timestamp_matches_botocore(value):
    assert _parse_timestamp(value) == parse_timestamp(value)


LISTING = (
    '<?xml version="1.0" encoding="UTF-8"?><ListVersionsResult '
    'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name><Prefix>p/</Prefix>'
    "<KeyMarker></KeyMarker><VersionIdMarker></VersionIdMarker><MaxKeys>2</MaxKeys>"
    "<EncodingType>url</EncodingType><IsTruncated>true</IsTruncated>"
    "<NextKeyMarker>p%2Fb</NextKeyMarker><NextVersionIdMarker>v2</NextVersionIdMarker>"
    "<Version><Key>p%2Fa</Key><VersionId>v1</VersionId><IsLatest>true</IsLatest>"
    "<LastModified>2026-09-30T23:00:00.000Z</LastModified><ETag>&quot;e&quot;</ETag>"
    "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><Size>3</Size>"
    "<StorageClass>STANDARD</StorageClass><Owner><ID>o</ID></Owner>"
    "<RestoreStatus><IsRestoreInProgress>false</IsRestoreInProgress>"
    "<RestoreExpiryDate>2026-10-01T00:00:00.000Z</RestoreExpiryDate></RestoreStatus>"
    "</Version><DeleteMarker><Key>p%2Fb</Key><VersionId>v2</VersionId>"
    "<IsLatest>false</IsLatest><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
    "</DeleteMarker><CommonPrefixes><Prefix>p%2Fc%2F</Prefix></CommonPrefixes>"
    "</ListVersionsResult>"
)


@pytest.mark.parametrize("operation", ["ListObjectVersions", "ListObjectsV2"])
def test_listing_parser_matches_botocore(operation):
    from aiobotocore.parsers import AioRestXMLParser
    from botocore.session import get_session
    from fsspec.asyn import get_loop, sync

    from datachain.client.s3 import _ListingParser

    body = LISTING.encode()
    if operation == "ListObjectsV2":
        body = (
            body.replace(b"ListVersionsResult", b"ListBucketResult")
            .replace(b"<Version>", b"<Contents>")
            .replace(b"</Version>", b"</Contents>")
        )
    shape = (
        get_session().get_service_model("s3").operation_model(operation).output_shape
    )
    response = {"status_code": 200, "headers": {}, "body": body}
    expected = sync(get_loop(), AioRestXMLParser().parse, response, shape)
    assert sync(get_loop(), _ListingParser().parse, response, shape) == expected
