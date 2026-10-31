import inspect
from datetime import UTC, datetime

from fsspec.asyn import get_loop, sync

import datachain as dc
from datachain.lib.file import FileRow


def _s3_page(keys: int) -> bytes:
    contents = "".join(
        f"<Contents><Key>data/{i:07d}.jpg</Key>"
        "<LastModified>2026-09-30T23:00:00.000Z</LastModified>"
        f"<ETag>&quot;{i:032x}&quot;</ETag><Size>{i}</Size>"
        "<StorageClass>STANDARD</StorageClass></Contents>"
        for i in range(keys)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult '
        'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name>'
        f"<KeyCount>{keys}</KeyCount><MaxKeys>{keys}</MaxKeys>"
        f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
    ).encode()


def test_s3_listing_pages_parse(benchmark):
    from botocore.session import get_session

    from datachain.client.s3 import _ListingParser

    model = get_session().get_service_model("s3")
    shape = model.operation_model("ListObjectsV2").output_shape
    body, parser = _s3_page(1000), _ListingParser()

    async def parse():
        for _ in range(20):
            response = {"status_code": 200, "headers": {}, "body": body}
            parsed = parser.parse(response, shape)
            if inspect.isawaitable(parsed):
                await parsed

    benchmark(sync, get_loop(), parse)


def test_listing_rows_are_saved(benchmark, test_session):
    when = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [
        FileRow("s3://bucket", f"data/{i:07d}.jpg", i, "", f"{i:032x}", True, when)
        for i in range(100_000)
    ]

    def listing():
        yield from rows

    def save():
        chain = dc.read_values(seed=[0], session=test_session).settings(prefetch=0)
        chain.gen(listing, output={"file": dc.File}).exec()

    benchmark.pedantic(save, rounds=3)
