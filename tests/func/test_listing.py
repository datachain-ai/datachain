import pytest

import datachain as dc
from datachain.lib.file import File
from datachain.lib.listing import list_bucket, parse_listing_uri
from tests.data import ENTRIES


def test_listing_generator(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    catalog = ctc.catalog

    uri = f"{ctc.src_uri}/cats"

    chain = dc.read_records(dc.DataChain.DEFAULT_FILE_RECORD).gen(
        file=list_bucket(uri, catalog.cache, client_config=catalog.client_config)
    )
    assert chain.count() == 2

    entries = sorted(
        [e for e in ENTRIES if e.path.startswith("cats/")], key=lambda e: e.path
    )
    files = chain.order_by("file.path").to_values("file")

    for cat_file, cat_entry in zip(files, entries, strict=False):
        assert cat_file.source == ctc.src_uri
        assert cat_file.path == cat_entry.path
        assert cat_file.size == cat_entry.size
        assert cat_file.etag is not None
        # If version_aware is not passed it is enforced to be True internally
        if catalog.client_config.get("version_aware", True):
            assert cat_file.version is not None
        else:
            assert cat_file.version == ""
        assert cat_file.is_latest == cat_entry.is_latest
        assert cat_file.location is None


def test_read_storage_percent_encoding_is_opaque_across_backends(
    cloud_test_catalog_upload,
):
    ctc = cloud_test_catalog_upload
    src_uri = ctc.src_uri
    catalog = ctc.catalog

    prefix = "dir with space"
    payloads = {
        "he?llo.txt": b"hi",
        "hello#.txt": b"hi#",
        "file with space.txt": b"h i",
        "file%20literal.txt": b"hi %20",
    }

    for name, content in payloads.items():
        File.upload(content, f"{src_uri}/{prefix}/{name}", catalog)

    # Reading via the literal (non-encoded) storage URI works.
    listed = dc.read_storage(f"{src_uri}/{prefix}", session=ctc.session).to_values(
        "file"
    )
    listed_paths = {f.path for f in listed}
    expected_paths = (
        set(payloads)
        if str(src_uri).startswith("file://")
        else {f"{prefix}/{name}" for name in payloads}
    )
    assert listed_paths == expected_paths

    # Percent-encoding is treated literally: it addresses a different prefix.
    encoded_prefix = "dir%20with%20space"
    with pytest.raises(FileNotFoundError, match=r"dir%20with%20space"):
        dc.read_storage(
            f"{src_uri}/{encoded_prefix}",
            session=ctc.session,
        ).to_values("file")

    # And addressing a specific key with %20 in the path fails.
    for name in payloads:
        with pytest.raises(FileNotFoundError):
            File.at(
                f"{src_uri}/{encoded_prefix}/{name}",
                session=ctc.session,
            ).read()


def test_parse_listing_uri(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(f"{ctc.src_uri}/dogs")
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs/"
    if cloud_type == "file":
        assert listing_path == ""
    else:
        assert listing_path == "dogs/"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_parse_listing_uri_with_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(f"{ctc.src_uri}/dogs/*")
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs"
    assert listing_path == "dogs/*"
