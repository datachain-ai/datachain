from .fsspec import BucketStatus, Client


def bucket_status(uri: str, **client_config) -> BucketStatus:
    """Check bucket existence and access level without listing objects.

    Args:
        uri: Bucket URI, e.g. "s3://my-bucket/", "gs://my-bucket/", "az://my-container/"
        **client_config: Storage client configuration (aws_key, etc.)
            For Azure, pass ``account_name`` to enable anonymous access detection;
            without it, only authenticated access is probed.

    Returns:
        BucketStatus(exists, access) where access is one of:
        'anonymous', 'authenticated', 'denied'
    """
    client_cls = Client.get_implementation(uri)
    name, path = client_cls.split_url(uri)
    if path:
        raise ValueError(f"path in a bucket is not allowed, only bucket name: {uri!r}")
    return client_cls.bucket_status(name, **client_config)


__all__ = ["BucketStatus", "Client", "bucket_status"]
