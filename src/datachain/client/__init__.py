from .fsspec import BucketStatus, Client


def bucket_status(uri: str, **client_config) -> BucketStatus:
    """Check bucket existence and access level without listing objects.

    Args:
        uri: Bucket URI, e.g. "s3://my-bucket/", "gs://my-bucket/", "az://my-container/"
        **client_config: Storage client configuration (anon, aws_key, etc.)
            For Azure, pass ``account_name`` to enable anonymous access detection.
            Without it, Azure container status detection may fail and report the
            container as non-existent or access as ``denied``.

    Returns:
        BucketStatus(exists, access) where access is one of:
        'anonymous', 'authenticated', 'denied'
    """
    client_cls = Client.get_implementation(uri)
    name, _ = client_cls.split_url(uri)
    return client_cls.bucket_status(name, **client_config)


__all__ = ["BucketStatus", "Client", "bucket_status"]
