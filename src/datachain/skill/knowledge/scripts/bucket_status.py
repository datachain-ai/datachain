"""Fast bucket status check — existence and access level without listing.

Usage:
    python bucket_status.py <uri>

Output:
    Status: exists
    Access: authenticated

Exit code 0 if the bucket exists, 1 otherwise.
access ∈ {anonymous, authenticated, denied}
"""

import sys
from urllib.parse import urlparse


class BucketStatus:
    """Result of a fast bucket access check."""

    __slots__ = ("access", "error", "exists")

    def __init__(self, exists: bool, access: str, error: str | None = None):
        self.exists = exists
        self.access = access  # "anonymous" | "authenticated" | "denied"
        self.error = error

    @property
    def anon(self) -> bool:
        return self.access == "anonymous"

    def __repr__(self) -> str:
        return (
            f"BucketStatus(exists={self.exists!r}, access={self.access!r}"
            f"{f', error={self.error!r}' if self.error else ''})"
        )


def bucket_status(uri: str) -> BucketStatus:
    """Check if a bucket exists and whether it allows anonymous access.

    Uses cloud SDKs directly — no listing, so the call is fast.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme
    bucket = parsed.netloc

    if scheme == "s3":
        return _check_s3(bucket)
    if scheme == "gs":
        return _check_gcs(bucket)
    if scheme == "az":
        # az://container/path — container is netloc, account comes from env
        return _check_azure(container=bucket)
    return BucketStatus(False, "denied", f"Unsupported scheme: {scheme}")


# ---------------------------------------------------------------------------
# Provider-specific probes
# ---------------------------------------------------------------------------


def _check_s3(bucket: str) -> BucketStatus:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        return BucketStatus(False, "denied", "boto3 not installed")

    # Anonymous probe
    try:
        anon = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        anon.head_bucket(Bucket=bucket)
        return BucketStatus(True, "anonymous")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            return BucketStatus(False, "denied", f"S3 bucket '{bucket}' not found")
    except Exception:  # noqa: BLE001
        pass

    # Authenticated probe
    try:
        client = boto3.client("s3")
        client.head_bucket(Bucket=bucket)
        return BucketStatus(True, "authenticated")
    except NoCredentialsError:
        return BucketStatus(
            True,
            "denied",
            f"S3 bucket '{bucket}' exists but no AWS credentials are configured",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            return BucketStatus(False, "denied", f"S3 bucket '{bucket}' not found")
        if code in ("403", "401"):
            return BucketStatus(
                True,
                "denied",
                f"Access denied to S3 bucket '{bucket}'"
                " — check AWS credentials/permissions",
            )
        return BucketStatus(False, "denied", str(e))
    except Exception as e:  # noqa: BLE001
        return BucketStatus(False, "denied", str(e))


def _check_gcs(bucket: str) -> BucketStatus:
    try:
        from google.api_core.exceptions import Forbidden, NotFound
        from google.cloud import storage as gcs_storage
    except ImportError:
        return BucketStatus(False, "denied", "google-cloud-storage not installed")

    # Anonymous probe — list one blob (bucket.reload() requires
    # storage.buckets.get which public buckets don't grant anonymously)
    try:
        client = gcs_storage.Client.create_anonymous_client()
        list(client.list_blobs(bucket, max_results=1))
        return BucketStatus(True, "anonymous")
    except NotFound:
        return BucketStatus(False, "denied", f"GCS bucket '{bucket}' not found")
    except Forbidden:
        pass
    except Exception:  # noqa: BLE001
        pass

    # Authenticated probe
    try:
        client = gcs_storage.Client()
        list(client.list_blobs(bucket, max_results=1))
        return BucketStatus(True, "authenticated")
    except NotFound:
        return BucketStatus(False, "denied", f"GCS bucket '{bucket}' not found")
    except Forbidden:
        return BucketStatus(
            True,
            "denied",
            f"Access denied to GCS bucket '{bucket}' — check credentials/permissions",
        )
    except Exception as e:  # noqa: BLE001
        return BucketStatus(False, "denied", str(e))


def _check_azure(container: str) -> BucketStatus:
    # DataChain Azure URIs: az://container/path
    # Account name comes from AZURE_STORAGE_ACCOUNT_NAME env var or adlfs config.
    try:
        from adlfs import AzureBlobFileSystem
    except ImportError:
        return BucketStatus(False, "denied", "adlfs not installed")

    # Anonymous probe
    try:
        fs = AzureBlobFileSystem(anon=True)
        fs.ls(container, detail=False)[:1]
        return BucketStatus(True, "anonymous")
    except Exception:  # noqa: BLE001
        pass

    # Authenticated probe (uses env vars / DefaultAzureCredential)
    try:
        fs = AzureBlobFileSystem()
        fs.ls(container, detail=False)[:1]
        return BucketStatus(True, "authenticated")
    except FileNotFoundError:
        return BucketStatus(False, "denied", f"Azure container '{container}' not found")
    except Exception as e:  # noqa: BLE001
        err = str(e)
        if "AuthenticationError" in err or "credential" in err.lower():
            return BucketStatus(
                True,
                "denied",
                f"Access denied to Azure container '{container}'"
                " — check credentials/configuration",
            )
        return BucketStatus(False, "denied", err)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <uri>", file=sys.stderr)
        sys.exit(2)

    result = bucket_status(sys.argv[1])

    print(f"Status: {'exists' if result.exists else 'not found'}")
    print(f"Access: {result.access}")
    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)

    sys.exit(0 if result.exists else 1)
