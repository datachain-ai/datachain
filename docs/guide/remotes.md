# Interacting with remote storage

DataChain supports reading and writing data from different remote storages using methods like `dc.read_storage` and `dc.to_storage`. The supported storages includes: local file system, AWS S3 storage, Google Cloud Storage, Azure Blob Storage, Hugging Face and more.

Example implementation for reading and writing data from/to different remote storages:

```python
import datachain as dc

dc = dc.read_storage("s3://bucket-name/path/to/data")
dc.to_storage("gs://bucket-name/path/to/data")
```

DataChain uses [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) to interact with different remote storages. You can pass the following fsspec-supported URIs to `read_storage` and `to_storage` methods.

- Local file system: `file://path/to/data`
- AWS S3 storage: `s3://bucket-name/path/to/data`
- Google Cloud Storage: `gs://bucket-name/path/to/data`
- Azure Blob Storage: `az://container-name/path/to/data`
- Hugging Face: `hf://dataset-name`

## Export placement strategies

Both `File.export` and `DataChain.to_storage` expose a `placement`
parameter that controls how the target path is built for each file. Choose the
strategy that fits your destination layout:

- `filename`: keep only the original file name (no directories). Simple, but
    colliding names overwrite each other.
- `filepath`: preserve the relative directory structure inside the dataset.
    This is the right choice when you want to mirror the source folder layout at
    the destination.
- `fullpath`: for remote sources, prefix paths with the storage host (for
    example `s3://bucket/data/a.jpg` exports to `<output>/bucket/data/a.jpg`).
    Local sources behave the same as `filepath`.
- `etag`: use the file ETag with the original extension to guarantee unique
    names when your storage exposes object digests.
- `checksum`: reserved for future use. Calling it currently raises
    `NotImplementedError`.

Relative output directories such as `"."` or `".."`, as well as absolute
paths, are supported. The placement strategy simply determines the appended
sub-path.

## Extra configuration
For the configuration parameters to the filesystem, you can pass the key and value pair as client_config dictionary that will be passed to the respective filesystem.


### AWS S3 compatible storage

DataChain uses [s3fs](https://s3fs.readthedocs.io/en/latest/) to interact with AWS S3 storage. Authentication can be configured using standard AWS credential locations, such as `~/.aws/credentials` and `~/.aws/config`. You can also pass the following configuration parameters to the s3fs filesystem as `client_config` dictionary.

- `anon`: `bool` (default: `False`)

    Whether to use anonymous connection (public buckets only). If `False`,
    uses the key/secret given, or boto's credential resolver (client_kwargs,
    environment, variables, config files, EC2 IAM server, in that order)

- `endpoint_url`: `string` (default: `None`)

    Use this endpoint URL, if specified. Needed for connecting to non-AWS
    S3 buckets. Takes precedence over `endpoint_url` in client_kwargs.

- `key`: `string` (default: `None`)

    If not anonymous, use this access key ID, if specified. Takes precedence
    over `aws_access_key_id` in client_kwargs.

- `secret`: `string` (default: `None`)

    If not anonymous, use this secret access key, if specified. Takes
    precedence over `aws_secret_access_key` in client_kwargs.

- `token`: `string` (default: `None`)

    If not anonymous, use this security token, if specified

- `use_ssl`: `bool` (default: `True`)

    Whether to use SSL in connections to S3; may be faster without, but
    insecure. If `use_ssl` is also set in `client_kwargs`,
    the value set in `client_kwargs` will take priority.

- `s3_additional_kwargs`: `dict` (default: `{}`)

    Dict of parameters that are used when calling s3 api
    methods. Typically used for things like "ServerSideEncryption".

- `client_kwargs`: `dict` (default: `{}`)

    Dict of parameters for the botocore client.

- `requester_pays`: `bool` (default: `False`)

    If RequesterPays buckets are supported.

- `default_block_size`: `int` (default: `None`)

    If given, the default block size value used for `open()`, if no
    specific value is given at all time. The built-in default is 5MB.

- `default_fill_cache`: `bool` (default: `True`)

    Whether to use cache filling with open by default. Refer to `S3File.open`.

- `default_cache_type`: `string` (default: `"readahead"`)

    If given, the default cache_type value used for `open()`. Set to `None`
    if no caching is desired. See fsspec's documentation for other available
    `cache_type` values. Default cache_type is `"readahead"`.

- `version_aware`: `bool` (default: `True`)

    Whether to support bucket versioning. If enable this will require the
    user to have the necessary IAM permissions for dealing with versioned
    objects. Note that in the event that you only need to work with the
    latest version of objects in a versioned bucket, and do not need the
    VersionId for those objects, you should set `version_aware` to `False`
    for performance reasons. When set to `True`, filesystem instances will
    use the S3 `ListObjectVersions` API call to list directory contents,
    which requires listing all historical object versions.

- `cache_regions`: `bool` (default: `False`)

    Whether to cache bucket regions or not. Whenever a new bucket is used,
    it will first find out which region it belongs and then use the client
    for that region.

- `asynchronous`: `bool` (default: `False`)

    Whether this instance is to be used from inside coroutines.

- `config_kwargs`: `dict` (default: `{}`)

    Dict of parameters passed to `botocore.client.Config`.

- `kwargs`: `dict` (default: `{}`)

    Other parameters for core session.

- `session`: `aiobotocore.session.AioSession` (default: `None`)

    Aiobotocore `AioSession` object to be used for all connections.
    This session will be used inplace of creating a new session inside S3FileSystem.

    For example: `aiobotocore.session.AioSession(profile='test_user')`

- `max_concurrency`: `int` (default: `1`)

    The maximum number of concurrent transfers to use per file for multipart
    upload (`put()`) operations. Defaults to `1` (sequential). When used in
    conjunction with `S3FileSystem.put(batch_size=...)` the maximum number of
    simultaneous connections is `max_concurrency * batch_size`. We may extend
    this parameter to affect `pipe()`, `cat()` and `get()`. Increasing this
    value will result in higher memory usage during multipart upload operations (by
    `max_concurrency * chunksize` bytes per file).


Example:

```python
chain = dc.read_storage(
    "s3://my-bucket/my-dir",
    client_config = {
		"endpoint_url": "<minio-endpoint-url>",
		"key": "<minio-access-key",
		"secret": "<minio-secret-key"
	}
)
```

### Google Cloud Storage

DataChain uses [gcsfs](https://gcsfs.readthedocs.io/en/latest/) to interact with Google Cloud Storage. Authentication can be achieved by using any of the method described at [gcsfs documentation](https://gcsfs.readthedocs.io/en/latest/#credentials). You can also pass the following configuration parameters to the gcsfs filesystem as client_config dictionary.

- `project`: `string` (default: `None`)

    The project to work under. Note that this is not the same as, but often
    very similar to, the project name. This is required in order to list all
    the buckets you have access to within a project and to create/delete
    buckets, or update their access policies. If `token='google_default'`,
    the value is overridden by the default, if `token='anon'`, the value is
    ignored.

- `access`: `string` (default: `None`)

    One of `"read_only"`, `"read_write"`, `"full_control"`. Full control implies
    read/write as well as modifying metadata, e.g., access control.

- `token`: `None`, `dict` or `string` (default: `None`)

    The token to use for authentication. If `None`, the default is used. If
    a string, it is interpreted as a path to a token file. If a dict, it is
    interpreted as a token dictionary, such as that provided by Google Cloud
    Platform. See also description of authentication methods, from link above.

- `consistency`: `string` (default: `None`)

    One of `"none"`, `"size"`, `"md5"`. Check method when writing files.
    Can be overridden in `open()`.

- `cache_timeout`: `float` (default: `None`)

    Cache expiration time in seconds for object metadata cache. Set
    `cache_timeout <= 0` for no caching, `None` for no cache expiration.

- `secure_serialize`: `bool` (default: `None`)

    Whether to use secure serialization. This is a deprecated option and
    will be removed in future versions.

- `requester_pays`: `bool` or `str` (default: `False`)

    Whether to use requester-pays requests. This will include your
    project ID `project` in requests as the `userProject`, and you'll be
    billed for accessing data from requester-pays buckets. Optionally,
    pass a project-id here as a string to use that as the `userProject`.

- `session_kwargs`: `dict` (default: `{}`)

    Passed on to `aiohttp.ClientSession`. Can contain, for example, proxy
    settings.

- `endpoint_url`: `string` (default: `None`)

    If given, use this URL (format: `protocol://host:port`, *without* any
    path part) for communication. If not given, defaults to the value
    of environment variable `"STORAGE_EMULATOR_HOST"`; if that is not set
    either, will use the standard Google endpoint.

- `default_location`: `str` (default: `None`)

    Default location where buckets are created, like `"US"` or `"EUROPE-WEST3"`.
    You can find a list of all available locations here:
    https://cloud.google.com/storage/docs/locations#available-locations

- `version_aware`: `bool` (default: `False`)

    Whether to support object versioning. If enabled this will require the
    user to have the necessary permissions for dealing with versioned objects.


### Azure Blob Storage

DataChain uses [adlfs](https://fsspec.github.io/adlfs/) to interact with Azure Blob Storage. Authentication can be achieved by using any of the method described at [adlfs documentation](https://github.com/fsspec/adlfs?tab=readme-ov-file#setting-credentials). You can also pass the following configuration parameters to the adlfs filesystem as client_config dictionary.

- `account_name`: `str` (default: `None`)

    The storage account name. This is used to authenticate requests
    signed with an account key and to construct the storage endpoint. It
    is required unless a connection string is given, or if a custom
    domain is used with anonymous authentication.

- `account_key`: `str` (default: `None`)

    The storage account key. This is used for shared key authentication.
    If any of account key, sas token or client_id is specified, anonymous access
    will be used.

- `sas_token`: `str` (default: `None`)

    A shared access signature token to use to authenticate requests
    instead of the account key. If account key and sas token are both
    specified, account key will be used to sign. If any of account key, sas token
    or client_id are specified, anonymous access will be used.

- `request_session`: `requests.Session` (default: `None`)

    The session object to use for http requests.

- `connection_string`: `str` (default: `None`)

    If specified, this will override all other parameters besides
    request session. See
    http://azure.microsoft.com/en-us/documentation/articles/storage-configure-connection-string/
    for the connection string format.

- `credential`: `azure.core.credentials_async.AsyncTokenCredential` or SAS token (default: `None`)

    The credentials with which to authenticate. Optional if the account URL already has a SAS token.
    Can include an instance of TokenCredential class from azure.identity.aio.

- `blocksize`: `int` (default: `None`)

    The block size to use for download/upload operations. Defaults to hardcoded value of
    `BlockBlobService.MAX_BLOCK_SIZE`

- `client_id`: `str` (default: `None`)

    Client ID to use when authenticating using an AD Service Principal client/secret.

- `client_secret`: `str` (default: `None`)

    Client secret to use when authenticating using an AD Service Principal client/secret.

- `tenant_id`: `str` (default: `None`)

    Tenant ID to use when authenticating using an AD Service Principal client/secret.

- `anon`: `boolean` (default: `None`)

    The value to use for whether to attempt anonymous access if no other credential is
    passed. By default (`None`), the `AZURE_STORAGE_ANON` environment variable is
    checked. False values (`false`, `0`, `f`) will resolve to `False` and
    anonymous access will not be attempted. Otherwise the value for `anon` resolves
    to `True`.

- `default_fill_cache`: `bool` (default: `True`)

    Whether to use cache filling with open by default

- `default_cache_type`: `string` (default: `"bytes"`)

    If given, the default cache_type value used for `open()`. Set to `None` if no caching
    is desired. Docs in fsspec.

- `version_aware`: `bool` (default: `False`)

    Whether to support blob versioning. If enable this will require the user to have the
    necessary permissions for dealing with versioned blobs.

- `assume_container_exists`: `bool` (default: `None`)

    Set this to `True` to not check for existence of containers at all, assuming they exist.
    `None` (default) means to warn in case of a failure when checking for existence of a container.
    `False` throws if retrieving container properties fails, which might happen if your
    authentication is only valid at the storage container level, and not the
    storage account level.

- `max_concurrency`: `int` (default: `None`)

    The number of concurrent connections to use when uploading or downloading a blob.
    If `None` it will be inferred from `fsspec.asyn._get_batch_size()`.

- `timeout`: `int` (default: `None`)

    Sets the server-side timeout when uploading or downloading a blob.

- `connection_timeout`: `int` (default: `None`)

    The number of seconds the client will wait to establish a connection to the server
    when uploading or downloading a blob.

- `read_timeout`: `int` (default: `None`)

    The number of seconds the client will wait, between consecutive read operations,
    for a response from the server while uploading or downloading a blob.

- `account_host`: `str` (default: `None`)

    The storage account host. This string is the entire url to the for the storage
    after the `https://`, i.e. `"https://{account_host}"`. This parameter is only
    required for Azure clouds where account urls do not end with `"blob.core.windows.net"`.
    Note that the `account_name` parameter is still required.


### Hugging Face

DataChain uses [huggingface_hub](https://pypi.org/project/huggingface-hub/) to interact with Hugging Face. You can pass the following parameters to client config to interact with Hugging Face.

- `token`: `str` or `bool` (default: `None`)

    A valid user access token (string). Defaults to the locally saved
    token, which is the recommended method for authentication (see
    https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
    To disable authentication, pass `False`.

- `endpoint`: `str` (default: `None`)

    Endpoint of the Hub. Defaults to `https://huggingface.co`.
