import fsspec
from packaging.version import Version, parse

# fsspec==2025.2.0 added support for a proper `open()` in `ReferenceFileSystem`.
# Remove this module when `fsspec` minimum version requirement can be bumped.
if parse(fsspec.__version__) < Version("2025.2.0"):
    from fsspec.core import split_protocol
    from fsspec.implementations import reference

    class ReferenceFileSystem(reference.ReferenceFileSystem):
        def _open(self, path, mode="rb", *args, **kwargs):
            # overriding because `fsspec`'s `ReferenceFileSystem._open`
            # reads the whole file in-memory.
            (uri,) = self.references[path]
            protocol, _ = split_protocol(uri)
            return self.fss[protocol].open(uri, mode, *args, **kwargs)
else:
    from fsspec.implementations.reference import ReferenceFileSystem  # type: ignore[no-redef]


__all__ = ["ReferenceFileSystem"]
