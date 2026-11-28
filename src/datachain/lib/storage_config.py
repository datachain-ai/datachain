from collections.abc import Sequence

from fsspec.config import apply_config

from datachain.client import Client
from datachain.query.session import Session


def _backends_have_auth_config(uris, client_config: dict | None) -> bool:
    seen: set[type[Client]] = set()
    for uri in uris:
        try:
            seen.add(Client.get_implementation(str(uri)))
        except NotImplementedError:
            return True
    for client in seen:
        config = apply_config(client.FS_CLASS, client_config or {})
        if config.get("anon") is not None or client.has_explicit_credentials(config):
            return True
    return False


def _all_buckets_anonymous(uris, client_config: dict | None) -> bool:
    """Probe each unique bucket; True iff all probe as anonymous."""
    to_probe: set[tuple[type[Client], str]] = set()
    for uri in uris:
        try:
            client_cls = Client.get_implementation(str(uri))
        except NotImplementedError:
            return False
        name, _ = client_cls.split_url(str(uri))
        if not name:
            return False
        to_probe.add((client_cls, name))

    if not to_probe:
        return False

    for client_cls, name in to_probe:
        try:
            status = client_cls.bucket_status(name, **(client_config or {}))
        except NotImplementedError:
            # Backend doesn't support bucket_status (e.g. local files).
            return False
        if status.access != "anonymous":
            return False
    return True


def client_configs(
    uris: Sequence,
    session: Session | None,
    anon: bool | None,
    client_config: dict | None,
    versions: bool,
    in_memory: bool = False,
) -> tuple[dict, dict]:
    """The client config to read the files of ``uris`` with, and the client
    config to list them with.

    ``client_config`` if given, else the config of ``session`` or the active
    session, the shared in-memory session's for ``in_memory``. ``anon`` is set
    on it, True when None and every bucket is public and has no credentials,
    and ``version_aware=False`` without ``versions``. An explicit session is
    never changed.
    """
    from datachain.lib.dc.storage_pattern import validate_cloud_bucket_name

    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))
    config = _session_client_config(session, client_config, in_memory)
    if anon is None:
        anon = config.get("anon")
    if (
        anon is None
        and not _backends_have_auth_config(uris, config)
        and _all_buckets_anonymous(uris, config)
    ):
        anon = True
    if anon is not None:
        config = config | {"anon": anon}
    session_config = config
    if not versions:
        config = config | {"version_aware": False}
    return session_config, config


def session_for(
    session: Session | None, client_config: dict, in_memory: bool
) -> Session:
    """A view of the session with a catalog of its own for one chain: it reads
    with `client_config` and records the sources the chain lists. The caller
    and its cleanup scope are unchanged."""
    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog.copy()
    catalog.client_config = client_config
    catalog.source_client_configs = {}
    return session._with_catalog(catalog)


def _session_client_config(
    session: Session | None, client_config: dict | None, in_memory: bool
) -> dict:
    """``client_config`` if given, else the session's, without creating one."""
    if client_config is not None:
        return client_config
    if session is None:
        session = Session.active()
        if in_memory and session and not session.catalog.in_memory:
            # The chain runs in the shared in-memory session, which has no config.
            session = None
    return session.catalog.client_config if session else {}
