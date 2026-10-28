from collections.abc import Sequence

from datachain.client import Client
from datachain.lib.dc.storage_pattern import validate_cloud_bucket_name
from datachain.query.session import Session


def _backends_have_credentials(uris, client_config: dict | None) -> bool:
    """True if any backend behind ``uris`` sees credentials in client_config."""
    if not client_config:
        return False
    seen: set[type[Client]] = set()
    for uri in uris:
        try:
            seen.add(Client.get_implementation(str(uri)))
        except NotImplementedError:
            return True
    return any(c.has_explicit_credentials(client_config) for c in seen)


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
) -> tuple[dict | None, dict]:
    """The client config a session is made with, if not the active one's, and
    the client config to list ``uris`` with.

    ``client_config`` if given, else the config of ``session`` or the active
    session. ``anon`` is set
    on it, True when None and every bucket is public and has no credentials,
    and ``version_aware=False`` without ``versions``. An explicit session is
    never changed.
    """
    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))
    config = _session_client_config(session, client_config)
    if (
        anon is None
        and not _backends_have_credentials(uris, config)
        and _all_buckets_anonymous(uris, config)
    ):
        anon = True
    if anon is not None:
        config = config | {"anon": anon}
    session_config = config if client_config is not None or anon is not None else None
    if not versions:
        config = config | {"version_aware": False}
    return session_config, config


def session_for(
    session: Session | None, client_config: dict | None, in_memory: bool
) -> Session:
    """Apply client options without changing the caller or its cleanup scope."""
    session = Session.get(session, in_memory=in_memory)
    if client_config is None or session.catalog.client_config == client_config:
        return session
    catalog = session.catalog.copy()
    catalog.client_config = client_config
    return session._with_catalog(catalog)


def _session_client_config(session: Session | None, client_config: dict | None) -> dict:
    """``client_config`` if given, else the session's, without creating one."""
    if client_config is not None:
        return client_config
    session = session or Session.active()
    return session.catalog.client_config if session else {}
