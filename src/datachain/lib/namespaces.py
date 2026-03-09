from datachain.error import (
    NamespaceCreateNotAllowedError,
    NamespaceDeleteNotAllowedError,
)
from datachain.lib.projects import delete as delete_project
from datachain.namespace import Namespace, parse_name
from datachain.query import Session


def create(
    name: str, descr: str | None = None, session: Session | None = None
) -> Namespace:
    """
    Creates a new namespace.

    Namespaces organize projects, which in turn organize datasets. A default
    namespace always exists and is used if none is specified. Multiple namespaces
    can be created in Studio, but only the default is available in the CLI.

    Parameters:
        name: Name of the new namespace.
        descr: Optional description of the namespace.
        session: Optional session to use for the operation.

    Example:
        ```py
        from datachain.lib.namespaces import create as create_namespace
        namespace = create_namespace("dev", "Dev namespace")
        ```
    """
    session = Session.get(session)

    from datachain.lib.dc.utils import is_studio

    if not is_studio():
        raise NamespaceCreateNotAllowedError("Creating namespace is not allowed")

    Namespace.validate_name(name)

    return session.catalog.metastore.create_namespace(name, descr)


def get(name: str, session: Session | None = None) -> Namespace:
    """
    Gets a namespace by name.
    If the namespace is not found, a `NamespaceNotFoundError` is raised.

    Parameters:
        name : The name of the namespace.
        session : Session to use for getting namespace.

    Example:
        ```py
        import datachain as dc
        namespace = dc.get_namespace("local")
        ```
    """
    session = Session.get(session)
    return session.catalog.metastore.get_namespace(name)


def ls(session: Session | None = None) -> list[Namespace]:
    """
    Gets a list of all namespaces.

    Parameters:
        session : Session to use for getting namespaces.

    Example:
        ```py
        from datachain.lib.namespaces import ls as ls_namespaces
        namespaces = ls_namespaces()
        ```
    """
    return Session.get(session).catalog.metastore.list_namespaces()


def delete_namespace(name: str, session: Session | None = None) -> None:
    """
    Removes a namespace by name.

    Raises:
        NamespaceNotFoundError: If the namespace does not exist.
        NamespaceDeleteNotAllowedError: If the namespace is non-empty,
            is the default namespace, or is a system namespace,
            as these cannot be removed.

    Parameters:
        name: The name of the namespace.
        session: Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        dc.delete_namespace("dev")
        ```
    """
    session = Session.get(session)
    metastore = session.catalog.metastore

    namespace_name, project_name = parse_name(name)

    if project_name:
        return delete_project(project_name, namespace_name, session)

    namespace = metastore.get_namespace(name)

    if name == metastore.system_namespace_name:
        raise NamespaceDeleteNotAllowedError(
            f"Namespace {metastore.system_namespace_name} cannot be removed"
        )

    if name == metastore.default_namespace_name:
        raise NamespaceDeleteNotAllowedError(
            f"Namespace {metastore.default_namespace_name} cannot be removed"
        )

    num_projects = metastore.count_projects(namespace.id)
    if num_projects > 0:
        raise NamespaceDeleteNotAllowedError(
            f"Namespace cannot be removed. It contains {num_projects} project(s). "
            "Please remove the project(s) first."
        )

    metastore.remove_namespace(namespace.id)
