from datachain.error import ProjectCreateNotAllowedError, ProjectDeleteNotAllowedError
from datachain.project import Project
from datachain.query import Session


def create(
    namespace: str,
    name: str,
    descr: str | None = None,
    session: Session | None = None,
) -> Project:
    """
    Creates a new project under a specified namespace.

    Projects help organize datasets. A default project is always available,
    but users can create additional ones (only in Studio, not via CLI).


    Parameters:
        name: Name of the new project.
        namespace: Namespace to create the project in. Created if it doesn't exist.
        descr: Optional description of the project.
        session: Optional session to use for the operation.

    Example:
        ```py
        import datachain as dc
        project = dc.create_project("dev", "my-project", "My personal project")
        ```
    """
    session = Session.get(session)

    from datachain.lib.dc.utils import is_studio

    if not is_studio():
        raise ProjectCreateNotAllowedError("Creating project is not allowed")

    Project.validate_name(name)

    return session.catalog.metastore.create_project(namespace, name, descr)


def get(name: str, namespace: str, session: Session | None) -> Project:
    """
    Gets a project by name in some namespace.
    If the project is not found, a `ProjectNotFoundError` is raised.

    Parameters:
        name : The name of the project.
        namespace : The name of the namespace.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        from datachain.lib.projects import get as get_project
        project = get_project("my-project", "local")
        ```
    """
    return Session.get(session).catalog.metastore.get_project(name, namespace)


def ls(namespace: str | None = None, session: Session | None = None) -> list[Project]:
    """
    Gets a list of projects in a specific namespace or from all namespaces.

    Parameters:
        namespace : An optional namespace name.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        from datachain.lib.projects import ls as ls_projects
        local_namespace_projects = ls_projects("local")
        all_projects = ls_projects()
        ```
    """
    session = Session.get(session)
    namespace_id = None
    if namespace:
        namespace_id = session.catalog.metastore.get_namespace(namespace).id

    return session.catalog.metastore.list_projects(namespace_id)


def delete(name: str, namespace: str, session: Session | None = None) -> None:
    """
    Removes a project by name within a namespace.

    Raises:
        ProjectNotFoundError: If the project does not exist.
        ProjectDeleteNotAllowedError: If the project is non-empty,
            is the default project, or is a listing project,
            as these cannot be removed.

    Parameters:
        name : The name of the project.
        namespace : The name of the namespace.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        dc.delete_project("my-project", "local")
        ```
    """
    session = Session.get(session)
    metastore = session.catalog.metastore

    project = metastore.get_project(name, namespace)

    if metastore.is_listing_project(name, namespace):
        raise ProjectDeleteNotAllowedError(
            f"Project {metastore.listing_project_name} cannot be removed"
        )

    if metastore.is_default_project(name, namespace):
        raise ProjectDeleteNotAllowedError(
            f"Project {metastore.default_project_name} cannot be removed"
        )

    num_datasets = metastore.count_datasets(project.id)
    if num_datasets > 0:
        raise ProjectDeleteNotAllowedError(
            f"Project cannot be removed. It contains {num_datasets} dataset(s). "
            "Please remove the dataset(s) first."
        )

    metastore.remove_project(project.id)
