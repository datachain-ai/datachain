import pytest

import datachain as dc
from datachain.error import (
    InvalidProjectNameError,
    ProjectCreateNotAllowedError,
    ProjectDeleteNotAllowedError,
    ProjectNotFoundError,
)
from datachain.lib.namespaces import create as create_namespace
from datachain.lib.namespaces import delete_namespace
from datachain.lib.namespaces import get as get_namespace
from datachain.lib.projects import get as get_project
from datachain.lib.projects import ls as ls_projects
from tests.utils import skip_if_not_sqlite


# Default to is_studio=True for namespace/project CRUD; individual tests may override
@pytest.fixture
def is_studio():
    return True


@pytest.fixture
def dev_namespace(test_session):
    return create_namespace("dev", "Dev namespace")


@pytest.fixture
def chatbot_project(test_session):
    return dc.create_project("dev", "chatbot", "Chatbot project")


@pytest.mark.parametrize("namespace_created_upfront", (True, False))
def test_create_project(test_session, namespace_created_upfront):
    if namespace_created_upfront:
        create_namespace("dev")

    project = dc.create_project("dev", "chatbot", session=test_session)
    assert project.id
    assert project.uuid
    assert project.created_at
    assert project.name == "chatbot"
    assert project.namespace == get_namespace("dev", session=test_session)


def test_create_project_that_already_exists_in_namespace(test_session):
    name = "chatbot"
    dc.create_project("dev", name, "desc 1", session=test_session)
    project = dc.create_project("dev", name, "desc 2", session=test_session)
    assert project.descr == "desc 1"


def test_create_project_with_the_same_name_in_different_namespace(test_session):
    name = "chatbot"

    dev_project = dc.create_project("dev", name, "Dev chatbot", session=test_session)
    prod_project = dc.create_project("prod", name, "Prod chatbot", session=test_session)

    assert dev_project.name == name
    assert dev_project.descr == "Dev chatbot"
    assert prod_project.name == name
    assert prod_project.descr == "Prod chatbot"


@pytest.mark.parametrize("name", ["local", "with.dots", ""])
def test_invalid_name(test_session, name):
    with pytest.raises(InvalidProjectNameError):
        dc.create_project("dev", name, session=test_session)


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", [False])
def test_create_by_user_not_allowed(test_session):
    with pytest.raises(ProjectCreateNotAllowedError) as excinfo:
        dc.create_project("dev", "chatbot", session=test_session)

    assert str(excinfo.value) == "Creating project is not allowed"


def test_get_project(test_session, chatbot_project, dev_namespace):
    project = get_project(
        chatbot_project.name, dev_namespace.name, session=test_session
    )
    assert project.id == chatbot_project.id
    assert project.uuid == chatbot_project.uuid
    assert project.name == chatbot_project.name
    assert project.descr == chatbot_project.descr
    assert project.created_at == chatbot_project.created_at
    assert project.namespace == dev_namespace


def test_get_project_not_found(test_session, dev_namespace):
    with pytest.raises(ProjectNotFoundError) as excinfo:
        get_project("wrong", dev_namespace.name, session=test_session)

    assert str(excinfo.value) == "Project wrong in namespace dev not found."


def test_get_project_not_found_but_exists_in_other_namespace(
    test_session, dev_namespace, chatbot_project
):
    name = "images"
    dc.create_project("prod", name, session=test_session)

    get_project(name, "prod", session=test_session)
    with pytest.raises(ProjectNotFoundError) as excinfo:
        get_project(name, dev_namespace.name, session=test_session)

    assert str(excinfo.value) == f"Project {name} in namespace dev not found."


def test_ls_projects(test_session):
    metastore = test_session.catalog.metastore

    p_names = ["p1", "p2", "p3"]
    for name in p_names:
        dc.create_project("ns1", name, "", session=test_session)
        dc.create_project("ns2", name, "", session=test_session)

    projects = ls_projects(session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            (metastore.system_namespace_name, metastore.listing_project_name),
            ("ns1", "p1"),
            ("ns1", "p2"),
            ("ns1", "p3"),
            ("ns2", "p1"),
            ("ns2", "p2"),
            ("ns2", "p3"),
        ]
    )


def test_ls_projects_one_namespace(test_session):
    p_names = ["p1", "p2", "p3"]
    for name in p_names:
        dc.create_project("ns1", name, "", session=test_session)
        dc.create_project("ns2", name, "", session=test_session)

    projects = ls_projects("ns1", session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            ("ns1", "p1"),
            ("ns1", "p2"),
            ("ns1", "p3"),
        ]
    )


def test_ls_projects_just_default(test_session):
    metastore = test_session.catalog.metastore

    projects = ls_projects(session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            (metastore.system_namespace_name, metastore.listing_project_name),
        ]
    )


def test_ls_projects_empty_in_namespace(test_session):
    create_namespace("ns1")
    projects = ls_projects("ns1", session=test_session)
    assert [(p.namespace.name, p.name) for p in projects] == []


def test_delete_project(test_session, chatbot_project, dev_namespace):
    delete_namespace(
        f"{dev_namespace.name}.{chatbot_project.name}", session=test_session
    )
    with pytest.raises(ProjectNotFoundError):
        get_project(chatbot_project.name, dev_namespace.name, session=test_session)

    # namespace should not be deleted
    get_namespace(dev_namespace.name, session=test_session)


def test_delete_project_not_found(test_session, chatbot_project, dev_namespace):
    with pytest.raises(ProjectNotFoundError):
        delete_namespace(f"{dev_namespace.name}.missing", session=test_session)


def test_delete_project_listing(test_session):
    metastore = test_session.catalog.metastore
    with pytest.raises(ProjectDeleteNotAllowedError) as excinfo:
        delete_namespace(
            f"{metastore.system_namespace_name}.{metastore.listing_project_name}",
            session=test_session,
        )
    assert str(excinfo.value) == (
        f"Project {metastore.listing_project_name} cannot be removed"
    )


def test_delete_project_default(test_session):
    metastore = test_session.catalog.metastore
    with pytest.raises(ProjectDeleteNotAllowedError) as excinfo:
        delete_namespace(
            f"{metastore.default_namespace_name}.{metastore.default_project_name}",
            session=test_session,
        )
    assert str(excinfo.value) == (
        f"Project {metastore.default_project_name} cannot be removed"
    )


def test_delete_project_non_empty(
    test_session, studio_job, chatbot_project, dev_namespace
):
    (
        dc.read_values(num=[1, 2, 3])
        .settings(namespace=dev_namespace.name, project=chatbot_project.name)
        .save("numbers")
    )

    with pytest.raises(ProjectDeleteNotAllowedError) as excinfo:
        delete_namespace(
            f"{dev_namespace.name}.{chatbot_project.name}", session=test_session
        )

    assert str(excinfo.value) == (
        "Project cannot be removed. It contains 1 dataset(s)."
        " Please remove the dataset(s) first."
    )
