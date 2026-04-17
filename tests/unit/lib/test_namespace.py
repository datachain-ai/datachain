import pytest

from datachain.error import (
    InvalidNamespaceNameError,
    NamespaceCreateNotAllowedError,
    NamespaceDeleteNotAllowedError,
    NamespaceNotFoundError,
)
from datachain.lib.namespaces import create as create_namespace
from datachain.lib.namespaces import delete_namespace
from datachain.lib.namespaces import get as get_namespace
from datachain.lib.namespaces import ls as ls_namespaces
from datachain.lib.projects import create as create_project
from tests.utils import skip_if_not_sqlite


# Default to is_studio=True for namespace/project CRUD; individual tests may override
@pytest.fixture
def is_studio():
    return True


@pytest.fixture
def dev_namespace(test_session):
    return create_namespace("dev", "Dev namespace")


def test_create_namespace(test_session):
    namespace = create_namespace("dev", session=test_session)
    assert namespace.name
    assert namespace.id
    assert namespace.uuid
    assert namespace.created_at


@pytest.mark.parametrize("is_studio", [False])
@skip_if_not_sqlite
def test_create_by_user_not_allowed(test_session, is_studio):
    with pytest.raises(NamespaceCreateNotAllowedError) as excinfo:
        create_namespace("dev", session=test_session)

    assert str(excinfo.value) == "Creating namespace is not allowed"


def test_create_namespace_already_exists(test_session):
    namespace1 = create_namespace("dev", session=test_session)
    namespace2 = create_namespace("dev", session=test_session)
    assert namespace1.id == namespace2.id


@pytest.mark.parametrize("name", ["local", "with.dots", ""])
def test_invalid_name(test_session, name):
    with pytest.raises(InvalidNamespaceNameError):
        create_namespace(name, session=test_session)


def test_get_namespace(test_session, dev_namespace):
    namespace = get_namespace("dev", session=test_session)
    assert namespace.id == dev_namespace.id
    assert namespace.uuid == dev_namespace.uuid
    assert namespace.name == dev_namespace.name
    assert namespace.descr == dev_namespace.descr
    assert namespace.created_at == dev_namespace.created_at


def test_get_namespace_not_found(test_session, dev_namespace):
    with pytest.raises(NamespaceNotFoundError) as excinfo:
        get_namespace("wrong", session=test_session)

    assert str(excinfo.value) == "Namespace wrong not found."


def test_ls_namespaces(test_session):
    system_namespace_name = test_session.catalog.metastore.system_namespace_name

    create_namespace("ns1")
    create_namespace("ns2")

    namespaces = ls_namespaces(session=test_session)
    assert sorted([n.name for n in namespaces]) == sorted(
        [system_namespace_name, "ns1", "ns2"]
    )


def test_ls_namespaces_just_local(test_session):
    system_namespace_name = test_session.catalog.metastore.system_namespace_name
    namespaces = ls_namespaces(session=test_session)
    assert [n.name for n in namespaces] == [system_namespace_name]


def test_delete_namespace(test_session, dev_namespace):
    delete_namespace(dev_namespace.name, session=test_session)
    with pytest.raises(NamespaceNotFoundError):
        get_namespace(dev_namespace.name, session=test_session)


def test_delete_namespace_no_session():
    create_namespace("dev", "Dev namespace")
    delete_namespace("dev")
    with pytest.raises(NamespaceNotFoundError):
        get_namespace("dev")


def test_delete_namespace_not_found(test_session, dev_namespace):
    with pytest.raises(NamespaceNotFoundError):
        delete_namespace("missing", session=test_session)


def test_delete_namespace_system(test_session):
    metastore = test_session.catalog.metastore
    with pytest.raises(NamespaceDeleteNotAllowedError) as excinfo:
        delete_namespace(metastore.system_namespace_name, session=test_session)
    assert str(excinfo.value) == (
        f"Namespace {metastore.system_namespace_name} cannot be removed"
    )


def test_delete_namespace_default(test_session):
    metastore = test_session.catalog.metastore
    metastore.create_namespace(metastore.default_namespace_name, validate=False)
    with pytest.raises(NamespaceDeleteNotAllowedError) as excinfo:
        delete_namespace(metastore.default_namespace_name, session=test_session)
    assert str(excinfo.value) == (
        f"Namespace {metastore.default_namespace_name} cannot be removed"
    )


def test_delete_namespace_non_empty(test_session, dev_namespace):
    create_project(dev_namespace.name, "my-project")

    with pytest.raises(NamespaceDeleteNotAllowedError) as excinfo:
        delete_namespace(dev_namespace.name, session=test_session)

    assert str(excinfo.value) == (
        "Namespace cannot be removed. It contains 1 project(s)."
        " Please remove the project(s) first."
    )
