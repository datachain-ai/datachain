import pytest

from datachain.lib.settings import Settings, SettingsError


def test_settings_default_values():
    settings = Settings()
    assert settings.cache is False  # Default cache value
    assert settings.prefetch == 2  # Default prefetch value
    assert settings.parallel is None
    assert settings.workers is None
    assert settings.namespace is None
    assert settings.project is None
    assert settings.min_task_size is None
    assert settings.batch_size is None

    d = settings.to_dict()
    assert d == {}


def test_settings_add():
    base = Settings(
        cache=True,
        parallel=2,
        workers=4,
        min_task_size=10,
        prefetch=3,
        namespace="base",
        project="base_project",
        batch_size=1000,
    )

    # add empty settings does nothing
    base.add(Settings())
    assert base.cache is True
    assert base.parallel == 2
    assert base.workers == 4
    assert base.min_task_size == 10
    assert base.prefetch == 3
    assert base.namespace == "base"
    assert base.project == "base_project"
    assert base.batch_size == 1000

    # add new settings overrides existing ones
    base.add(
        Settings(
            cache=False,
            parallel=4,
            workers=2,
            min_task_size=20,
            prefetch=5,
            namespace="new",
            project="new_project",
            batch_size=500,
        )
    )
    assert base.cache is False
    assert base.parallel == 4
    assert base.workers == 2
    assert base.min_task_size == 20
    assert base.prefetch == 5
    assert base.namespace == "new"
    assert base.project == "new_project"
    assert base.batch_size == 500


def test_settings_to_dict():
    # Empty settings
    settings = Settings()
    assert settings.to_dict() == {}

    # All values are None, so empty dict
    settings = Settings(
        cache=None,
        prefetch=None,
        parallel=None,
        workers=None,
        min_task_size=None,
        namespace=None,
        project=None,
        batch_size=None,
    )
    assert settings.to_dict() == {}

    # All parameters set
    settings = Settings(
        cache=True,
        prefetch=3,
        parallel=4,
        workers=2,
        min_task_size=10,
        namespace="test_namespace",
        project="test_project",
        batch_size=1000,
    )
    assert settings.to_dict() == {
        "cache": True,
        "prefetch": 3,
        "parallel": 4,
        "workers": 2,
        "min_task_size": 10,
        "namespace": "test_namespace",
        "project": "test_project",
        "batch_size": 1000,
    }

    # Settings with only some values set
    settings = Settings(cache=True, workers=5)
    d = settings.to_dict()
    expected = {"cache": True, "workers": 5}
    assert d == expected

    # Mixed None and set values
    settings = Settings(
        cache=True,
        prefetch=False,
        parallel=None,
        workers=2,
        min_task_size=None,
        namespace=None,
        project="test_project",
        batch_size=None,
    )
    assert settings.to_dict() == {
        "cache": True,
        "prefetch": 0,
        "workers": 2,
        "project": "test_project",
    }


def test_settings_immutability():
    # Create original settings
    original = Settings(batch_size=1000, cache=True)

    # Create a copy
    copy_settings = Settings()
    copy_settings.add(original)

    # Modify original
    original.add(Settings(batch_size=2000))

    # Copy should not be affected
    assert copy_settings.batch_size == 1000
    assert original.batch_size == 2000


def test_settings_add_with_none_values():
    base = Settings(cache=True, workers=5, prefetch=3)

    # Add settings with None values
    to_add = Settings(cache=None, workers=None, prefetch=None)
    base.add(to_add)

    # Values should remain unchanged since None values don't override
    assert base.cache is True
    assert base.workers == 5
    assert base.prefetch == 3

    # Add settings with explicit values
    to_add = Settings(cache=False, workers=10, prefetch=5)
    base.add(to_add)

    # Values should be updated
    assert base.cache is False
    assert base.workers == 10
    assert base.prefetch == 5


def test_settings_error_messages():
    # Test various invalid parameter types
    with pytest.raises(SettingsError) as exc_info:
        Settings(cache="invalid")
    assert "cache" in str(exc_info.value)
    assert "bool" in str(exc_info.value)

    with pytest.raises(SettingsError) as exc_info:
        Settings(parallel="invalid")
    assert "parallel" in str(exc_info.value)
    assert "int or bool" in str(exc_info.value)

    with pytest.raises(SettingsError) as exc_info:
        Settings(batch_size="invalid")
    assert "batch_size" in str(exc_info.value)
    assert "int" in str(exc_info.value)

    with pytest.raises(SettingsError) as exc_info:
        Settings(namespace=123)
    assert "namespace" in str(exc_info.value)
    assert "str" in str(exc_info.value)


def test_settings_cache_parameter():
    # Default cache value
    settings = Settings()
    assert settings.cache is False

    # Custom cache values
    settings = Settings(cache=True)
    assert settings.cache is True

    settings = Settings(cache=False)
    assert settings.cache is False

    # Invalid cache type
    with pytest.raises(SettingsError):
        Settings(cache="invalid")

    with pytest.raises(SettingsError):
        Settings(cache=1)

    # None is allowed
    settings = Settings(cache=None)
    assert settings.cache is False  # Default value when None

    # to_dict method
    d = Settings(cache=True).to_dict()
    assert d["cache"] is True


def test_settings_parallel_parameter():
    # Default parallel value
    settings = Settings()
    assert settings.parallel is None

    # Boolean values
    settings = Settings(parallel=True)
    assert settings.parallel is True

    settings = Settings(parallel=False)
    assert settings.parallel is None  # False becomes None

    # Integer values
    settings = Settings(parallel=1)
    assert settings.parallel == 1

    settings = Settings(parallel=4)
    assert settings.parallel == 4

    # Invalid parallel values
    with pytest.raises(SettingsError):
        Settings(parallel=0)

    with pytest.raises(SettingsError):
        Settings(parallel=-1)

    with pytest.raises(SettingsError):
        Settings(parallel="invalid")

    # None is allowed
    settings = Settings(parallel=None)
    assert settings.parallel is None

    # to_dict method
    d = Settings(parallel=8).to_dict()
    assert d["parallel"] == 8

    d = Settings(parallel=True).to_dict()
    assert d["parallel"] is True


def test_settings_workers_parameter():
    # Default workers value
    settings = Settings()
    assert settings.workers is None

    # Valid workers values
    settings = Settings(workers=1)
    assert settings.workers == 1

    settings = Settings(workers=10)
    assert settings.workers == 10

    # Boolean values are rejected
    with pytest.raises(SettingsError):
        Settings(workers=True)
    with pytest.raises(SettingsError):
        Settings(workers=False)

    # Zero and negative values are rejected
    with pytest.raises(SettingsError):
        Settings(workers=0)

    with pytest.raises(SettingsError):
        Settings(workers=-1)

    # Invalid workers type
    with pytest.raises(SettingsError):
        Settings(workers="invalid")

    with pytest.raises(SettingsError):
        Settings(workers=1.5)

    # None is allowed
    settings = Settings(workers=None)
    assert settings.workers is None

    # to_dict method
    d = Settings(workers=5).to_dict()
    assert d["workers"] == 5


def test_settings_min_task_size_parameter():
    # Default min_task_size value
    settings = Settings()
    assert settings.min_task_size is None

    # Valid min_task_size values
    settings = Settings(min_task_size=1)
    assert settings.min_task_size == 1

    settings = Settings(min_task_size=100)
    assert settings.min_task_size == 100

    # Boolean values are rejected
    with pytest.raises(SettingsError):
        Settings(min_task_size=True)
    with pytest.raises(SettingsError):
        Settings(min_task_size=False)

    # Zero and negative values are rejected
    with pytest.raises(SettingsError):
        Settings(min_task_size=0)

    with pytest.raises(SettingsError):
        Settings(min_task_size=-1)

    # Invalid min_task_size type
    with pytest.raises(SettingsError):
        Settings(min_task_size="invalid")

    with pytest.raises(SettingsError):
        Settings(min_task_size=1.5)

    # None is allowed
    settings = Settings(min_task_size=None)
    assert settings.min_task_size is None

    # to_dict method
    d = Settings(min_task_size=50).to_dict()
    assert d["min_task_size"] == 50


def test_settings_prefetch_parameter():
    # Default prefetch value
    settings = Settings()
    assert settings.prefetch == 2  # Default prefetch value

    # Boolean values
    settings = Settings(prefetch=True)
    assert settings.prefetch == 2  # True becomes 2 (default value)

    settings = Settings(prefetch=False)
    assert settings.prefetch == 0  # False becomes 0 (disabled)

    # Integer values
    settings = Settings(prefetch=0)
    assert settings.prefetch == 0

    settings = Settings(prefetch=1)
    assert settings.prefetch == 1

    settings = Settings(prefetch=5)
    assert settings.prefetch == 5

    # Invalid prefetch values
    with pytest.raises(SettingsError):
        Settings(prefetch=-1)

    with pytest.raises(SettingsError):
        Settings(prefetch="invalid")

    # None is allowed
    settings = Settings(prefetch=None)
    assert settings.prefetch == 2  # Default value when None

    # to_dict method
    d = Settings(prefetch=3).to_dict()
    assert d["prefetch"] == 3

    d = Settings(prefetch=False).to_dict()
    assert d["prefetch"] == 0


def test_settings_namespace_parameter():
    # Default namespace value
    settings = Settings()
    assert settings.namespace is None

    # Valid namespace values
    settings = Settings(namespace="my_namespace")
    assert settings.namespace == "my_namespace"

    settings = Settings(namespace="")
    assert settings.namespace == ""

    # Invalid namespace type
    with pytest.raises(SettingsError):
        Settings(namespace=123)

    with pytest.raises(SettingsError):
        Settings(namespace=True)

    # None is allowed
    settings = Settings(namespace=None)
    assert settings.namespace is None

    # to_dict method
    d = Settings(namespace="test").to_dict()
    assert d["namespace"] == "test"


def test_settings_project_parameter():
    # Default project value
    settings = Settings()
    assert settings.project is None

    # Valid project values
    settings = Settings(project="my_project")
    assert settings.project == "my_project"

    settings = Settings(project="")
    assert settings.project == ""

    # Invalid project type
    with pytest.raises(SettingsError):
        Settings(project=123)

    with pytest.raises(SettingsError):
        Settings(project=True)

    # None is allowed
    settings = Settings(project=None)
    assert settings.project is None

    # to_dict method
    d = Settings(project="test_project").to_dict()
    assert d["project"] == "test_project"


def test_settings_batch_size_parameter():
    # Default values
    settings = Settings()
    assert settings.batch_size is None

    # Custom values
    settings = Settings(batch_size=500)
    assert settings.batch_size == 500

    # Invalid batch_size
    with pytest.raises(SettingsError):
        Settings(batch_size="invalid")

    with pytest.raises(SettingsError):
        Settings(batch_size=0)

    with pytest.raises(SettingsError):
        Settings(batch_size=-10)

    # Boolean values are now explicitly rejected
    with pytest.raises(SettingsError):
        Settings(batch_size=True)
    with pytest.raises(SettingsError):
        Settings(batch_size=False)

    # None is allowed
    settings = Settings(batch_size=None)
    assert settings.batch_size is None

    # to_dict method
    d = Settings(batch_size=500).to_dict()
    assert d["batch_size"] == 500
