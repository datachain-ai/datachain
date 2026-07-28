import os

import pytest
from tomlkit import TOMLDocument, dump

from datachain.config import Config, ConfigLevel
from datachain.utils import DataChainDir


@pytest.fixture(autouse=True)
def current_dir(monkeypatch, tmp_dir):
    monkeypatch.chdir(tmp_dir)
    yield str(tmp_dir)


def create_global_config(global_config_dir):
    conf = TOMLDocument()
    conf["level"] = "global"
    conf["global"] = True
    conf["studio"] = {
        "url": "https://studio.datachain.ai",
        "token": "global-token",
        "global-conf": "exists",
    }

    with open(DataChainDir(global_config_dir).config, "w") as f:
        dump(conf, f)


def create_system_config(system_config_dir):
    conf = TOMLDocument()
    conf["level"] = "system"
    conf["system"] = True
    conf["studio"] = {
        "url": "https://studio.datachain.ai",
        "token": "system-token",
        "system-conf": "exists",
    }

    with open(DataChainDir(system_config_dir).config, "w") as f:
        dump(conf, f)


def create_local_config(current_dir):
    conf = TOMLDocument()
    conf["level"] = "local"
    conf["local"] = True
    conf["studio"] = {
        "url": "https://studio.datachain.ai",
        "token": "local-token",
        "local-conf": "exists",
    }

    with open(DataChainDir(os.path.join(current_dir, ".datachain")).config, "w") as f:
        dump(conf, f)


def test_get_dir(global_config_dir, system_config_dir, current_dir):
    assert Config.get_dir(ConfigLevel.GLOBAL) == global_config_dir
    assert Config.get_dir(ConfigLevel.SYSTEM) == system_config_dir

    assert Config.get_dir(ConfigLevel.LOCAL) == os.path.join(current_dir, ".datachain")


def test_read_config(global_config_dir, system_config_dir, current_dir):
    config = Config()

    # Test for empty config
    assert config.read() == TOMLDocument()

    # Add system config
    create_system_config(system_config_dir)
    assert config.read() == {
        "level": "system",
        "system": True,
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "system-token",
            "system-conf": "exists",
        },
    }

    # Add global config
    create_global_config(global_config_dir)
    # Both system and global config are merged.

    assert config.read() == {
        "level": "global",
        "global": True,
        "system": True,
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "global-token",
            "global-conf": "exists",
            "system-conf": "exists",
        },
    }

    # Add local config
    create_local_config(current_dir)

    # All configs are merged
    assert config.read() == {
        "level": "local",
        "global": True,
        "system": True,
        "local": True,
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "local-token",
            "global-conf": "exists",
            "system-conf": "exists",
            "local-conf": "exists",
        },
    }

    # Get the config for a single level only
    assert Config(ConfigLevel.GLOBAL).read() == {
        "level": "global",
        "global": True,
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "global-token",
            "global-conf": "exists",
        },
    }


def test_edit_config_local_level(current_dir):
    # Assert the local config is empty first
    assert Config().read() == {}

    create_local_config(current_dir)

    # Edit the local config
    with Config().edit() as config:
        config["new_key"] = "new_value"
        config["studio"]["token"] = "new-token"  # noqa: S105 # nosec B105

    # Assert the local config is updated
    assert Config().read() == {
        "level": "local",
        "local": True,
        "new_key": "new_value",
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "new-token",
            "local-conf": "exists",
        },
    }


def test_edit_config_global_level(global_config_dir):
    # Assert the global config is empty first
    assert Config(ConfigLevel.GLOBAL).read() == {}

    create_global_config(global_config_dir)

    with Config(ConfigLevel.GLOBAL).edit() as config:
        config["new_key"] = "new_value"
        config["studio"]["token"] = "new-token"  # noqa: S105 # nosec B105

    # Assert the global config is updated
    assert Config(ConfigLevel.GLOBAL).read() == {
        "level": "global",
        "global": True,
        "new_key": "new_value",
        "studio": {
            "url": "https://studio.datachain.ai",
            "token": "new-token",
            "global-conf": "exists",
        },
    }
