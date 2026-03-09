from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum

from tomlkit import TOMLDocument, dump, load

from datachain.utils import DataChainDir, global_config_dir, system_config_dir


# Define an enum with value system, global and local
class ConfigLevel(Enum):
    SYSTEM = "system"
    GLOBAL = "global"
    LOCAL = "local"


class Config:
    SYSTEM_LEVELS = (ConfigLevel.SYSTEM, ConfigLevel.GLOBAL)
    LOCAL_LEVELS = (ConfigLevel.LOCAL,)

    # In the order of precedence
    LEVELS = SYSTEM_LEVELS + LOCAL_LEVELS

    def __init__(self, level: ConfigLevel | None = None):
        self.level = level

        self.init()

    @classmethod
    def get_dir(cls, level: ConfigLevel | None) -> str:
        if level == ConfigLevel.SYSTEM:
            return system_config_dir()
        if level == ConfigLevel.GLOBAL:
            return global_config_dir()

        return str(DataChainDir.find().root)

    def init(self):
        d = DataChainDir(self.get_dir(self.level))
        d.init()

    def load_one(self, level: ConfigLevel | None = None) -> TOMLDocument:
        config_path = DataChainDir(self.get_dir(level)).config

        try:
            with open(config_path, encoding="utf-8") as f:
                return load(f)
        except FileNotFoundError:
            return TOMLDocument()

    def load_config_to_level(self) -> TOMLDocument:
        merged_conf = TOMLDocument()

        for merge_level in self.LEVELS:
            if merge_level == self.level:
                break
            config = self.load_one(merge_level)
            if config:
                merge(merged_conf, config)

        return merged_conf

    def read(self) -> TOMLDocument:
        if self.level is None:
            return self.load_config_to_level()
        return self.load_one(self.level)

    @contextmanager
    def edit(self):
        config = self.load_one(self.level)
        yield config

        self.write(config)

    def config_file(self):
        return DataChainDir(self.get_dir(self.level)).config

    def write(self, config: TOMLDocument):
        with open(self.config_file(), "w") as f:
            dump(config, f)

    def get_remote_config(self, remote: str = "") -> Mapping[str, str]:
        config = self.read()

        if not config:
            return {"type": "local"}
        if not remote:
            try:
                remote = config["core"]["default-remote"]  # type: ignore[index,assignment]
            except KeyError:
                return {"type": "local"}
        try:
            remote_conf: Mapping[str, str] = config["remote"][remote]  # type: ignore[assignment,index]
        except KeyError:
            raise Exception(
                f"missing config section for default remote: remote.{remote}"
            ) from None
        except Exception as exc:
            raise Exception("invalid config") from exc

        if not isinstance(remote_conf, Mapping):
            raise TypeError(f"config section remote.{remote} must be a mapping")

        remote_type = remote_conf.get("type")
        if remote_type not in ("local", "http"):
            raise Exception(
                f'config section remote.{remote} must have "type" with one of: '
                '"local", "http"'
            )

        if remote_type == "http":
            for key in ["url", "username", "token"]:
                try:
                    remote_conf[key]
                except KeyError:
                    raise Exception(
                        f"config section remote.{remote} of type {remote_type} "
                        f"must contain key {key}"
                    ) from None
        elif remote_type != "local":
            raise Exception(
                f"config section remote.{remote} has invalid remote type {remote_type}"
            )
        return remote_conf


def merge(into: TOMLDocument | dict, update: TOMLDocument | dict):
    """Merges second dict into first recursively"""
    for key, val in update.items():
        if isinstance(into.get(key), dict) and isinstance(val, dict):
            merge(into[key], val)  # type: ignore[arg-type]
        else:
            into[key] = val
