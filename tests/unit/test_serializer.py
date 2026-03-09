import base64
import json
from collections.abc import Callable
from typing import Any

import pytest

from datachain.data_storage.serializer import (
    CallableRegistry,
    Serializable,
    deserialize,
)


class MySerializableInit(Serializable):
    def __init__(self, name, optional=None):
        self.name = name
        self.optional = optional

    @classmethod
    def serialize_callable_name(cls):
        return "MySerializableInit"

    @classmethod
    def build(cls, name, optional=None):
        return cls(name, optional=optional)

    def clone_params(self):
        return self.__class__.build, [self.name], {"optional": self.optional}

    def get_params(self):
        return self.name, self.optional


class MySerializableFunc(Serializable):
    def __init__(self, name, optional=None):
        self.name = name
        self.optional = optional

    @classmethod
    def from_params(cls, name, optional=None):
        return cls(name, optional=optional)

    @classmethod
    def serialize_callable_name(cls):
        return "MySerializableFunc.from_params"

    def clone_params(self):
        return self.from_params, [self.name], {"optional": self.optional}

    def get_params(self):
        return self.name, self.optional


class MySerializableNoParams(Serializable):
    @classmethod
    def serialize_callable_name(cls):
        return "MySerializableNoParams"

    def clone_params(self):
        return self.__class__.build, [], {}

    @classmethod
    def build(cls):
        return cls()


# Register test classes/functions for the serializer with explicit names
CallableRegistry.register(MySerializableInit.build, "MySerializableInit")
CallableRegistry.register(
    MySerializableFunc.from_params, "MySerializableFunc.from_params"
)
CallableRegistry.register(MySerializableNoParams.build, "MySerializableNoParams")


@pytest.mark.parametrize(
    "cls,call,call_name",
    [
        (MySerializableInit, MySerializableInit.build, "MySerializableInit"),
        (
            MySerializableFunc,
            MySerializableFunc.from_params,
            "MySerializableFunc.from_params",
        ),
    ],
)
@pytest.mark.parametrize(
    "name,optional",
    [
        (None, None),
        ("foo", None),
        (None, 12),
        ("bar", 24),
    ],
)
def test_serializable_json_format(cls, call, call_name, name, optional):
    """Test the new JSON-based serialization format."""
    obj = cls(name, optional=optional)
    assert obj.clone_params() == (call, [name], {"optional": optional})

    # Test new JSON serialization
    serialized = obj.serialize()
    assert serialized

    # Verify it's JSON format by decoding
    serialized_decoded = base64.b64decode(serialized.encode())
    data = json.loads(serialized_decoded.decode())
    assert data["callable"] == call_name
    assert data["args"] == [name]
    assert data["kwargs"] == {"optional": optional}

    obj2 = deserialize(serialized)
    assert isinstance(obj2, cls)
    assert obj2.name == name  # type: ignore[attr-defined]
    assert obj2.optional == optional  # type: ignore[attr-defined]
    assert obj2.get_params() == (name, optional)  # type: ignore[attr-defined]


def test_serializable_no_params():
    """Test serialization with no parameters."""
    obj = MySerializableNoParams()
    assert obj.clone_params() == (MySerializableNoParams.build, [], {})

    # Test new JSON serialization
    serialized = obj.serialize()
    assert serialized

    # Verify it's JSON format
    serialized_decoded = base64.b64decode(serialized.encode())
    data = json.loads(serialized_decoded.decode())
    assert data["callable"] == "MySerializableNoParams"
    assert data["args"] == []
    assert data["kwargs"] == {}

    obj2 = deserialize(serialized)
    assert isinstance(obj2, MySerializableNoParams)


def test_callable_registry():
    """Test the CallableRegistry functionality."""

    # Test registration
    def dummy_func():
        pass

    CallableRegistry.register(dummy_func, "dummy_func")
    assert CallableRegistry.get("dummy_func") is dummy_func

    # Test error cases
    with pytest.raises(KeyError):
        CallableRegistry.get("nonexistent")

    def unregistered_func():
        pass

    with pytest.raises(KeyError):
        CallableRegistry.get("unregistered_func")


def test_reject_unregistered_callable():
    """Ensure unregistered callable names cannot be deserialized."""
    data = {"callable": "nonexistent_callable", "args": [], "kwargs": {}}
    malicious_serialized = base64.b64encode(json.dumps(data).encode()).decode()
    with pytest.raises(KeyError):
        deserialize(malicious_serialized)


class NestedSerializable(Serializable):
    def __init__(self, value: int, child: "NestedSerializable | None" = None):
        self.value = value
        self.child = child

    @classmethod
    def factory(
        cls,
        value: int,
        child: tuple[Callable, list, dict[str, Any]] | None = None,
    ) -> "NestedSerializable":
        if child is not None:
            f, a, kw = child
            child_obj = f(*a, **kw)
        else:
            child_obj = None
        return cls(value, child_obj)

    @classmethod
    def serialize_callable_name(cls):
        return "NestedSerializable.factory"

    def clone_params(self):
        return (
            self.factory,
            [self.value],
            {"child": (self.child.clone_params() if self.child else None)},
        )


CallableRegistry.register(NestedSerializable.factory, "NestedSerializable.factory")


def test_nested_recursive_serialization():
    leaf = NestedSerializable(2)
    root = NestedSerializable(1, child=leaf)
    serialized = root.serialize()
    restored = deserialize(serialized)
    assert isinstance(restored, NestedSerializable)
    assert restored.value == 1
    assert isinstance(restored.child, NestedSerializable)
    assert restored.child.value == 2
    assert restored.child.child is None


def test_deserialize_invalid_top_level():
    bad = base64.b64encode(json.dumps({"foo": 1}).encode()).decode()
    with pytest.raises(ValueError):
        deserialize(bad)
    with pytest.raises(ValueError):
        deserialize("Zm9vYmFy")  # base64 for 'foobar'
