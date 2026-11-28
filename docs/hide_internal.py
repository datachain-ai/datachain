# ruff: noqa: INP001
"""Griffe extension: leave members marked with `datachain.utils.internal` out of
the API reference."""

import ast

import griffe


def _is_internal(node) -> bool:
    return any(
        (d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None))
        == "internal"
        for d in getattr(node, "decorator_list", [])
    )


class HideInternal(griffe.Extension):
    def on_function_instance(self, *, node, func, **kwargs) -> None:
        if _is_internal(node):
            func.extra["internal"] = True

    def on_attribute_instance(self, *, node, attr, **kwargs) -> None:
        # Griffe turns properties into attributes, without their decorators.
        if _is_internal(node):
            attr.extra["internal"] = True

    def on_class_members(self, *, cls, **kwargs) -> None:
        for name, member in list(cls.members.items()):
            if member.extra.get("internal"):
                cls.del_member(name)
