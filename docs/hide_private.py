# ruff: noqa: INP001
import griffe


class HidePrivate(griffe.Extension):
    """Leave out of the API reference members whose docstring has `:meta private:`."""

    def on_class_members(self, *, cls: griffe.Class, **kwargs) -> None:
        for name, member in list(cls.members.items()):
            if member.docstring and ":meta private:" in member.docstring.value:
                cls.del_member(name)
