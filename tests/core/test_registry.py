"""Tests for the Registry class and the generator/transform registry instances."""
import pytest

from dsl.core.extension.registry import Registry


def test_register_and_get_round_trip():
    registry = Registry("widget")

    @registry.register("foo")
    class Foo:
        pass

    assert registry.get("foo") is Foo


def test_decorator_returns_class_unchanged():
    registry = Registry("widget")

    @registry.register("foo")
    class Foo:
        pass

    # The decorator's job is a side effect (recording); the class itself
    # must come back untouched so it can still be used normally.
    assert Foo.__name__ == "Foo"


def test_duplicate_name_raises():
    registry = Registry("widget")

    @registry.register("foo")
    class Foo:
        pass

    with pytest.raises(ValueError, match="'foo' is already registered"):

        @registry.register("foo")
        class Bar:
            pass


def test_get_unknown_raises_and_lists_available():
    registry = Registry("widget")

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    with pytest.raises(KeyError) as excinfo:
        registry.get("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "alpha" in message and "beta" in message
