"""A name -> class lookup used to plug features in without editing the engine.

A registry is just a dictionary mapping a name (the string written in the
YAML, e.g. ``"seasonal_spike"``) to the class that implements it. The
``register`` decorator adds an entry; ``get`` retrieves it. Written once as
a class so the generator and transform registries share the same logic.
"""


class Registry:
    """Maps DSL names (strings from YAML) to the classes that implement them."""

    def __init__(self, kind: str):
        # `kind` is only used to make error messages clearer ("generator"/"transform").
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        """Decorator: ``@my_registry.register("foo")`` stores the class under "foo".

        A decorator is a function that takes a class and returns it (here,
        unchanged) after running a side effect — in this case, recording it in
        the dictionary. That side effect is what lets a feature "announce itself".
        """

        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Look up a registered class by name, with a helpful error if missing."""
        if name not in self._items:
            available = sorted(self._items)
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            )
        return self._items[name]
