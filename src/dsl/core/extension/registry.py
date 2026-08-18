"""A name -> class lookup used to plug features in without editing the engine.

Maps the string written in YAML (e.g. ``"seasonal_spike"``) to the class
implementing it. One class shared by the generator and transform registries.
"""


class Registry:
    """Maps DSL names (strings from YAML) to the classes that implement them."""

    def __init__(self, kind: str):
        self.kind = kind  # only for error messages ("generator"/"transform")
        self._items: dict[str, type] = {}

    def register(self, name: str):
        """Class decorator: ``@registry.register("foo")`` stores the class."""

        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return self._items[name]
