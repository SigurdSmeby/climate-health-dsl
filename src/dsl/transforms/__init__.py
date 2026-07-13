"""Auto-import every transform module so each one registers itself on import."""
import importlib
import pkgutil

for _finder, module_name, _is_pkg in pkgutil.iter_modules(__path__):
    if not module_name.startswith("_"):
        importlib.import_module(f"{__name__}.{module_name}")
