"""Auto-import every generator module so each one registers itself on startup."""
import importlib
import pkgutil

# pkgutil.iter_modules lists the .py files in this folder. Importing each one
# runs its @register_generator decorator, adding it to the registry. New files
# are picked up automatically — that is the whole point of this loop.
for _finder, module_name, _is_pkg in pkgutil.iter_modules(__path__):
    if not module_name.startswith("_"):
        importlib.import_module(f"{__name__}.{module_name}")
