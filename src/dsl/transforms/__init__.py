"""Auto-import every transform module so each one registers itself on startup."""
import importlib
import pkgutil

# Same auto-discovery as dsl.generators: importing each sibling module runs
# its @register_transform decorator, so new transform files need no edits here.
for _finder, module_name, _is_pkg in pkgutil.iter_modules(__path__):
    if not module_name.startswith("_"):
        importlib.import_module(f"{__name__}.{module_name}")
