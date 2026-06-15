import pkgutil
import importlib

# Import all submodules so decorators run
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")