import importlib
import sys


def register_globalpool_aliases():
    mappings = {
        "models.GlobalPooing": "models.baseline.GlobalPooling",
        "models.GlobalPooing.vade_pooling": "models.baseline.GlobalPooling.vade_pooling",
        "models.GlobalPooing.vade_pooling.GlobalPool": (
            "models.baseline.GlobalPooling.vade_pooling.GlobalPool"
        ),
    }
    for old_name, new_name in mappings.items():
        sys.modules.setdefault(old_name, importlib.import_module(new_name))
