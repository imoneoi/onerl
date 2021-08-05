import importlib


def get_class_from_str(import_name: str, class_name: str):
    if import_name:
        class_inst = getattr(importlib.import_module(import_name), class_name)
    else:
        class_inst = globals()[class_name]

    return class_inst
