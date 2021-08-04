import importlib


def get_class_from_str(import_name: str, class_name: str):
    if import_name:
        module = importlib.import_module(import_name)
    else:
        module = globals()
    return module[class_name]
