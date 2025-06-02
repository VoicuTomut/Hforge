import importlib

import yaml


def get_object_from_module(class_name, module="hforge.mace.modules"):
    try:
        return getattr(importlib.import_module(module), class_name)
    except AttributeError:
        return None  # Or raise an error if you prefer

def load_config(path="examples/training_loop/training_loop_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)