import importlib

def get_interaction_block(class_name, module="hforge.mace.modules"):
    try:
        return getattr(importlib.import_module(module), class_name)
    except AttributeError:
        return None  # Or raise an error if you prefer