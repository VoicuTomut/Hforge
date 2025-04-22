"""
Functions for nicer prints
"""
import pprint

def pretty_print_dict(dictionary):
    """
    Pretty prints a dictionary with a clear and readable format.

    Args:
        dictionary (dict): The dictionary to be pretty printed.

    Returns:
        None
    """
    if not isinstance(dictionary, dict):
        raise TypeError("Input must be a dictionary")

    pp = pprint.PrettyPrinter(indent=4, width=80, sort_dicts=True)
    pp.pprint(dictionary)