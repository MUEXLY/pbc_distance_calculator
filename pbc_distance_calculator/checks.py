"""
module for engine checks
"""


from types import ModuleType


TENSOR_TYPES = ["tensor", "array"]
NECESSARY_METHODS_IN_PARENT = ["einsum", "round"]

NECESSARY_METHODS_IN_CHILDREN = {
    "linalg": ["inv", "norm"]
}


def is_valid(engine: ModuleType) -> bool:

    """
    method for checking if an engine is valid
    """

    parent_scope = dir(engine)

    if any(method not in parent_scope for method in NECESSARY_METHODS_IN_PARENT):
        return False

    for child_name, methods in NECESSARY_METHODS_IN_CHILDREN.items():
        submodule = getattr(engine, child_name)
        child_scope = dir(submodule)
        if any(method not in child_scope for method in methods):
            return False

    if all(tensor_type not in parent_scope for tensor_type in TENSOR_TYPES):
        return False

    return True
