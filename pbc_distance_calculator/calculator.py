"""
module for getting pairwise distances
"""

from types import ModuleType
from typing import Callable
import warnings


import numpy as np
from numpy.typing import NDArray


class PrivateWarning(Warning):

    """
    warning for accessing a private method
    """


def private(func: Callable) -> Callable:

    """
    decorator for marking a function as private
    just raises a PrivateWarning upon call
    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            "This method is private and is subject to backwards-incompatible changes",
            PrivateWarning
        )
        return func(*args, **kwargs)

    return wrapper


@private
def _get_difference_vectors(
    positions: NDArray, cell_matrix: NDArray, engine: ModuleType = np, dim_warn: bool = True
) -> NDArray:

    """
    function for computing pairwise difference vectors
    """

    if dim_warn and positions.shape[1] != 3:
        warnings.warn("positions array has shape other than (*, 3)")

    if dim_warn and cell_matrix.shape != (3, 3):
        warnings.warn("cell matrix has shape other than (3, 3)")

    if engine.__name__ == "torch":
        positions = engine.tensor(positions)
        cell_matrix = engine.tensor(cell_matrix)

    # first, invert cell matrix
    inverted_cell_matrix = engine.linalg.inv(cell_matrix)

    # calculate physical difference tensor
    differences = positions[:, None] - positions

    # get fractional differences, changing from distance units to supercell lattice units
    # positions[:, None] - positions is difference tensor
    # difference[i, j] = positions[i] - positions[j]

    fractional_differences = engine.einsum(
        "km,ijm->ijk", inverted_cell_matrix, differences
    )

    # get images
    # invert fractional distances to physical units
    # round fractional differences
    images = engine.einsum("km,ijm->ijk", cell_matrix, np.round(fractional_differences))

    # subtract off the images to get the minimum image differences
    differences = differences - images

    return differences


def get_pairwise_distances(
    positions: NDArray, cell_matrix: NDArray, engine: ModuleType = np, dim_warn: bool = True
) -> NDArray:

    """
    function for computing pairwise distances
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrivateWarning)
        differences = _get_difference_vectors(
            positions,
            cell_matrix,
            engine=engine,
            dim_warn=dim_warn
        )
    minimum_image_distances = engine.linalg.norm(differences, axis=2)

    return np.array(minimum_image_distances)


@private
def _get_difference_vector(
    difference: NDArray, cell_matrix: NDArray, engine: ModuleType = np, dim_warn: bool = True
) -> NDArray:

    """
    function for computing pairwise difference vector
    """

    if dim_warn and difference.shape not in {(1, 3), (3,)}:
        warnings.warn("difference vector has shape other than (1, 3) or (3,)")

    if dim_warn and cell_matrix.shape != (3, 3):
        warnings.warn("cell matrix has shape other than (3, 3)")

    if engine.__name__ != "numpy":
        warnings.warn(
            f"Using {engine.__name__} here is likely a waste. Consider using numpy"
        )

    if engine.__name__ == "torch":
        difference = engine.tensor(difference)
        cell_matrix = engine.tensor(cell_matrix)
    elif "jax" in engine.__name__:
        difference = engine.array(difference)
        cell_matrix = engine.array(cell_matrix)

    inverted_cell_matrix = engine.linalg.inv(cell_matrix)
    fractional_difference = inverted_cell_matrix @ difference
    image = cell_matrix @ engine.round(fractional_difference)
    difference = difference - image

    return difference


def get_pairwise_distance(
    difference: NDArray, cell_matrix: NDArray, engine: ModuleType = np, dim_warn: bool = True
) -> float:

    """
    function for computing pairwise distance
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrivateWarning)
        difference = _get_difference_vector(
            difference,
            cell_matrix,
            engine=engine,
            dim_warn=dim_warn
        )
    distance = engine.linalg.norm(difference)
    if not isinstance(distance, float):
        return float(distance)
    return distance
