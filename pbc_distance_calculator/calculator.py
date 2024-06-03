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


class WasteWarning(Warning):

    """
    warning for performing wasteful call to expensive engine
    """


class UnfinishedWarning(Warning):

    """
    warning for function that isn't finished yet
    """


def unfinished(func: Callable) -> Callable:

    """
    decorator for warning the user that a function's implementation isn't finished
    """

    def wrapper(*args, **kwargs) -> Callable:
        warnings.warn(
            f"{func.__name__}'s implementation is not yet finished"
        )
        return func(*args, **kwargs)
    
    return wrapper


def private(func: Callable) -> Callable:

    """
    decorator for marking a function as private
    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is private and is subject to backwards-incompatible changes",
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
        if not isinstance(positions, engine.Tensor):
            positions = engine.tensor(positions)
        if not isinstance(cell_matrix, engine.Tensor):
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

    if len(positions) == 1:
        warnings.warn("You called get_pairwise_distances for a single position. "
                      "Did you mean to call get_pairwise_distance instead?")

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

    if engine.__name__ == "torch":
        if not isinstance(difference, engine.Tensor):
            difference = engine.tensor(difference)
        if not isinstance(cell_matrix, engine.Tensor):
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

    if engine.__name__ != "numpy":
        warnings.warn(
            f"Using {engine.__name__} here is likely a waste. Consider using numpy",
            WasteWarning
        )

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


@unfinished
def get_pairwise_distance_cascade(
    positions: NDArray, cell_matrix: NDArray, engine: ModuleType, cutoff: float = np.inf
) -> NDArray:
    
    if engine.__name__ != "torch":
        raise ValueError("cascade algorithm only works with PyTorch backend")
    
    inverse_cell_matrix = engine.linalg.inv(cell_matrix)

    sampled = set()
    num_sites = len(positions)

    seed = 0
    flower = {seed}

    distance_tensor = engine.zeros((num_sites, num_sites))

    while len(sampled) < num_sites:

        differences = positions.unsqueeze(0) - positions[list(flower)].unsqueeze(1)
        fractional_differences = engine.einsum(
            "km,ijm->ijk", inverse_cell_matrix, differences
        )
        images = engine.einsum("km,ijm->ijk", cell_matrix, engine.round(fractional_differences))
        distances = engine.linalg.norm(differences - images, dim=2)
        neighbors = ((0 < distances) & (distances < cutoff)).int()
        distances[engine.where(neighbors == 0)] = 0.0

        distance_tensor[list(flower), :] = distances[..., :]

        neighbor_indices = set(map(int, engine.nonzero(neighbors)[:, 1]))
        sampled.update(flower)

        flower = neighbor_indices - sampled

    return distance_tensor