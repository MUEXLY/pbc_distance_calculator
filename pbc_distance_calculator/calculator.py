"""
module for getting pairwise distances
"""

from types import ModuleType
import warnings


import numpy as np
from numpy.typing import NDArray


def get_pairwise_distances(
    positions: NDArray, cell_matrix: NDArray, engine: ModuleType = np
) -> NDArray:

    """
    function for computing pairwise distances
    """

    assert positions.shape[1] == 3
    assert cell_matrix.shape == (3, 3)

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
    minimum_image_distances = engine.linalg.norm(differences, axis=2)

    return np.array(minimum_image_distances)


def get_pairwise_distance(
    difference: NDArray, cell_matrix: NDArray, engine: ModuleType = np
) -> float:

    """
    function for computing pairwise distance
    """

    assert difference.shape in {(1, 3), (3,)}
    assert cell_matrix.shape == (3, 3)

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

    return float(engine.linalg.norm(difference))
