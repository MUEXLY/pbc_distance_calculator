"""
module for getting pairwise distances
"""


import numpy as np
from numpy.typing import NDArray


def get_pairwise_distances(positions: NDArray, cell_matrix: NDArray) -> NDArray:

    """
    function for computing pairwise distances
    """

    assert positions.shape[1] == 3
    assert cell_matrix.shape == (3, 3)

    # first, invert cell matrix
    inverted_cell_matrix = np.linalg.inv(cell_matrix)

    # calculate physical difference tensor
    differences = positions[:, None] - positions

    # get fractional differences, changing from distance units to supercell lattice units
    # positions[:, None] - positions is difference tensor
    # difference[i, j] = positions[i] - positions[j]

    fractional_differences = np.einsum("km,ijm->ijk", inverted_cell_matrix, differences)

    # get images
    # invert fractional distances to physical units
    # round fractional differences
    images = np.einsum("km,ijm->ijk", cell_matrix, np.round(fractional_differences))

    # subtract off the images to get the minimum image differences
    differences = differences - images
    minimum_image_distances = np.linalg.norm(differences, axis=2)

    return minimum_image_distances


def get_pairwise_distance(difference: NDArray, cell_matrix: NDArray) -> float:

    """
    function for computing pairwise distance
    """

    assert difference.shape in {(1, 3), (3, )}
    assert cell_matrix.shape == (3, 3)

    inverted_cell_matrix = np.linalg.inv(cell_matrix)
    fractional_difference = inverted_cell_matrix @ difference
    image = cell_matrix @ np.round(fractional_difference)
    difference = difference - image

    return float(np.linalg.norm(difference))
