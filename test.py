"""
script for testing the pairwise distance calculation
"""


from dataclasses import dataclass

import pytest
import numpy as np
from numpy.typing import NDArray

from pbc_distance_calculator import get_pairwise_distances


@dataclass
class CoordinationNumberTestingContainer:

    """
    container defining test
    """

    atomic_basis: NDArray
    dimensions: NDArray
    cutoffs: NDArray
    target_coordination_numbers: NDArray

    def get_pairwise_distances(self) -> NDArray:

        """
        function for getting the sites and cell matrix for a supercell
        """

        site_labels = cartesian_product(
            *[np.arange(int(dim)) for dim in self.dimensions],
            np.arange(len(self.atomic_basis))
        )

        unit_cell = np.eye(3)

        num_sites = np.prod(self.dimensions) * len(self.atomic_basis)
        site_positions = np.zeros((int(num_sites), 3))

        for site in range(int(num_sites)):

            # get site label, tuple (IJKL)
            site_label = site_labels[site]

            # get corresponding unit cell position from unit cell matrix
            unit_cell_position = np.einsum("i,ji->j", site_label[:3], unit_cell)

            # get the position of the site relative to unit cell for label L
            atomic_site_lattice_units = self.atomic_basis[site_label[3]]

            # convert from lattice units to physical units
            offset = np.einsum("i,ji->j", atomic_site_lattice_units, unit_cell)

            # store site position
            site_positions[site] = unit_cell_position + offset

        # create the supercell matrix from unit cell matrix
        cell_matrix = np.einsum("j,ij->ij", self.dimensions, unit_cell)

        distances = get_pairwise_distances(site_positions, cell_matrix)

        return distances


def cartesian_product(*arrays):

    """
    cartesian product of arrays
    stolen from https://stackoverflow.com/a/11146645
    """

    length = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [length], dtype=dtype)
    for i, element in enumerate(np.ix_(*arrays)):
        arr[..., i] = element
    return arr.reshape(-1, length)


# define testing containers for different lattices
FCC = CoordinationNumberTestingContainer(
    atomic_basis=np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]
    ),
    dimensions=np.array([5, 5, 5]),
    cutoffs=np.array([0.5 * np.sqrt(2.0), 1.0]) * (1.0 + 1.0e-3),
    target_coordination_numbers=np.array([12, 6]),
)

BCC = CoordinationNumberTestingContainer(
    atomic_basis=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    dimensions=np.array([10, 5, 5]),
    cutoffs=np.array([0.5 * np.sqrt(3.0), 1.0]) * (1.0 + 1.0e-3),
    target_coordination_numbers=np.array([8, 6]),
)

SC = CoordinationNumberTestingContainer(
    atomic_basis=np.array([[0.0, 0.0, 0.0]]),
    dimensions=np.array([10, 10, 5]),
    cutoffs=np.array([1.0, np.sqrt(2.0)]) * (1.0 + 1.0e-3),
    target_coordination_numbers=np.array([6, 12]),
)


# perform tests
@pytest.mark.parametrize("container", (FCC, BCC, SC))
def test_coordination_numbers(container: CoordinationNumberTestingContainer):

    """
    function for performing coordination number test on a container
    """

    distances = container.get_pairwise_distances()

    for i, coordination_number in enumerate(container.target_coordination_numbers):

        if i == 0:
            lower_bound = 0.0
        else:
            lower_bound = container.cutoffs[i - 1]

        within_shell = np.logical_and(
            lower_bound < distances, distances < container.cutoffs[i]
        )
        coordination_numbers = np.sum(within_shell, axis=0)

        assert np.isclose(0, np.std(coordination_numbers))
        assert np.isclose(coordination_number, np.mean(coordination_numbers))
