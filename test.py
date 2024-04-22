"""
script for testing the pairwise distance calculation
"""


from dataclasses import dataclass, field
from copy import deepcopy

import pytest
import numpy as np
from numpy.typing import NDArray

from pbc_distance_calculator import get_pairwise_distances, get_pairwise_distance


def get_default_unit_cell() -> NDArray:

    """
    default unit cell with lattice parameters 1 and angles 90 degrees
    """

    return np.eye(3)


@dataclass
class CoordinationNumberTestingContainer:

    """
    container defining test
    """

    atomic_basis: NDArray
    dimensions: NDArray
    cutoffs: NDArray
    target_coordination_numbers: NDArray
    unit_cell: NDArray = field(default_factory=get_default_unit_cell)

    @property
    def cell_matrix(self) -> NDArray:

        """
        cell matrix property
        """

        return np.einsum("j,ij->ij", self.dimensions, self.unit_cell)

    @property
    def site_positions(self) -> NDArray:

        """
        site positions property, compute from atomic basis and dimensions
        """

        site_labels = cartesian_product(
            *[np.arange(int(dim)) for dim in self.dimensions],
            np.arange(len(self.atomic_basis))
        )

        num_sites = np.prod(self.dimensions) * len(self.atomic_basis)
        site_positions = np.zeros((int(num_sites), 3))

        for site in range(int(num_sites)):

            # get site label, tuple (IJKL)
            site_label = site_labels[site]

            # get corresponding unit cell position from unit cell matrix
            unit_cell_position = np.einsum("i,ji->j", site_label[:3], self.unit_cell)

            # get the position of the site relative to unit cell for label L
            atomic_site_lattice_units = self.atomic_basis[site_label[3]]

            # convert from lattice units to physical units
            offset = np.einsum("i,ji->j", atomic_site_lattice_units, self.unit_cell)

            # store site position
            site_positions[site] = unit_cell_position + offset

        return site_positions

    def get_pairwise_distances(self) -> NDArray:

        """
        function for getting the sites and cell matrix for a supercell
        """

        return get_pairwise_distances(self.site_positions, self.cell_matrix)


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

FCC_SMALL = deepcopy(FCC)
FCC_SMALL.dimensions = np.array([2, 2, 2])

BCC = CoordinationNumberTestingContainer(
    atomic_basis=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    dimensions=np.array([10, 5, 5]),
    cutoffs=np.array([0.5 * np.sqrt(3.0), 1.0]) * (1.0 + 1.0e-3),
    target_coordination_numbers=np.array([8, 6]),
)

BCC_SMALL = deepcopy(BCC)
BCC_SMALL.dimensions = np.array([4, 2, 2])

SC = CoordinationNumberTestingContainer(
    atomic_basis=np.array([[0.0, 0.0, 0.0]]),
    dimensions=np.array([10, 10, 5]),
    cutoffs=np.array([1.0, np.sqrt(2.0)]) * (1.0 + 1.0e-3),
    target_coordination_numbers=np.array([6, 12]),
)

SC_SMALL = deepcopy(SC)
SC_SMALL.dimensions = np.array([4, 4, 2])


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


@pytest.mark.parametrize("container", (FCC_SMALL, BCC_SMALL, SC_SMALL))
def test_serial_and_vectorized_equal(container: CoordinationNumberTestingContainer):

    """
    function for performing test to assure that serial and vectorized calculation are the same
    """

    vectorized_distances = container.get_pairwise_distances()

    for i, first_site in enumerate(container.site_positions):
        for j, second_site in enumerate(container.site_positions):

            distance = get_pairwise_distance(
                first_site - second_site, container.cell_matrix
            )
            assert np.isclose(vectorized_distances[i, j], distance)
