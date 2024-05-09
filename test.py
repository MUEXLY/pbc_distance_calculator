"""
script for testing the pairwise distance calculation
"""


from dataclasses import dataclass, field
from copy import deepcopy
from importlib import import_module
from types import ModuleType
from itertools import product
from contextlib import nullcontext

import pytest
from numpy.typing import NDArray

from pbc_distance_calculator import get_pairwise_distances, get_pairwise_distance, is_valid, utils


np = import_module("numpy")
torch = import_module("torch")
jax = import_module("jax.numpy")
random = import_module("random")


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

    def get_pairwise_distances(self, engine: ModuleType) -> NDArray:

        """
        function for getting the sites and cell matrix for a supercell
        """

        return get_pairwise_distances(
            self.site_positions, self.cell_matrix, engine=engine
        )


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


TEST_PARAMETERS = list(product((FCC, BCC, SC), (np, torch, jax)))


# perform tests
@pytest.mark.parametrize("container,engine", TEST_PARAMETERS)
def test_coordination_numbers(
    container: CoordinationNumberTestingContainer, engine: ModuleType
):

    """
    function for performing coordination number test on a container
    """

    distances = container.get_pairwise_distances(engine=engine)
    assert isinstance(distances, np.ndarray)

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


TEST_PARAMETERS = list(product((FCC_SMALL, BCC_SMALL, SC_SMALL), (np, torch, jax)))


@pytest.mark.parametrize("container,engine", TEST_PARAMETERS)
def test_serial_and_vectorized_equal(
    container: CoordinationNumberTestingContainer, engine: ModuleType
):

    """
    function for performing test to assure that serial and vectorized calculation are the same
    """

    vectorized_distances = container.get_pairwise_distances(engine=engine)

    if engine.__name__ == "numpy":
        def ctx():
            return nullcontext()
    else:
        def ctx():
            return pytest.warns(UserWarning)

    for i, first_site in enumerate(container.site_positions):
        for j, second_site in enumerate(container.site_positions):

            with ctx():
                distance = get_pairwise_distance(
                    first_site - second_site, container.cell_matrix, engine=engine
                )
            assert np.isclose(vectorized_distances[i, j], distance)


ENGINE_VALIDITY = [
    (np, True),
    (torch, True),
    (jax, True),
    (random, False)
]


@pytest.mark.parametrize("engine,expected_result", ENGINE_VALIDITY)
def test_engine_validity(engine: ModuleType, expected_result: bool):

    """
    test for engine validity
    checks if necessary functions present in module  API
    """

    assert is_valid(engine) == expected_result


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("engine", (np, torch, jax))
def test_util_methods_callable(engine: ModuleType):

    """
    test to make sure util functions are accessible and callable
    """

    cell_matrix = engine.eye(3)
    if engine.__name__ == "torch":
        positions = engine.tensor([
            [0.0, 0.0, 0.0],
            [0.75, 0.75, 0.75]
        ])
    else:
        positions = engine.array([
            [0.00, 0.00, 0.00],
            [0.75, 0.75, 0.75]
        ])

    # pylint: disable=protected-access
    with pytest.warns(utils.PrivateWarning):
        diff_vector = utils._get_difference_vector(
            positions[0, :] - positions[1, :],
            cell_matrix,
            engine=engine
        )
        diff_vectors = utils._get_difference_vectors(positions, cell_matrix, engine=engine)

    assert tuple(diff_vector) == (0.25, 0.25, 0.25)
    assert engine.all(diff_vectors[0, 1, :] == diff_vector)
