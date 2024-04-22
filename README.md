# pbc_distance_calculator

![](https://img.shields.io/badge/python-3.8--3.12-blue?logo=python&logoColor=white&labelColor=blue&color=grey)

This Python package computes pairwise distances in a simulation box accounting for [periodic boundary conditions](https://en.wikipedia.org/wiki/Periodic_boundary_conditions).

The only inputs are the positions of each particle and the simulation supercell matrix.

To install:

```bash
pip install pbc_distance_calculator
```

Example usage:

```python
from numpy.typing import NDArray
from pbc_distance_calculator import get_pairwise_distances

# array of shape (N, 3) where N is the number of particles
positions: NDArray = ...

# array of shape (3, 3)
cell_matrix: NDArray = ...

# array of shape (N, N)
# element (i, j) is minimum image distance between i and j
pairwise_distances: NDArray = get_pairwise_distances(positions, cell_matrix)
```

The cell matrix, is, in general:

$$
\begin{pmatrix} \mathbf{a} & \mathbf{b} & \mathbf{c} \end{pmatrix}
$$

where $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ are the lattice vectors of the supercell. Note that this definition works for any set of lattice parameters! So, no matter how weird your crystal, this package should work. If there are any issues, feel free to open an issue 🙂.