# pbc_distance_calculator

![](https://img.shields.io/badge/python-3.9--3.12-blue?logo=python&logoColor=white&labelColor=blue&color=grey)

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

The above script performs the calculation in a vectorized form, computing every pairwise distance at once. To do it serially instead:

```python
from numpy.typing import NDArray
from pbc_distance_calculator import get_pairwise_distance

# arrays of shape (1, 3) or (3, 1)
first_position: NDArray = ...
second_position: NDArray = ...

# array of shape (3, 3)
cell_matrix: NDArray = ...

# minimum image distance
pairwise_distance: float = get_pairwise_distance(
    first_position - second_position,
    cell_matrix
)
```

In both functions, you can also specify different engines to compute the distances. This is especially advantageous for large systems, where you can specify ``jax.numpy`` or ``torch`` as an engine. For example:

```python
import torch
from pbc_distance_calculator import get_pairwise_distances

...

torch.set_default_device("cuda")
pairwise_distances = get_pairwise_distances(
    positions,
    cell_matrix,
    engine=torch
)
```

which will calculate the pairwise distances using the CUDA-backend of PyTorch. Note that the only engine installed by default is ``numpy``, so make sure to separately install ``jax`` or ``torch`` if you want to use these modules.

Note that the cell matrix, is, in general:

$$
\begin{pmatrix} \mathbf{a} & \mathbf{b} & \mathbf{c} \end{pmatrix}
$$

where $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ are the lattice vectors of the supercell. Note that this definition works for any set of lattice parameters! So, no matter how weird your crystal, this package should work. If there are any problems, feel free to [open an issue](https://github.com/MUEXLY/pbc_distance_calculator/issues) 🙂.