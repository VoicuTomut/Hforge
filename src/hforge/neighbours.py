"""
This is the neighborhood function extracted from:
https://github.com/ACEsuit/mace/blob/main/mace/data/neighborhood.py
"""


from typing import Optional, Tuple
import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,
    true_self_interaction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute neighbor list for atoms within a given cutoff, accounting for periodic boundary conditions (PBC),
    using the neighbour_list function from matscipy.

    Args:
        positions (np.ndarray): Cartesian coordinates of atoms [num_atoms, 3].
        cutoff (float): Cutoff radius for neighbor search.
        pbc (Tuple[bool, bool, bool], optional): Periodic Boundary Conditions flags for x, y, z directions. Defaults to (False, False, False).
        cell (np.ndarray, optional): 3x3 lattice matrix. If None or zero, defaults to identity matrix.
        true_self_interaction (bool, optional): If False, removes self-edges that do not involve periodic images.

    Returns:
        Tuple[
            np.ndarray,  # edge_index: [2, num_edges], atom pair indices (i->j)
            np.ndarray,  # shifts: [num_edges, 3], displacement vectors for neighbors
            np.ndarray,  # unit_shifts: [num_edges, 3], integer periodic image shifts
            np.ndarray   # cell: 3x3 lattice matrix (possibly modified)
        ]
    """
    # Default to no periodic boundary conditions
    if pbc is None:
        pbc = (False, False, False)

    # Default to identity matrix if cell is None or zero
    if cell is None or np.allclose(cell, 0):
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(flag, (bool, np.bool_)) for flag in pbc)
    assert cell.shape == (3, 3)

    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.abs(positions)) + 1

    # Extend non-periodic directions of the cell to avoid missing neighbors at the edges
    for dim, periodic in enumerate(pbc):
        if not periodic:
            cell[dim, :] = max_positions * 5 * cutoff * identity[dim, :]

    # Compute the neighbor list with unit cell shifts
    # print("pbc= ", pbc)
    # print("cell.shape= ", cell.shape)
    # print("cell= ", cell)
    # print("positions.shape= ", positions.shape)
    # print("cutoff= ", cutoff)
    sender, receiver, unit_shifts = neighbour_list(
        "ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )
    # print("\ncutoff= ", cutoff)
    # print("sender= ", sender)
    # print("receiver= ", receiver)
    # print("unit_shifts.shape= ", unit_shifts.shape)
    # print("unit_shifts= ", unit_shifts)


    # Remove self-edges that are not across periodic boundaries
    if not true_self_interaction:
        self_edges = (sender == receiver)
        no_shift = np.all(unit_shifts == 0, axis=1)
        mask = ~(self_edges & no_shift)

        sender = sender[mask]
        receiver = receiver[mask]
        unit_shifts = unit_shifts[mask]

    # print("\ncutoff= ", cutoff)
    # print("sender= ", sender)
    # print("receiver= ", receiver)
    # print("unit_shifts.shape= ", unit_shifts.shape)
    # print("unit_shifts= ", unit_shifts)

    # Pair indices for neighbor edges
    edge_index = np.stack((sender, receiver), axis=0)

    # Convert unit cell shifts to real-space displacement vectors
    shifts = np.dot(unit_shifts, cell)
    # print("shifts= ", shifts)


    return edge_index, shifts, unit_shifts, cell

