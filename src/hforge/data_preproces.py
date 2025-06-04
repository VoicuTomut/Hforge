"""
Codes to preproces data.
"""

import base64
import io

from scipy import sparse
import torch


def preprocess_edges(edge_list):
    """
    Converts the edge list into a set of frozensets for O(1) lookup. Unique edges are represented as frozensets to ensure that the order of nodes does not matter.
    """
    return set(frozenset((edge[0], edge[1])) for edge in zip(edge_list[0], edge_list[1]))

def find_edge(edge, edge_set):
    """
    Checks if an edge exists in the edge set.
    :param edge: List or tuple [a, b] representing the edge to check.
    :param edge_set: Preprocessed set of edges.
    :return: True if the edge exists, False otherwise.
    """
    return frozenset(edge) in edge_set

def decompose_matrix(system_mat, orbitals, elements_z, processed_edges):
    """
    Decomposes a system matrix into on-site and hopping matrices based on connectivity.

    This function divides the input system matrix into smaller submatrices,
    which are categorized as either on-site blocks (diagonal blocks) or
    hopping blocks (off-diagonal blocks corresponding to existing edges).

    The connectivity between blocks is determined by the `processed_edges` set.

    :param system_mat: 2D NumPy array representing the full system matrix.
    :param orbitals: Dictionary mapping each atomic number (Z) to the number
                     of orbitals associated with that element.
    :param elements_z: List of atomic numbers (Z) corresponding to each block in the system matrix.
    :param processed_edges: A preprocessed set of edges (frozensets) indicating connectivity between blocks.
    :return: A tuple containing:
             - on_sites: List of 2D NumPy arrays representing diagonal blocks (on-site matrices).
             - hop: List of 2D NumPy arrays representing off-diagonal blocks (hopping matrices)
                    for connected edges.
    """

    i, j = 0, 0
    on_sites = []
    hop = []
    for a, a_z in enumerate(elements_z):
        # print(a_z)
        for b, b_z in enumerate(elements_z):
            if a == b:
                # We are ona  diagonal block
                matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                on_sites.append(matrix_block)
            elif a != b:
                # We need to check if th edge exists
                # print("edge:", [a, b])
                edge_exists = find_edge([a, b], processed_edges)
                # print(f"edge:{[a, b]} ,{edge_exists=}")
                if edge_exists:
                    matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                    hop.append(matrix_block)
            j += orbitals[b_z]
        i += orbitals[a_z]
        j = 0

    # print(f"{on_sites=},\n {hop=}")
    # print(f"{len(on_sites)=},\n {len(hop)=}")
    return on_sites, hop

def decompose_matrix_supercell(system_mat, orbitals, elements_z, processed_edges):
    """
    Decomposes a system matrix (sparse) into on-site and hopping matrices for a supercell.

    This function divides the input sparse system matrix into smaller submatrices,
    which are categorized as either on-site blocks or
    hopping blocks (corresponding to existing edges).

    The connectivity between blocks is determined by the `processed_edges` set.

    :param system_mat: 2D NumPy array representing the full system matrix.
    :param orbitals: Dictionary mapping each atomic number (Z) to the number
                     of orbitals associated with that element.
    :param elements_z: List of atomic numbers (Z) corresponding to each block in the system matrix.
    :param processed_edges: A preprocessed set of edges (frozensets) indicating connectivity between blocks.
    :return: A tuple containing:
             - on_sites: List of 2D NumPy arrays representing diagonal blocks (on-site matrices).
             - hop: List of 2D NumPy arrays representing off-diagonal blocks (hopping matrices)
                    for connected edges.
    """

    # === Convert the csr matrix to a dense matrix ===
    if sparse.issparse(system_mat):
        system_mat = system_mat.todense()
    else:
        raise ValueError("Matrix is not in sparse format")

    on_sites = []
    hoppings = []
    i, j = 0, 0
    # / print("system_mat.shape[1] // sum(orbitals.values()) = ", system_mat.shape[1] // sum(orbitals.values()))
    for a, a_z in enumerate(elements_z):

        for cell_idx in range(system_mat.shape[1] // sum(orbitals.values())): #Iterate through unit cells
            for b, b_z in enumerate(elements_z): # In each unit cell, there are n_atoms.

                if a == b and cell_idx==0:
                    # Onsite. Extract the orbitals matrix block.
                    matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                    on_sites.append(matrix_block)
                else:
                    # Hopping. Extract the orbitals matrix block.
                    # Check first if edge exists. 
                    # // TODO: If not, we should create it and make it predict 0. However, at this point in the code all necessary edges should have been correctly created. If there is no edge when needed, it means tht the cutoff radius is too low.
                    edge_exists = find_edge([a, b], processed_edges)
                    if edge_exists:
                        matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                        hoppings.append(matrix_block)
                    else:
                        # There should be enough edges during the training.
                        raise ValueError(f"Edge {a, b} does not exist in the preprocessed edges. Check the cutoff radius or the edge preprocessing step.")
                    
                j += orbitals[b_z] # Move to the next block in the row
            j = 0 # Reset j to 0 for the next cell
        i += orbitals[a_z] # Move to the next block in the column

    # Once we have extracted all th blocks, we can convert them to sparse format (COO).
    # on_sites = [m.to_sparse() for m in on_sites]  # Convert each on-site matrix to sparse format
    # hoppings = [m.to_sparse() for m in hoppings]  # Convert each hopping matrix to sparse format

    return on_sites, hoppings

def to_device(data, device):
    """
    Recursively moves all tensors in a data structure to the specified device.
    Args:
        data: The data structure (e.g., tensor, list, dict) to move to the device.
        device: The target device (e.g., 'cuda' or 'cpu').
    Returns:
        The data structure with all tensors moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data  # Non-tensor data remains unchanged
    
def deserialize_csr_matrix(b64_str):
    """Deserialize a base64-encoded string to a scipy.sparse.csr_matrix."""
    return sparse.load_npz(io.BytesIO(base64.b64decode(b64_str)))