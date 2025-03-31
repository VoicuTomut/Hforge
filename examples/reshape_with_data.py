"""
Play a bit with the data and analize it.
"""

import numpy as np
from datasets import load_from_disk
from hforge.neighbours import get_neighborhood
from hforge.graph_dataset import create_graph

def preprocess_edges(edge_list):
    """
    Converts the edge list into a set of frozensets for O(1) lookup.
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

def decompose_matrix(system_mat, orbitals, elements_z, proces_edges):
    """
    Decomposes a system matrix into on-site and hopping matrices based on connectivity.

    This function divides the input system matrix into smaller submatrices,
    which are categorized as either on-site blocks (diagonal blocks) or
    hopping blocks (off-diagonal blocks corresponding to existing edges).

    The connectivity between blocks is determined by the `proces_edges` set.

    :param system_mat: 2D NumPy array representing the full system matrix.
    :param orbitals: Dictionary mapping each atomic number (Z) to the number
                     of orbitals associated with that element.
    :param elements_z: List of atomic numbers (Z) corresponding to each block in the system matrix.
    :param proces_edges: A preprocessed set of edges (frozensets) indicating connectivity between blocks.
    :return: A tuple containing:
             - on_sites: List of 2D NumPy arrays representing diagonal blocks (on-site matrices).
             - hop: List of 2D NumPy arrays representing off-diagonal blocks (hopping matrices)
                    for connected edges.
    """

    i, j = 0, 0
    on_sites = []
    hop = []
    for a, a_z in enumerate(elements_z):
        print(a_z)
        for b, b_z in enumerate(elements_z):
            if a == b:
                # We are ona  diagonal block
                matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                on_sites.append(matrix_block)
            elif a != b:
                # We need to check if th edge exists
                # print("edge:", [a, b])
                edge_exists = find_edge([a, b], proces_edges)
                # print(f"edge:{[a, b]} ,{edge_exists=}")
                if edge_exists:
                    matrix_block = system_mat[i:i + orbitals[a_z], j:j + orbitals[b_z]]
                    hop.append(matrix_block)
            j += orbitals[b_z]
        i += orbitals[a_z]
        j = 0

    print(f"{on_sites=},\n {hop=}")
    print(f"{len(on_sites)=},\n {len(hop)=}")
    return on_sites, hop

def main():
    # Load the dataset
    dataset = load_from_disk("/Users/voicutomut/Documents/GitHub/Hforge/Data/aBN_HSX/nr_atoms_2")
    # features: ['nr_atoms', 'atomic_types_z', 'atomic_positions', 'lattice_nsc', 'lattice_origin',
    #            'lattice_vectors', 'boundary_condition', 'h_matrix', 's_matrix']
    print(dataset)


    # Playing_row
    row_index = 2
    row = dataset[row_index]  # Replace 'train' with the correct split if applicable
    #print("Extracted Row:", row)

    # Now let's pass it tru mace:
    positions=np.array(row["atomic_positions"])
    print("positions.shape:", positions.shape)
    cell=np.array(row["lattice_vectors"])
    print("cell.shape:", cell.shape)
    edge_index, shifts, unit_shifts, cell = get_neighborhood(positions=positions,
    cutoff=2.0,
    pbc=(True,True,True),
    cell = cell,
    true_self_interaction=False)

    # print(f"{edge_index=},\n {shifts=},\n {unit_shifts=},\n {cell=}")

    # Now we finally have al the information required for building a graph
    # One extra detail that we need to se up the nr of atom tipes that our mode will handle and the orbitals:
    # atoms by atomic numbers:
    available_atoms=[1,2,3,4,5,6,7,8]
    # dictionary that maps the atomic number to the nr of orbitals
    orbitals={
        1:13,
        2:13,
        3:13,
        4:13,
        5:13,
        6:13,
        7:13,
        8:13,}
    # For etch edge is time now to extract the describing block save for the onsites for both H and S matrix
    proces_edges=preprocess_edges(edge_index)
    # print("h matrix:", row['h_matrix'])
    hm=np.array(row['h_matrix'])
    # print("h matrix.shape:", hm.shape)
    sm = np.array(row['h_matrix'])
    # print("s matrix.shape:", sm.shape)

    h_on_sites, h_hop = decompose_matrix(system_mat=hm,
                                         orbitals=orbitals,
                                         elements_z=row["atomic_types_z"],
                                         proces_edges=proces_edges)
    s_on_sites, s_hop = decompose_matrix(system_mat=sm,
                                         orbitals=orbitals,
                                         elements_z=row["atomic_types_z"],
                                         proces_edges=proces_edges)

    # Now we relly have evrything let's conver tit to graph specific dataset
    # The grap will have the following atributes:
    # atomic_number_for_nodes, atomic coordinates  , edge_index, h_on_sites, h_hop, s_on_sites, s_hop
    row_graph=create_graph(atomic_numbers=row["atomic_types_z"],
                           atomic_coordinates=positions,
                           edge_index=edge_index,
                           h_on_sites=h_on_sites,
                           h_hop=h_hop,
                           s_on_sites=s_on_sites,
                           s_hop=s_hop)
    print(row_graph)








if __name__ == "__main__":
    main()