import numpy as np
import os
import torch
from torch_geometric.data import Data, Dataset
from hforge.data_preproces import decompose_matrix,preprocess_edges
from hforge.neighbours import get_neighborhood


def create_graph(atomic_numbers, atomic_coordinates, edge_index,shifts, h_on_sites, h_hop, s_on_sites, s_hop):
    """
    Create a graph object with the given properties.
    Args:
        atomic_numbers (list or array): Atomic numbers for the nodes.
        atomic_coordinates (list or array): 3D coordinates of each node.
        edge_index (list or array): Edge connections between nodes.
        h_on_sites (list or array): Hamiltonian on-site terms for nodes.
        h_hop (list or array): Hamiltonian hopping terms for edges.
        s_on_sites (list or array): Overlap matrix on-site terms for nodes.
        s_hop (list or array): Overlap matrix hopping terms for edges.
    Returns:
        Data: A PyTorch Geometric Data object.
    """
    # Convert inputs to tensors
    x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)  # Node atomic numbers as features
    pos = torch.tensor(atomic_coordinates, dtype=torch.float)  # Node positions
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # Edge connections
    h_on_sites = torch.tensor(np.array(h_on_sites), dtype=torch.float)  # Node on-site Hamiltonian
    s_on_sites = torch.tensor(np.array(s_on_sites), dtype=torch.float)  # Node on-site overlap
    h_hop = torch.tensor(np.array(h_hop), dtype=torch.float)  # Edge hopping Hamiltonian
    s_hop = torch.tensor(np.array(s_hop), dtype=torch.float)  # Edge hopping overlap
    shifts = torch.tensor(np.array(shifts), dtype=torch.float)

    # Create a graph data object
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        shifts=shifts,
        h_on_sites=h_on_sites,
        s_on_sites=s_on_sites,
        h_hop=h_hop,
        s_hop=s_hop,
    )
    return data


def save_dataset(graphs, save_dir):
    """
    Save a list of graphs to a directory.
    Args:
        graphs (list): List of PyTorch Geometric Data objects.
        save_dir (str): Directory to save the dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, graph in enumerate(graphs):
        torch.save(graph, os.path.join(save_dir, f"graph_{i}.pt"))


def load_dataset(load_dir):
    """
    Load a dataset from a directory.
    Args:
        load_dir (str): Directory containing the dataset.
    Returns:
        list: List of PyTorch Geometric Data objects.
    """
    graph_files = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if f.endswith(".pt")]
    graphs = [torch.load(f) for f in graph_files]
    return graphs


class AtomicGraphDataset(Dataset):
    def __init__(self, data_list=None, root=None, transform=None, pre_transform=None):
        self.data_list = data_list  # Save the list of graphs
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        # Handle the reloading case when data_list is None
        if self.data_list is None:
            # Count processed files in the directory
            return sorted([f for f in os.listdir(self.processed_dir) if f.startswith('data_')])
        return [f'data_{i}.pt' for i in range(len(self.data_list))]

    def process(self):
        # Process only if data_list is provided
        if self.data_list is None:
            raise ValueError("Cannot process dataset without data_list.")
        for i, data in enumerate(self.data_list):
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        if self.data_list is not None:
            return len(self.data_list)
        # Reloading case: count files in the processed directory
        return len(self.processed_file_names)

    def get(self, idx):
        data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        return torch.load(data_path)


def graph_from_row(row,orbitals, cutoff=3.0):
    # Now let's pass it tru mace:

    # print(row.keys())
    # print("row[nr_atoms]: ", row["nr_atoms"])
    # Get the atomic positions
    positions = np.array(row["atomic_positions"])
    # print("positions.shape:", positions.shape) # [nr_atoms, 3]; each atom has 3D position

    # Get the cell (described by the lattice vectors)
    cell = np.array(row["lattice_vectors"])
    # print("cell.shape:", cell.shape) # [3,3]; 3 lattice vectors that are 3D.

    edge_index, shifts, unit_shifts, cell = get_neighborhood(positions=positions,
                                                             cutoff=cutoff,
                                                             pbc=(True, True, True),
                                                             cell=cell,
                                                             true_self_interaction=False)

    # print(f"{edge_index=},\n {shifts=},\n {unit_shifts=},\n {cell=}")

    # Now we finally have al the information required for building a graph
    # One extra detail that we need to se up the nr of atom tipes that our mode will handle and the orbitals:

    # For etch edge is time now to extract the describing block save for the onsites for both H and S matrix
    proces_edges = preprocess_edges(edge_index)
    # print("h matrix:", row['h_matrix'])
    hm = np.array(row['h_matrix'])
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
    row_graph = create_graph(atomic_numbers=row["atomic_types_z"],
                             atomic_coordinates=positions,
                             edge_index=edge_index,
                             shifts=shifts,
                             h_on_sites=h_on_sites,
                             h_hop=h_hop,
                             s_on_sites=s_on_sites,
                             s_hop=s_hop)

    # print("row_graph[h_on_sites].shape= ", row_graph["h_on_sites"].shape)
    # print("row_graph[h_hop].shape= ", row_graph["h_hop"].shape)

    return row_graph