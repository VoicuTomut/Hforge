"""

"""

import numpy as np
import torch
from datasets import load_from_disk
from torch_geometric.loader import DataLoader



from hforge.graph_dataset import create_graph, AtomicGraphDataset, graph_from_row







def main():
    # Load the dataset
    dataset = load_from_disk(r"./Data/aBN_HSX/nr_atoms_3")
    # features: ['nr_atoms', 'atomic_types_z', 'atomic_positions', 'lattice_nsc', 'lattice_origin',
    #            'lattice_vectors', 'boundary_condition', 'h_matrix', 's_matrix']
    print(dataset)
    # dictionary that maps the atomic number to the nr of orbitals
    orbitals = {
        1: 13,
        2: 13,
        3: 13,
        4: 13,
        5: 13,
        6: 13,
        7: 13,
        8: 13, }


    # Playing_row
    graphs=[]
    for row in dataset:
        graph=graph_from_row(row,orbitals)
        graphs.append(graph)
    print("Graphs:",graphs)

    # Dataset:
    deprecated_conversion_to_graph.py

    # Load the dataset from disk
    print("load data:")
    loaded_dataset = AtomicGraphDataset(data_list=None, root=dataset_place)
    print("Loaded dataset:", loaded_dataset)

    # Iterate tru Batches:
    # Assume `loaded_dataset` is your AtomicGraphDataset instance
    batch_size = 2  # Number of graphs per batch
    data_loader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)


    for i, batch in enumerate(data_loader):
        print("batch:", i)
        print(f"{batch}")  # This is a Batch object
        print(f"{batch.x=}")  # Node features of the batch
        print(f"{batch.edge_index=}")  # Edge indices of the batch
        print(f"{batch.batch=}")  # Batch vector indicating graph membership
        print("----")
        break

if __name__ == "__main__":
    main()