import os
from itertools import cycle

import torch
from hforge.graph_dataset import graph_from_row, graph_from_row_supercell

def circular_iterate(n_types, available_samples_list, initial_samples_list, max_samples):
    """Needed for the below function"""
    # print(f"{n_types=}")
    # print(f"{available_samples_list=}")
    # print(f"{initial_samples_list=}")
    # print(f"{max_samples=}")
    pool = cycle(range(n_types))
    current_samples_list = initial_samples_list
    current_samples = sum(current_samples_list)
    for i in pool:
        # Check if there are available samples
        if available_samples_list[i] > current_samples_list[i]:
            current_samples_list[i] += 1
            current_samples += 1
        else:
            # Don't add samples from this dataset
            continue
        # Check if we reached the desired max_samples
        if current_samples == max_samples:
            break
    return current_samples_list

def get_stratified_dataset(datasets, max_samples, seed=42):
    n_samples_list = get_amount_to_stratify(datasets, max_samples)

    # Some prints:
    n_atoms_list = [ds["nr_atoms"][0] for ds in datasets]
    print("Sample distribution in the dataset w.r.t. the number of atoms:")
    print(f"{n_atoms_list}")
    print(f"{n_samples_list}")

    # Random shuffle and take only current_samples_list of them
    datasets_uniform = [dataset.shuffle(seed=seed).select(range(n_samples_list[i])) for i, dataset in enumerate(datasets)]

    return datasets_uniform

def get_amount_to_stratify(datasets, max_samples):
    # Check if there will be enough samples
    if max_samples > sum(len(dataset) for dataset in datasets):
        raise ValueError("max_samples cannot be larger than the total number of samples")

    # Get how many samples of each type we need
    n_types = len(datasets)
    n_initial_each = max_samples // n_types
    n_remaining = max_samples % n_types

    # Get available samples of each type
    available_samples_list = [len(dataset) for dataset in datasets]

    # Check if there are enough samples of each type
    current_samples_list = [n_initial_each if len(dataset) >= n_initial_each else len(dataset) for dataset in datasets] # List of initial samples per each type
    is_enough = all(n >= n_initial_each for n in current_samples_list) # True if there are enough samples o each type

    # Set the number of samples of each type
    if is_enough:
        # There are enough samples of each type
        current_samples_list = [n_initial_each for _ in range(n_types)]

        # Add the remaining (very few, just 1 cycle)
        if n_remaining != 0:
            n_samples_list = circular_iterate(n_types, available_samples_list, current_samples_list, max_samples)
        else:
            n_samples_list = current_samples_list

    else:
        # Iterate through datasets circulary
        n_samples_list = circular_iterate(n_types, available_samples_list, current_samples_list, max_samples)
    return n_samples_list

def graph_conversion(dataset, orbitals, cutoff=4.0):
    """Convert dataset to graphs"""
    # print("dataset= ", dataset)
    graph_dataset = []
    for row in dataset:
        # print("row= ", row)
        graph = graph_from_row(row, orbitals, cutoff=cutoff)
        graph_dataset.append(graph)
    return graph_dataset

def preprocess_and_save_graphs(datasets, orbitals, parent_output_path, cutoff=4.0):
    # Create directory
    os.makedirs(parent_output_path, exist_ok=True)

    # Save graphs
    for dataset in datasets:
        # Create directory
        n_atoms = dataset["nr_atoms"][0]
        path = os.path.join(parent_output_path, f"{n_atoms}_atoms")
        os.makedirs(path, exist_ok=True)

        # Save all graphs of this type in this directory
        count = 0
        for i, row in enumerate(dataset):
            graph = graph_from_row_supercell(row, orbitals, cutoff=cutoff)
            torch.save(graph, os.path.join(path, f"graph_{n_atoms}atoms_{i:04d}.pt"))
            count += 1

    print(f"Dataset preprocessing completed and saved to {parent_output_path}")


    # os.makedirs(parent_output_path, exist_ok=True)
    #
    # count = 0
    # for i, row in enumerate(dataset):
    #     graph = graph_from_row(row, orbitals, cutoff=cutoff)
    #     torch.save(graph, os.path.join(parent_output_path, f"graph_{i:03d}.pt"))
    #     count += 1