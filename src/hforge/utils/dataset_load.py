
from hforge.graph_dataset import graph_from_row
from datasets import load_from_disk, concatenate_datasets
from torch_geometric.loader import DataLoader
from torch import Generator
from torch_geometric.data import Batch
from collections import defaultdict
import random

def prepare_dataset(dataset_path, orbitals, training_split_ratio=0.8, test_split_ratio = 0.5, cutoff=4.0, max_samples=None, load_other_nr_atoms=False, print_finish_message=True):
    import os
    import random
    """
    Prepare dataset for training by returning the train and validation data loaders.

    Args:
        dataset_path: Path to the dataset
        orbitals: Dictionary mapping atomic numbers to number of orbitals
        training_split_ratio: Train/validation split ratio
        test_split_ratio: Validation/Test split ratio
        batch_size: Batch size for dataloaders
        cutoff: Cutoff distance for graph construction
        max_samples: Maximum number of samples to load (for debugging)
        load_other_nr_atoms (bool, optional): If set to True, it will load all the data in the parent folder of the specified data folder in `dataset_path` into a unified dataset. Default is False.
        return_dataloaders (bool, optional): If True, returns the train and validation dataloaders. If False, returns the dataset itself. Default is True

    Returns:
        train_loader, val_loader
    """
    # Load all datasets the the parent folder of the specified folder
    if load_other_nr_atoms:
        # Load all subfolders in the parent directory
        parent_folder_path = os.path.abspath(os.path.join(dataset_path, os.pardir))
        # Iterate through all the files in the parent folder
        dataset_list = []
        for folder in os.listdir(parent_folder_path):
            folder_path = os.path.join(parent_folder_path, folder)
            # Ensure it's a folder
            if os.path.isdir(folder_path):
                dataset_list.append(load_from_disk(folder_path))
        # Concatenate all datasets into one
        dataset = concatenate_datasets(dataset_list)

    # Only load the specified dataset
    else:
        dataset = load_from_disk(dataset_path)

    # Convert dataset to graph form
    graph_dataset = []
    sample_count = 0
    for row in dataset:
        graph = graph_from_row(row, orbitals, cutoff=cutoff)
        graph_dataset.append(graph)
        sample_count += 1

        # Break if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break
    # print("Graph generation done!")

    # Split into train and validation
    # First shuffle the dataset, but always keep the same seed for reproducibility
    random.Random(4).shuffle(graph_dataset)

    # Then split it
    split_idx = int(len(graph_dataset) * training_split_ratio)
    train_dataset = graph_dataset[:split_idx]
    validation_dataset = graph_dataset[split_idx:]

    split_idx = int(len(validation_dataset) * test_split_ratio)
    test_dataset = validation_dataset[:split_idx]
    validation_dataset = validation_dataset[split_idx:]
    if print_finish_message:
        print(f"Created {len(train_dataset)} training samples, {len(validation_dataset)} validation samples and {len(test_dataset)} test samples.")

    return train_dataset, validation_dataset, test_dataset


def prepare_dataloaders(train_dataset, validation_dataset, batch_size=1, seed=4, shuffle_train_dataloader=True):
    # Set a manual seed for reproducibility
    # generator = Generator()
    # generator.manual_seed(seed)

    # Custom collate function for batching graph data
    def custom_collate(batch):
        return Batch.from_data_list(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train_dataloader,
        pin_memory=True,
        collate_fn=custom_collate,
        # generator=generator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate
    )

    print("Training batch example:")
    for batch in train_loader:
        print(batch)
        break

    return train_loader, validation_loader

# def get_stratified_indices(dataset, samples_per_group=3, seed=4):
#     """
#     Return indices of `samples_per_group` samples per unique n_atoms type in the dataset.
#
#     Args:
#         dataset: A list or torch Dataset of PyG Data objects
#         samples_per_group: Number of samples to return for each n_atoms group
#         seed: Random seed for reproducibility
#
#     Returns:
#         A list of selected indices
#     """
#     # Group indices by n_atoms
#     n_atoms_to_indices = defaultdict(list)
#     for idx, data in enumerate(dataset):
#         n_atoms = data.x.size(0)  # assuming x is node feature matrix
#         n_atoms_to_indices[n_atoms].append(idx)
#     # print(n_atoms_to_indices)
#
#     # Set seed for reproducibility
#     random.seed(seed)
#
#     selected_indices = []
#     for n_atoms, indices in n_atoms_to_indices.items():
#         if len(indices) < samples_per_group:
#             print(f"Warning: Not enough samples for n_atoms={n_atoms}, only found {len(indices)}")
#             selected = indices  # take all if fewer than needed
#         else:
#             selected = random.sample(indices, samples_per_group)
#         selected_indices.extend(selected)
#
#     return selected_indices

def get_stratified_datasets(train_dataset,
                            val_dataset,
                            n_train_samples=3,
                            n_validation_samples=3,
                            max_n_atoms=None,
                            seed=4,
                            print_finish_message=True):
    """
    Subsample training and validation datasets to have fixed number of samples per n_atoms value.

    Args:
        train_dataset: List of PyG Data objects (original train set)
        val_dataset: List of PyG Data objects (original val set)
        n_train_samples: Samples per n_atoms group for training
        n_validation_samples: Samples per n_atoms group for validation
        max_n_atoms: If set, skip samples with n_atoms > max_n_atoms
        seed: Random seed
        print_finish_message: Whether to log the result

    Returns:
        (train_subset, train_indices, val_subset, val_indices): Subsampled datasets and original indices
    """
    rng = random.Random(seed)

    def subsample(dataset, samples_per_group):
        grouped = defaultdict(list)
        for idx, data in enumerate(dataset):
            n_atoms = data.x.size(0)
            if max_n_atoms is not None and n_atoms > max_n_atoms:
                continue
            grouped[n_atoms].append((idx, data))

        subset = []
        subset_indices = []
        for n_atoms, items in grouped.items():
            rng.shuffle(items)
            count = min(len(items), samples_per_group)
            if count < samples_per_group:
                print(f"WARNING: Only {count} samples available for n_atoms={n_atoms} (needed {samples_per_group})")
            selected = items[:count]
            subset.extend([d for _, d in selected])
            subset_indices.extend([i for i, _ in selected])
        return subset, subset_indices

    train_subset, train_indices = subsample(train_dataset, n_train_samples)
    val_subset, val_indices = subsample(val_dataset, n_validation_samples)

    if print_finish_message:
        print(f"[Subset Builder] Final: {len(train_subset)} training and {len(val_subset)} validation samples (stratified by n_atoms)")

    return train_subset, train_indices, val_subset, val_indices

