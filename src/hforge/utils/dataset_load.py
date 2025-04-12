
from hforge.graph_dataset import graph_from_row
from datasets import load_from_disk
from torch_geometric.loader import DataLoader

def prepare_dataset(dataset_path, orbitals, split_ratio=0.8, batch_size=1, cutoff=4.0, max_samples=None):
    """
    Prepare dataset for training by returning the train and validation data loaders.

    Args:
        dataset_path: Path to the dataset
        orbitals: Dictionary mapping atomic numbers to number of orbitals
        split_ratio: Train/validation split ratio
        batch_size: Batch size for dataloaders
        cutoff: Cutoff distance for graph construction
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        train_loader, val_loader
    """
    # Load dataset
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
    # print("Graph right after conversion from row: ",  graph["edge_index"])

    print("Graph generation done!")

    # Custom collate function to ensure proper ordering of h and s matrices
    def custom_collate(batch):
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(batch)

        # Ensure h and s matrices are properly aligned
        # This depends on your specific data structure, but might look like:
        # Reorganize h_on_sites and s_on_sites if needed
        # Reorganize h_hop and s_hop if needed

        return batch

    # Split into train and validation
    split_idx = int(len(graph_dataset) * split_ratio)
    train_dataset = graph_dataset[:split_idx]
    val_dataset = graph_dataset[split_idx:]

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate
    )

    print(f"Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print(f"Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print("Training batch example:")
    for batch in train_loader:
        print(batch)
        break

    return train_loader, val_loader