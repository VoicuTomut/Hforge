"""Useful functions to load a .pt model"""

# Standard library imports
import torch
import os
import yaml

# Local application imports
from hforge.data_management.dataset_load import prepare_dataloaders, prepare_dataset
from hforge.model_shell import ModelShell
from hforge.utils.importing_facilities import load_config


#! DEPRECATED:
def load_model(model, optimizer=None, path="best_model.pt", device='cpu'):
    """
    Load the best model checkpoint from the saved file

    Args:
        model: Model instance to load the weights into
        optimizer: Optional optimizer to load state (for continued training)
        path: Path to the saved model checkpoint
        device: Device to load the model to ('cpu' or 'cuda')

    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (if provided)
        epoch: The epoch at which the model was saved
        history: Training history dictionary
    """
    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return model, optimizer, 0, {}

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    epoch = checkpoint['epoch']
    history = checkpoint.get('history', {})

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from {path} (saved at epoch {epoch + 1})")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model, optimizer, epoch, history

def load_model_from_directory(directory: str, model_filename: str, weights_only=True, device='cpu'):
    # === Lazy import to avoid circular import error ===
    from hforge.utils import get_object_from_module
    # === Load configuration ===
    config = load_config(directory + "/training_config.yaml")

    # === Model Configuration ===
    model_config = config["model"]
    model_config["atomic_descriptors"]["interaction_cls_first"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls_first"])
    model_config["atomic_descriptors"]["interaction_cls"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls"])
    model = ModelShell(model_config, device=device).to(device)

    # === Load the model ===
    model_filename = model_filename
    model_path = os.path.abspath(directory + "/" + model_filename)
    checkpoint = torch.load(model_path, weights_only=weights_only, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # === Load other interesting parameters
    history = checkpoint.get('history', {})

    return model, history, config

def load_model_and_dataset_from_directory(directory: str, model_filename: str, weights_only=True, device='cpu', return_datasets=False, return_dataloaders=False):
    """
    Load a .pt model file located inside the specified directory.
    As the .yaml configuration file specifies the dataset configuration, return also the corresponding datasets/dataloaders.

    Args:
        directory:
        model_filename:
        weights_only:
        device:

    Returns:

    """
    # === Load the model ===
    model, history, config = load_model_from_directory(directory, model_filename, weights_only=weights_only, device=device)

    # ! DEPRECATED:
    # === Load the dataset ===
    # Loading the dataset is expensive, so first we check if we need it.
    if return_datasets or return_dataloaders:
        dataset_config = config["dataset"]
        train_dataset, validation_dataset, _ = prepare_dataset(
            dataset_path=dataset_config["path"],
            orbitals=config["orbitals"],
            training_split_ratio=dataset_config["split_ratio"],
            cutoff=dataset_config["cutoff"],
            max_samples=dataset_config["max_samples"],
            load_other_nr_atoms=dataset_config["load_other_nr_atoms"],
        )

        # Return the datasets or the dataloaders depending on what user wants.
        if return_datasets:
            return model, history, train_dataset, validation_dataset, config

        if return_dataloaders:
            train_dataloader, validation_dataloader = prepare_dataloaders(train_dataset, validation_dataset, batch_size=dataset_config["batch_size"])
            return model, history, train_dataloader, validation_dataloader, config

    # === Return only the model and the history if no dataset loading needed
    return model, history, config
