import time

ti = time.time()
from hforge.model_shell import ModelShell
tf = time.time()
print(f"Time taken to import ModelShell: {tf - ti:.1f} seconds")

import yaml
from hforge.utils import load_model
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
import os
from hforge.utils import create_directory

INTERACTION_BLOKS={"RealAgnosticResidualInteractionBlock":RealAgnosticResidualInteractionBlock}

# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss. 

def main():
    # === Device setup ===
    device = "cpu"

    # === Load configuration ===
    directory = r"C:\Users\angel\OneDrive - Universitat de Barcelona\1A. MASTER I\TFM\example_results\new_example_results_5_model_comparison"
    plots_directory = os.path.join(directory, "plots")
    create_directory(plots_directory)

    for path, folders, files in os.walk(directory):
        for folder in folders:
            folder_path = os.path.join(directory, folder)

            #=== Plot ===
            with open(folder_path+"/training_config.yaml", "r") as f:
                config = yaml.safe_load(f)

            # === Model Configuration ===
            model_config = config["model"]
            model_config["atomic_descriptors"]["interaction_cls_first"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls_first"]]
            model_config["atomic_descriptors"]["interaction_cls"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls"]]

            model = ModelShell(model_config).to(device)

            # Load the model
            _, _, _, history = load_model(model, path=folder_path+"/train_best_model.pt", device=device)

            plot_last_epochs(history, folder_path, start_from_last_epochs=(100,'Last epochs'), plot_validation=True, save_in=(plots_directory, folder))
            plot_last_epochs(history, folder_path, start_from=5, save_in=(plots_directory, folder))


def plot_last_epochs(history, folder, start_from=0, start_from_last_epochs=(0,''), plot_validation=True, save_in=None):
    import matplotlib.pyplot as plt
    """Create final detailed plots from history of training/validation losses"""

    # Check correctness of inputs
    if start_from != 0 and start_from_last_epochs[0] != 0:
        raise ValueError("start_from and start_from_last_epochs cannot be both different from 0.")

    # Set last epoch
    last_epoch = len(history["train_loss"])

    # Plot only the last epochs if specified
    if start_from_last_epochs[0] != 0:
        start_from = last_epoch - start_from_last_epochs[0]

    epochs = range(len(history["train_loss"]))[start_from:]
    if len(epochs) != len(history["val_loss"][start_from:]):
        raise ValueError("The length of train_loss and val_loss do not coincide.")
    plt.figure(figsize=(12, 8))

    # Main loss plots
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history['learning_rate'][start_from:], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history['train_loss'][start_from:], 'b-', label='Training Loss')
    if plot_validation:
        plt.plot(epochs, history['val_loss'][start_from:], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Component losses
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history['train_edge_loss'][start_from:], 'b-', label='Train Edge Loss')
    if plot_validation:
        plt.plot(epochs, history['val_edge_loss'][start_from:], 'r-', label='Val Edge Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Edge Loss')
    plt.title('Edge Matrix Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(epochs, history['train_node_loss'][start_from:], 'b-', label='Train Node Loss')
    if plot_validation:
        plt.plot(epochs, history['val_node_loss'][start_from:], 'r-', label='Val Node Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Node Loss')
    plt.title('Node Matrix Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if start_from_last_epochs[0] != 0:
        plt.savefig(f'{folder}/plot_last_{start_from_last_epochs[0]}epochs.png')
        if save_in is not None:
            plt.savefig(f"{save_in[0]}/plot_last_{start_from_last_epochs[0]}epochs_{save_in[1]}")
        print(f"Plot saved as {folder}/training_history_last_{start_from_last_epochs[0]}epochs.png")
    elif start_from != 0:
        plt.savefig(f'{folder}/training_history_startfrom_{start_from}epochs.png')
        if save_in is not None:
            plt.savefig(f"{save_in[0]}/plot_startfrom_{start_from}epochs_{save_in[1]}")
        print(f"Plot saved as {folder}/training_history_startfrom_{start_from}_epochs.png")
    else:
        plt.savefig(f'{folder}/training_history.png')
        if save_in is not None:
            plt.savefig(f"{save_in[0]}/training_history_{save_in[1]}")
        print(f"Plot saved as {folder}/training_history.png")
    plt.close()
    

if __name__ == "__main__":
    main()