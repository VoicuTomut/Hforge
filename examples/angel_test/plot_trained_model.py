import torch
import os
from hforge.model_shell import ModelShell
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model.
    orbitals = {
        1: 13,
        2: 13,
        3: 13,
        4: 13,
        5: 13,
        6: 13,
        7: 13,
        8: 13,
    }
    avg_num_neighbors = 8
    config_model = {
        "embedding": {
            'hidden_irreps': "8x0e+8x1o",
            "r_max": 3,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "radial_type": "bessel",
            "distance_transform": None,
            "max_ell": 2,
            "num_elements": 2,
        },
        "atomic_descriptors": {
            'hidden_irreps': "8x0e+8x1o",
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            'avg_num_neighbors': avg_num_neighbors,
            "radial_mlp": [64, 64, 64],
            'num_interactions': 2,
            "correlation": 3,
            "num_elements": 2,
            "max_ell": 2,
        },
        "edge_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 400,
            "hidden_dim_matrix_extraction": 300,
        },
        "node_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 300,
            "hidden_dim_matrix_extraction": 200,
        },
    }
    model = ModelShell(config_model).to(device)

    # Load the model
    folder = "./EXAMPLE_info_important/"
    filename = "best_model_650Epochs_0.01lr.pt"
    path = os.path.abspath(folder + filename)
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create the plots
    history = checkpoint["history"]
    create_final_plots(history)

# def update_plot(train_losses, val_losses, plot_path):
#     """Update the live plot with new loss values"""
#     epochs = len(train_losses)
#     if epochs != len(val_losses):
#         raise ValueError("The length of train_losses and val_losses do not coincide.")

#     # Create a new figure
#     plt.figure(figsize=(10, 6))

#     # Plot training and validation loss
#     plt.plot(epochs, train_losses, 'b-', label='Training Loss')
#     plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

#     # Add labels and legend
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)

#     # Set y-axis to start from 0
#     plt.ylim(bottom=0)

#     # Save the plot to a file
#     plt.savefig(plot_path)
#     plt.close()  # Close the plot to avoid displaying it

#     print(f"Updated training plot saved to {plot_path}")

def create_final_plots(history):
    """Create final detailed plots from history of  training/validation losses"""
    start_from = 10

    epochs = range(len(history["train_loss"]))[start_from:]
    if len(epochs) != len(history["val_loss"][start_from:]):
        raise ValueError("The length of train_loss and val_loss do not coincide.")
    plt.figure(figsize=(12, 8))

    # Main loss plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'][start_from:], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'][start_from:], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Component losses
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_edge_loss'][start_from:], 'b-', label='Train Edge Loss')
    plt.plot(epochs, history['val_edge_loss'][start_from:], 'r-', label='Val Edge Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Edge Loss')
    plt.title('Edge Matrix Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_node_loss'][start_from:], 'b-', label='Train Node Loss')
    plt.plot(epochs, history['val_node_loss'][start_from:], 'r-', label='Val Node Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Node Loss')
    plt.title('Node Matrix Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./EXAMPLE_info/training_history_pretrained.png')
    plt.close()


if __name__ == "__main__":
    main()