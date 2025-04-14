"""Eigenvalues example.
"""
import torch
from os.path import abspath
from datasets import load_from_disk
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from plots import plot_matrices_true_prediction_difference

from hforge.graph_dataset import graph_from_row
from hforge.plots.plot_matrix import reconstruct_matrix
from hforge.model_shell import ModelShell

from hforge.mace.modules import RealAgnosticResidualInteractionBlock



def main():
    #* Change the row number to test different samples:
    predicted_h, original_h = generate_prediction(row=10, want_plot=True)

    # Calculate eigenvalues
    #! At the moment we are not taking into account the overlap matrix
    predicted_eigenvalues, predicted_eigenvectors = eigh(predicted_h)
    original_eigenvalues, original_eigenvectors = eigh(original_h)

    # Plot eigenvalues percentage difference
    save_path = "./EXAMPLE_info/eigenvalue_difference.png"
    plot_eigenvalue_percentage_difference(predicted_eigenvalues, original_eigenvalues, save_path, ascending_order=True)

    # Plot eigenvalues
    save_path = "./EXAMPLE_info/eigenvalues.png"
    plot_eigenvalue(predicted_eigenvalues, original_eigenvalues, save_path, ascending_order=True, zoom_y_bounds=(-100,100))

def plot_eigenvalue(predicted, original, save_path, ascending_order=False, zoom_y_bounds=None):
    """Plots the predicted and original eigenvalues.

    Args:
        predicted (array-like): Predicted eigenvalues.
        original (array-like): Original eigenvalues.
        save_path (str): Path to save the plot.
    """
    # Select the range of eigenvalues to plot
    x_i = 0 #* Change
    x_f = -1 #* Change
    y_i = zoom_y_bounds[0] if zoom_y_bounds else None
    y_f = zoom_y_bounds[1] if zoom_y_bounds else None

    # Sort the eigenvalues in ascending order if specified
    if ascending_order:
        predicted = sorted(predicted)
        original = sorted(original)
        x_label = 'Eigenvalue Index (Sorted)'

    plt.figure(figsize=(10, 6))
    x = range(len(original))

    # Plot the original eigenvalues
    plt.plot(x[x_i:x_f], original[x_i:x_f], marker='.', linestyle='', color='g', label='Original')

    # Plot the predicted eigenvalues
    plt.plot(x[x_i:x_f], predicted[x_i:x_f], marker='.', linestyle='', color='b', label='Predicted')


    plt.xlabel(x_label)
    plt.ylabel('Energy (eV)')
    plt.title('Eigenvalues')

    # Zoom in Y axis if specified
    plt.ylim(y_i, y_f) if zoom_y_bounds else None

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Eigenvalue difference plot saved to {save_path}.")

def plot_eigenvalue_percentage_difference(predicted, original, save_path, ascending_order=False):
    """Plots the percentage difference between predicted and original eigenvalues.

    Args:
        predicted (array-like): Predicted eigenvalues.
        original (array-like): Original eigenvalues.
        save_path (str): Path to save the plot.
    """
    # Select the range of eigenvalues to plot
    i = 3 #* Change
    f = 100 #* Change

    percentage_difference = ((predicted - original) / original) * 100

    # Sort the eigenvalues in ascending order if specified
    if ascending_order:
        percentage_difference = sorted(percentage_difference)
        x_label = 'Eigenvalue Index (Sorted)'

    plt.figure(figsize=(10, 6))
    x = range(len(percentage_difference))
    plt.plot(x[i:f], percentage_difference[i:f], marker='.', linestyle='', color='b', label='Percentage Difference')
    plt.axhline(0, color='r', linestyle='--', linewidth=1, label='Zero Difference')
    plt.xlabel(x_label)
    plt.ylabel('Difference (%)')
    plt.title('Percentage Difference Between Predicted and Original Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Eigenvalue difference plot saved to {save_path}.")


def generate_prediction(row, want_plot=True):
    """Loads raw dataset, converts it to a graph and returns the predicted and original Hamiltonian matrices.

    Args:
        row (int): The row from the dataset you want to make a prediction for.
        want_plot (bool, optional): Plots the results in a .png file. Defaults to True.

    Returns:
        predicted_h: The predicted Hamiltonian matrix.
        original_h: The original Hamiltonian matrix.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu' #! Idk why but with GPU is ~20 seconds slower than CPU per epoch
    print(f"Using device: {device}.")

    ########

    # TODO: Save the config in a json file instead of writing it here.
    # Initialize model
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
    nr_bits = 10
    config_model = {
        "embedding": {
            'hidden_irreps': "8x0e+8x1o",
            # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1

            "r_max": 3,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "radial_type": "bessel",
            "distance_transform": None,
            "max_ell": 3,
            "num_elements": nr_bits,
            "orbitals": orbitals,
            "nr_bits": nr_bits

        },
        "atomic_descriptors": {
            'hidden_irreps': "8x0e+8x1o",
            ## 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            'avg_num_neighbors': avg_num_neighbors,  # need to be computed
            "radial_mlp": [64, 64, 64],
            'num_interactions': 2,
            "correlation": 3,  # correlation order of the messages (body order - 1)
            "num_elements": nr_bits,
            "max_ell": 3,
            "orbitals": orbitals,
        },

        "edge_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 900,
            "hidden_dim_matrix_extraction": 900,

        },

        "node_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 900,
            "hidden_dim_matrix_extraction": 900,

        },
    }
    model = ModelShell(config_model).to(device)
    # print(model)

    ########

    # Load the model
    folder = "./EXAMPLE_info/"
    model_filename = "train_best_model.pt"
    path = abspath(folder + model_filename)
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the graph data to make prediction
    dataset_path = "./Data/aBN_HSX/nr_atoms_32/"

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Convert to graph
    sample_graph = graph_from_row(dataset[row], orbitals, cutoff=3.0).to(device)

    ########

    # Make prediction
    with torch.no_grad():
        model.eval()

        # Forward pass
        output_graph = model(sample_graph)

        # Matrix reconstruction
        predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"], output_graph["edge_index"]).cpu().numpy()
        original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"]).cpu().numpy()

    # Plot comparison
    if want_plot == True:
        plot_matrices_true_prediction_difference(original_h, predicted_h, label='Hamiltonian', path=folder+"h_comparison.png")
    
    return predicted_h, original_h

if __name__ == "__main__":
    main()