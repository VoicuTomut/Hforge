"""Plots the hamiltonian matrices predictions of the specified model for all samples in the dataset with which it has been trained/validated."""

# Standard library imports
import torch

# Local application imports
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix, plot_matrices_true_prediction_difference
from hforge.utils import load_model_and_dataset_from_directory, create_directory

# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss. 

def main():
    # === Load model ===

    # * Change the directory where the model is saved with its configuration .yaml file.
    directory = r"C:\Users\angel\OneDrive - Universitat de Barcelona\1A. MASTER I\TFM\example_results\usetapprox\example_results_usetapprox_2mp_sharing"
    model_filename = "train_best_model.pt"

    model, _, train_dataset, validation_dataset = load_model_and_dataset_from_directory(directory, model_filename, weights_only=False, return_datasets=True)

    # === Generate prediction and plots ===

    # * Change results directory
    results_directory = directory + "/" + "hamiltonian_plots"

    create_directory(results_directory)

    # Generate a plots for all elements in the dataset
    model.eval()
    with torch.no_grad():
        # * Change between training or validation dataset as you want.
        for i, sample_graph in enumerate(train_dataset):
            # Generate prediction
            output_graph = model(sample_graph)

            # Reconstruct hamiltonians
            predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"], output_graph["edge_index"])
            original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])

            # Plotly:
            plot_comparison_matrices(original_h*100, predicted_h, save_path=f"{results_directory}/hamiltonian_{i}.html", want_png_percent_diff=False)

            # Matplotlib:
            plot_matrices_true_prediction_difference(original_h.cpu().numpy(), predicted_h.cpu().numpy()/100, path=f"{results_directory}/hamiltonian_{i}.png", label="Hamiltonian") # Multiplication *100 is needed because it is done like that in the cost function

            print("Generated plots ", i)


if __name__ == "__main__":
    main()