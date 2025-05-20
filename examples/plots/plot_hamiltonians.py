"""Plots the hamiltonian matrices predictions of the specified model for all samples in the dataset with which it has been trained/validated."""

# Standard library imports
import torch

# Local application imports
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix, plot_error_matrices, \
    plot_error_matrices_interactive
from hforge.utils import load_model_and_dataset_from_directory, create_directory, get_object_from_module


# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss.

def main():
    # === Load model ===

    # * Change the directory where the model is saved with its configuration .yaml file.
    directory = r"C:\Users\angel\OneDrive - Universitat de Barcelona\1A. MASTER I\TFM\example_results\usetapprox\usetapprox_2mp_sharing_resume"
    model_filename = "train_best_model.pt"

    model, history, train_dataset, validation_dataset, config = load_model_and_dataset_from_directory(directory, model_filename, weights_only=False, return_datasets=True)

    # === Generate prediction and plots ===

    # * Change results directory
    results_directory = directory + "/" + "hamiltonian_plots"

    create_directory(results_directory)

    # Generate a plot for all elements in the dataset
    model.eval()
    with torch.no_grad():
        # * Change between training or validation dataset as you want.
        dataset_type = "training"

        if dataset_type == "training":
            dataset = train_dataset
        elif dataset_type == "validation":
            dataset = validation_dataset
        else:
            raise ValueError("Invalid dataset type")

        for i, sample_graph in enumerate(dataset):
            # Generate prediction
            output_graph = model(sample_graph)

            # Compute loss
            loss_fn_name = config["cost_function"]["function"]
            loss_fn = get_object_from_module(loss_fn_name, "hforge.graph_costfucntion")
            target_graph = {
                "edge_index": output_graph["edge_index"],
                "edge_description": sample_graph.h_hop,
                "node_description": sample_graph.h_on_sites
            }
            loss, _ = loss_fn(output_graph, target_graph)

            # Reconstruct hamiltonians
            predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"], output_graph["edge_index"])
            original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])

            # Get training losses
            history_training_loss = history.get("train_loss")[-1]
            history_last_epoch = len(history.get("train_loss"))

            n_atoms = len(sample_graph["x"])
            figure_title = f"Results of sample {i} from {dataset_type} dataset (seed: 4). {n_atoms} atoms in the unit cell."
            # Plotly:
            # plot_comparison_matrices(original_h*100, predicted_h, save_path=f"{results_directory}/hamiltonian_{i}.html", want_png_percent_diff=False)
            filepath = f"{results_directory}/hamiltonian_{i}.html"
            predicted_matrix_text = f"Saved training loss at epoch {history_last_epoch}:     {history_training_loss:.2f} eV²·100<br>{loss_fn_name} evaluation:     {loss:.2f} eV²·100"
            plot_error_matrices_interactive(original_h.cpu().numpy(),
                                            predicted_h.cpu().numpy() / 100,
                                            filepath=filepath,
                                            matrix_label="Hamiltonian",
                                            predicted_matrix_text=predicted_matrix_text,
                                            figure_title=figure_title,
                                            n_atoms=n_atoms,
                                            )

            # Matplotlib:
            filepath = f"{results_directory}/hamiltonian_{i}.png"
            predicted_matrix_text = f"Saved training loss at epoch {history_last_epoch}:     {history_training_loss:.2f} eV²·100\n{loss_fn_name} evaluation:     {loss:.2f} eV²·100"
            plot_error_matrices(original_h.cpu().numpy(),
                                predicted_h.cpu().numpy() / 100,
                                filepath=filepath,
                                matrix_label="Hamiltonian",
                                predicted_matrix_text=predicted_matrix_text,
                                figure_title=figure_title,
                                n_atoms=n_atoms,
                                ) # Multiplication *100 is needed because it is done like that in the cost function

            print("Generated plots ", i)

if __name__ == "__main__":
    main()