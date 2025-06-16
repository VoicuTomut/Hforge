"""Plots the hamiltonian matrices predictions of the specified model for all samples in the dataset with which it has been trained/validated."""

# Standard library imports
import os
import torch

# Local application imports
from hforge.data_management.dataset_load import load_preprocessed_dataset_from_parent_dir, split_graph_dataset
from hforge.model_shell import ModelShell
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix, plot_error_matrices, \
    plot_error_matrices_interactive
from hforge.utils import create_directory, get_object_from_module
from hforge.utils.importing_facilities import load_config
from hforge.utils.model_load import load_model


# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss.

def main():
    # === Load model ===

    # * Change the directory where the model is saved with its configuration .yaml file.
    directory = r"example_results/usetapprox_2mp_sharing_mse_mean"
    model_filename = "train_best_model.pt"

    # === Load configuration ===
    config = load_config(directory+"/training_config.yaml")

    # === Dataset Preparation ===
    dataset_config = config["dataset"]
    orbitals = config["orbitals"]

    dataset = load_preprocessed_dataset_from_parent_dir(
        parent_dir=dataset_config["path"],
        orbitals=orbitals,
        cutoff=dataset_config["cutoff"],
        max_samples=dataset_config["max_samples"],
        seed=dataset_config["seed"]
    )

    train_dataset, validation_dataset, _ = split_graph_dataset(
        dataset=dataset,
        training_split_ratio=dataset_config["training_split_ratio"],
        test_split_ratio=dataset_config["test_split_ratio"],
        seed=dataset_config["seed"],
        print_finish_message=True
    )

    # === Model Configuration ===
    model_config = config["model"]
    # Inject classes into model config (since YAML can't store class references)
    model_config["atomic_descriptors"]["interaction_cls_first"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls_first"], module='hforge.mace.modules')
    model_config["atomic_descriptors"]["interaction_cls"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls"], module='hforge.mace.modules')

    model = ModelShell(model_config)

    # === Model loading ===
    model_filename = "train_best_model.pt"
    model_path = os.path.abspath(directory + "/" + model_filename)
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    history = checkpoint.get('history', {})

    # model, history, train_dataset, validation_dataset, config = load_model_and_dataset_from_directory(directory, model_filename, weights_only=False, return_datasets=True)

    # === Generate prediction and plots ===

    # * Change results directory
    results_directory = directory + "/" + "hamiltonian_plots_after_training"

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
            loss_fn = get_object_from_module(loss_fn_name, "hforge.graph_costfunction")
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
            figure_title = f"Results of sample {i} from {dataset_type} dataset (seed: 4). There are {n_atoms} atoms in the unit cell."
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
                                            absolute_error_cbar_limit=0.1
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
                                absolute_error_cbar_limit=0.1
                                ) # Multiplication *100 is needed because it is done like that in the cost function

            print("Generated plots ", i)

if __name__ == "__main__":
    main()