import time

ti = time.time()
from hforge.model_shell import ModelShell
tf = time.time()
print(f"Time taken to import ModelShell: {tf - ti:.1f} seconds")

import torch
import os
import yaml
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix, plot_matrices_true_prediction_difference
from hforge.utils import prepare_dataset
from hforge.mace.modules import RealAgnosticResidualInteractionBlock

INTERACTION_BLOKS={"RealAgnosticResidualInteractionBlock":RealAgnosticResidualInteractionBlock}

# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss. 

def main():
    # === Load configuration ===
    folder_path = "example_results/already_trained_models/lr1e-3_wd1e-4_15T_1layersmp_shuffledseed4"
    with open(folder_path+"/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # === Device setup ===
    device = "cpu"

    # === Model Configuration ===
    model_config = config["model"]
    model_config["atomic_descriptors"]["interaction_cls_first"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls_first"]]
    model_config["atomic_descriptors"]["interaction_cls"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls"]]

    model = ModelShell(model_config).to(device)

    # Load the model
    filename = "/best_model.pt"
    path = os.path.abspath(folder_path + filename)
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the dataset
    dataset_config = config["dataset"]
    orbitals = config["orbitals"]

    train_dataset, val_dataset = prepare_dataset(
        dataset_path=dataset_config["path"],
        orbitals=orbitals,
        split_ratio=dataset_config["split_ratio"],
        batch_size=dataset_config["batch_size"],
        cutoff=dataset_config["cutoff"],
        max_samples=dataset_config["max_samples"],
        load_other_nr_atoms=dataset_config["load_other_nr_atoms"],
        return_dataloaders=False
    )

    # Make prediction
    model.eval()
    with torch.no_grad():
        for i, sample_graph in enumerate(train_dataset):
            output_graph = model(sample_graph)
            target_graph = {
                "edge_index": output_graph["edge_index"],
                "edge_description": sample_graph.h_hop,
                "node_description": sample_graph.s_on_sites
            }

            predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"], output_graph["edge_index"])
            original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])

            fig = plot_comparison_matrices(original_h*100, predicted_h, save_path=f"{folder_path}/plot_h_comparison_train_{i}.html", want_png_percent_diff=True)
            print("Generated plot ", i)
            # fig.show()

            # Angel's plot: 
            # Multiplication *100 is needed because it is done like that in the cost function
            plot_matrices_true_prediction_difference(original_h.cpu().numpy(), predicted_h.cpu().numpy()/100, path=f"{folder_path}/plot_angel__h_comparison_{i}.png", label="hamiltonian")


if __name__ == "__main__":
    main()