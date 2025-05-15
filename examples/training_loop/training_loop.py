"""
Training model with Comet.ml tracking and Matplotlib live plotting with optimized loss
and advanced learning rate scheduling for faster convergence and lower loss
"""

# Imports
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Comet ML not installed. Experiment tracking will be disabled.")

from hforge.utils import prepare_dataset, load_model, prepare_dataloaders
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell
from hforge.graph_costfucntion import  mse_cost_function, scale_dif_cost_function
from hforge.trainers.default_trainer import Trainer
import yaml
import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

def save_to_yaml(data, path):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

COST_FN={"mse_cost_function":mse_cost_function,
          "scale_dif_cost_function":scale_dif_cost_function}
INTERACTION_BLOKS={"RealAgnosticResidualInteractionBlock":RealAgnosticResidualInteractionBlock}



def main():

    # === Load configuration ===
    with open("examples/training_loop/training_loop_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_path=config["exp_path"]
    create_directory(exp_path)
    save_to_yaml(config, f"{exp_path}/training_config.yaml")

    # === Device setup ===
    device = torch.device('cuda' if (torch.cuda.is_available() and config["device"]!="cpu") else 'cpu')
    print(f"Using device: {device}")

    # === Dataset Preparation ===
    dataset_config = config["dataset"]
    orbitals = config["orbitals"]

    # TODO: Better? to, instead of keeping all the dataset loaded in memory, first save the conversion to graph in the disk and then load it from there. But we will then have to keep all the graphs loaded anyways! How can we do better?
    # Convert the data into graphs dataset
    train_dataset, val_dataset = prepare_dataset(
        dataset_path=dataset_config["path"],
        orbitals=orbitals,
        split_ratio=dataset_config["split_ratio"],
        cutoff=dataset_config["cutoff"],
        max_samples=dataset_config["max_samples"],
        load_other_nr_atoms=dataset_config["load_other_nr_atoms"]
    )

    # Create the dataloaders
    train_loader, val_loader = prepare_dataloaders(train_dataset, val_dataset, batch_size=config["batch_size"])

    # === Model Configuration ===
    model_config = config["model"]
    # Inject classes into model config (since YAML can't store class references)
    model_config["atomic_descriptors"]["interaction_cls_first"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls_first"]]
    model_config["atomic_descriptors"]["interaction_cls"] = INTERACTION_BLOKS[model_config["atomic_descriptors"]["interaction_cls"]]

    model = ModelShell(model_config, device=device).to(device)
    # print("\n Model:\n",model)

    # === Model loading ===
    path_trained_model = model_config["path_trained_model"]
    if path_trained_model:
        model, _, _, history = load_model(model, path=path_trained_model, device=device)
    else:
        history = None
        print("No pretrained model found. Starting training from scratch.")

    # === Optimizer ===
    optimizer_config = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"])
    )

    # === Scheduler ===
    scheduler_config = config["scheduler"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_config["T_0"]),
        T_mult=scheduler_config["T_mult"],
        eta_min=float(scheduler_config["eta_min"])
    )

    # === Cost function ===
    cost_function = COST_FN[config["cost_function"]["function"]]

    # === Trainer initialization ===
    trainer_config = config["trainer"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=cost_function,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        use_comet=trainer_config["use_comet"],
        live_plot=trainer_config["live_plot"],
        plot_update_freq=trainer_config["plot_update_freq"],
        grad_clip_value=trainer_config["grad_clip_value"],
        history=history,
        training_info_path=exp_path
    )

    # === Start training ===
    print("\nTraining starts:")
    model, history=trainer.train(
        num_epochs=trainer_config["num_epochs"],
        filename=trainer_config["filename"]
    )
    print("\nTraining completed successfully!")


    # Test inference with trained model
    model.eval()

    # Get a sample from training set
    sample_graph = next(iter(train_loader))
    if isinstance(sample_graph, list):
        sample_graph = sample_graph[0]

    sample_graph = sample_graph.to(device)

    # Forward pass
    with torch.no_grad():
        output_graph = model(sample_graph)

        # Create target graph for comparison
        target_graph = {
            "edge_index": output_graph["edge_index"],
            "edge_description": sample_graph.h_hop,
            "node_description": sample_graph.s_on_sites
        }

        # Calculate final inference loss
        inference_loss, component_losses = cost_function(output_graph, target_graph)

        print("\nFinal inference results:")
        print(f"Total loss: {inference_loss.item():.4f}")
        print(f"Edge loss: {component_losses['edge_loss']:.4f}")
        print(f"Node loss: {component_losses['node_loss']:.4f}")

        predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"],
                                         output_graph["edge_index"])
        original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])
        fig = plot_comparison_matrices(original_h * 100, predicted_h, save_path=f"{exp_path}/matrix_comparison_New_train.html")

        # Display the plots
        fig.show()

        # Get a sample from trsin set
    sample_graph = next(iter(val_loader))
    if isinstance(sample_graph, list):
        sample_graph = sample_graph[0]

    sample_graph = sample_graph.to(device)

    # Forward pass
    with torch.no_grad():
        output_graph = model(sample_graph)

        # Create target graph for comparison
        target_graph = {
            "edge_index": output_graph["edge_index"],
            "edge_description": sample_graph.h_hop,
            "node_description": sample_graph.s_on_sites
        }

        # Calculate final inference loss
        inference_loss, component_losses = cost_function(output_graph, target_graph)

        print("\nFinal inference results:")
        print(f"Total loss: {inference_loss.item():.4f}")
        print(f"Edge loss: {component_losses['edge_loss']:.4f}")
        print(f"Node loss: {component_losses['node_loss']:.4f}")

        predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"],
                                         output_graph["edge_index"])
        original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])
        fig = plot_comparison_matrices(original_h * 100, predicted_h,
                                       save_path=f"{exp_path}/matrix_comparison_New_val.html")

        # Display the plots
        fig.show()

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()