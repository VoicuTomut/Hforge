"""
Training model with Comet.ml tracking and Matplotlib live plotting with optimized loss
and advanced learning rate scheduling for faster convergence and lower loss
"""
import yaml

from hforge.data_management.dataset_load import load_and_process_raw_dataset_from_parent_dir, split_raw_dataset, \
    prepare_dataloaders, load_preprocessed_dataset_from_parent_dir, split_graph_dataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts

from hforge.utils import load_config, get_object_from_module
from hforge.utils.model_load import load_model

try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Comet ML not installed. Experiment tracking will be disabled.")

from hforge.model_shell import ModelShell
from hforge.trainers.default_trainer import Trainer
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


def main():

    # === Load configuration ===
    config = load_config()

    # Save current config in results directory
    results_dir=config["results_dir"]
    create_directory(results_dir)
    save_to_yaml(config, f"{results_dir}/training_config.yaml")

    # === Device setup ===
    device = torch.device('cuda' if (torch.cuda.is_available() and config["device"]!="cpu") else 'cpu')
    print(f"Using device: {device}")

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

    train_dataset, val_dataset, _ = split_graph_dataset(
        dataset=dataset,
        training_split_ratio=dataset_config["training_split_ratio"],
        test_split_ratio=dataset_config["test_split_ratio"],
        seed=dataset_config["seed"],
        print_finish_message=True
    )

    train_loader, val_loader = prepare_dataloaders(train_dataset, val_dataset, batch_size=dataset_config["batch_size"])

    # === Model Configuration ===
    model_config = config["model"]
    # Inject classes into model config (since YAML can't store class references)
    model_config["atomic_descriptors"]["interaction_cls_first"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls_first"], module='hforge.mace.modules')
    model_config["atomic_descriptors"]["interaction_cls"] = get_object_from_module(model_config["atomic_descriptors"]["interaction_cls"], module='hforge.mace.modules')

    model = ModelShell(model_config, device=device).to(device)

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
    cost_function = get_object_from_module(config["cost_function"]["function"], 'hforge.graph_costfunction')

    # === Trainer initialization ===
    trainer_config = config["trainer"]
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
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
        training_info_path=results_dir,
        plot_matrices_freq=trainer_config["plot_matrices_freq"],
        config=config,
        # TODO: Maybe it's a better idea to move all the dataloaders logic intor the Trainer class, instead of passing both datasets and dataloaders
    )

    # === Start training ===
    print("\nTraining starts:")
    model, history=trainer.train(
        num_epochs=trainer_config["num_epochs"],
        filename=trainer_config["filename"]
    )
    print("\nTraining completed successfully!")


    # # Test inference with trained model
    # model.eval()
    #
    # # Get a sample from training set
    # sample_graph = next(iter(train_loader))
    # if isinstance(sample_graph, list):
    #     sample_graph = sample_graph[0]
    #
    # sample_graph = sample_graph.to(device)

    # # Forward pass
    # with torch.no_grad():
    #     output_graph = model(sample_graph)
    #
    #     # Create target graph for comparison
    #     target_graph = {
    #         "edge_index": output_graph["edge_index"],
    #         "edge_description": sample_graph.h_hop,
    #         "node_description": sample_graph.s_on_sites
    #     }
    #
    #     # Calculate final inference loss
    #     inference_loss, component_losses = cost_function(output_graph, target_graph)
    #
    #     print("\nFinal inference results:")
    #     print(f"Total loss: {inference_loss.item():.4f}")
    #     print(f"Edge loss: {component_losses['edge_loss']:.4f}")
    #     print(f"Node loss: {component_losses['node_loss']:.4f}")

        # predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"],
        #                                  output_graph["edge_index"])
        # original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])
        # fig = plot_comparison_matrices(original_h * 100, predicted_h, save_path=f"{results_dir}/matrix_comparison_New_train.html")
        #
        # # Display the plots
        # fig.show()

    #     # Get a sample from trsin set
    # sample_graph = next(iter(val_loader))
    # if isinstance(sample_graph, list):
    #     sample_graph = sample_graph[0]
    #
    # sample_graph = sample_graph.to(device)
    #
    # # Forward pass
    # with torch.no_grad():
    #     output_graph = model(sample_graph)
    #
    #     # Create target graph for comparison
    #     target_graph = {
    #         "edge_index": output_graph["edge_index"],
    #         "edge_description": sample_graph.h_hop,
    #         "node_description": sample_graph.s_on_sites
    #     }
    #
    #     # Calculate final inference loss
    #     inference_loss, component_losses = cost_function(output_graph, target_graph)
    #
    #     print("\nFinal inference results:")
    #     print(f"Total loss: {inference_loss.item():.4f}")
    #     print(f"Edge loss: {component_losses['edge_loss']:.4f}")
    #     print(f"Node loss: {component_losses['node_loss']:.4f}")
    #
    #     predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"],
    #                                      output_graph["edge_index"])
    #     original_h = reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])
    #     fig = plot_comparison_matrices(original_h * 100, predicted_h,
    #                                    save_path=f"{results_dir}/matrix_comparison_New_val.html")
    #
    #     # Display the plots
    #     fig.show()

    # print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()