"""
Training model with Comet.ml tracking and Matplotlib live plotting (no IPython dependency)
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Try to import comet_ml, but make it optional
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Comet ML not installed. Experiment tracking will be disabled.")

from datasets import load_from_disk
from hforge.graph_dataset import graph_from_row
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell


def prepare_dataset(dataset_path, orbitals, split_ratio=0.8, batch_size=1, cutoff=4.0, max_samples=None):
    """
    Prepare dataset for training by returning the train and validation data loaders.

    Args:
        dataset_path: Path to the dataset
        orbitals: Dictionary mapping atomic numbers to number of orbitals
        split_ratio: Train/validation split ratio
        batch_size: Batch size for dataloaders
        cutoff: Cutoff distance for graph construction
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        train_loader, val_loader
    """
    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Convert dataset to graph form
    graph_dataset = []
    sample_count = 0
    for row in dataset:
        graph = graph_from_row(row, orbitals, cutoff=cutoff)
        graph_dataset.append(graph)
        sample_count += 1

        # Break if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break

    print("Graph generation done!")

    # Split into train and validation
    split_idx = int(len(graph_dataset) * split_ratio)
    train_dataset = graph_dataset[:split_idx]
    val_dataset = graph_dataset[split_idx:]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    return train_loader, val_loader


def cost_function(pred_graph, target_graph):
    """
    Calculate loss using MeanSquaredError between predicted and target Hamiltonian and overlap matrices.

    Args:
        pred_graph: Dictionary containing predicted edge_description (h_hop) and node_description (s_on_sites)
        target_graph: Dictionary containing target h_hop and s_on_sites values

    Returns:
        total_loss: The sum of edge_loss and node_loss.
        (Dict): Dictionary containing edge_loss and node_loss separately.
    """
    # Extract predictions and targets
    edge_pred = pred_graph["edge_description"]
    node_pred = pred_graph["node_description"]

    edge_target = target_graph["edge_description"]*100
    node_target = target_graph["node_description"]*100

    # Compute MSE loss for both matrices
    edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_target)
    node_loss = torch.nn.functional.mse_loss(node_pred, node_target)

    # Combined loss (can add weights if needed)
    total_loss = edge_loss + node_loss

    return total_loss, {"edge_loss": edge_loss.item(), "node_loss": node_loss.item()}


## TRAINER ##
class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cpu',
                 use_comet=False, live_plot=True, plot_update_freq=1, plot_path=os.path.abspath("./results/training_plot.png")):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')
        self.plot_path = plot_path

        # Comet.ml integration
        self.use_comet = use_comet and COMET_AVAILABLE
        if self.use_comet:
            self.experiment = Experiment(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="hforge-training",
                workspace=os.environ.get("COMET_WORKSPACE")
            )
            # Log hyperparameters
            self.experiment.log_parameters({
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else "unknown",
                "device": device
            })

        # Live plotting setup
        self.live_plot = live_plot
        self.plot_update_freq = plot_update_freq
        if live_plot:
            self.train_losses = []
            self.val_losses = []
            self.epochs = []

    def train_epoch(self):
        """Run one epoch of training

        Returns:
            avg_loss: Average total loss for the epoch.
            (Dict): Dictionary containing average edge loss and average node loss separately.
        """
        self.model.train()
        total_loss = 0.0
        total_edge_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            pred_graph = self.model(batch)

            # Create target graph
            target_graph = {
                "edge_index": pred_graph["edge_index"],
                "edge_description": batch.h_hop,
                "node_description": batch.s_on_sites
            }

            # Calculate loss
            loss, component_losses = self.loss_fn(pred_graph, target_graph)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_edge_loss += component_losses["edge_loss"]
            total_node_loss += component_losses["node_loss"]
            num_batches += 1

        # Average losses
        if num_batches == 0:
            return 0.0, {"edge_loss": 0.0, "node_loss": 0.0}

        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        return avg_loss, {"edge_loss": avg_edge_loss, "node_loss": avg_node_loss}

    def validate(self):
        """Run validation of the model

        Returns:
            avg_loss: Average total loss for the epoch.
            (Dict): Dictionary containing average edge loss and average node loss separately.
        """
        self.model.eval()

        total_loss = 0.0
        total_edge_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass
                pred_graph = self.model(batch)

                # Create target graph
                target_graph = {
                    "edge_index": pred_graph["edge_index"],  # Note the typo in the key name
                    "edge_description": batch.h_hop,
                    "node_description": batch.s_on_sites
                }

                # Calculate loss
                loss, component_losses = self.loss_fn(pred_graph, target_graph)

                total_loss += loss.item()
                total_edge_loss += component_losses["edge_loss"]
                total_node_loss += component_losses["node_loss"]
                num_batches += 1

        # Average losses
        if num_batches == 0:
            return 0.0, {"edge_loss": 0.0, "node_loss": 0.0}

        avg_loss = total_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        avg_node_loss = total_node_loss / num_batches

        return avg_loss, {"edge_loss": avg_edge_loss, "node_loss": avg_node_loss}

    def update_plot(self):
        """Update the live plot with new loss values"""
        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot training and validation loss
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')

        # Add labels and legend
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Set y-axis to start from 0
        plt.ylim(bottom=0)

        # Save the plot to a file
        plt.savefig(self.plot_path)
        plt.close()  # Close the plot to avoid displaying it

        print(f"Updated training plot saved to {self.plot_path}")

    def train(self, num_epochs, save_path=None):
        """
        Train the model for specified number of epochs

        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model (optional)

        Returns:
            model: Trained model
            history (Dict): History of training/validation losses
        """
        history = {
            # Total losses
            'train_loss': [],
            'val_loss': [],

            # Component losses
            'train_edge_loss': [],
            'train_node_loss': [],
            'val_edge_loss': [],
            'val_node_loss': [],
        }

        # Track the time of training.
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_components = self.train_epoch()
            history['train_loss'].append(train_loss)
            history['train_edge_loss'].append(train_components["edge_loss"])
            history['train_node_loss'].append(train_components["node_loss"])

            # Validation phase
            val_loss, val_components = self.validate()
            history['val_loss'].append(val_loss)
            history['val_edge_loss'].append(val_components["edge_loss"])
            history['val_node_loss'].append(val_components["node_loss"])

            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time

            # Update tracking for plotting
            if self.live_plot:
                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} (Edge: {train_components['edge_loss']:.4f}, Node: {train_components['node_loss']:.4f}) - "
                  f"Val Loss: {val_loss:.4f} (Edge: {val_components['edge_loss']:.4f}, Node: {val_components['node_loss']:.4f}) - "
                  f"Time: {epoch_time:.2f}s - Total: {elapsed_time/60:.2f}m")

            # Log to Comet.ml
            if self.use_comet:
                self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
                self.experiment.log_metric("train_edge_loss", train_components["edge_loss"], epoch=epoch)
                self.experiment.log_metric("train_node_loss", train_components["node_loss"], epoch=epoch)
                self.experiment.log_metric("val_loss", val_loss, epoch=epoch)
                self.experiment.log_metric("val_edge_loss", val_components["edge_loss"], epoch=epoch)
                self.experiment.log_metric("val_node_loss", val_components["node_loss"], epoch=epoch)
                self.experiment.log_metric("epoch_time", epoch_time, epoch=epoch)

            # Update live plot
            if self.live_plot and (epoch % self.plot_update_freq == 0 or epoch == num_epochs - 1):
                self.update_plot()

            # Save best model
            if save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, save_path)
                print(f"Model saved to {save_path}")

                if self.use_comet:
                    self.experiment.log_model("best_model", save_path)

        # Final plot update
        if self.live_plot:
            self.update_plot()

            # Create final detailed plots
            self.create_final_plots(history)

            if self.use_comet and os.path.exists("training_history.png"):
                self.experiment.log_image("training_history.png", name="training_curves")

        # End experiment
        if self.use_comet:
            self.experiment.end()

        return self.model, history

    def create_final_plots(self, history):
        """Create final detailed plots from history of  training/validation losses"""
        plt.figure(figsize=(12, 8))

        # Main loss plot
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Component losses
        plt.subplot(2, 2, 3)
        plt.plot(history['train_edge_loss'], 'b-', label='Train Edge Loss')
        plt.plot(history['val_edge_loss'], 'r-', label='Val Edge Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Edge Loss')
        plt.title('Edge Matrix Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(history['train_node_loss'], 'b-', label='Train Node Loss')
        plt.plot(history['val_node_loss'], 'r-', label='Val Node Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Node Loss')
        plt.title('Node Matrix Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()


def main():
    # Configuration
    dataset_path = "./Data/aBN_HSX/nr_atoms_32"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define orbital configuration based on atomic types
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

    # Prepare dataset
    train_loader, val_loader = prepare_dataset(
        dataset_path=dataset_path,
        orbitals=orbitals,
        split_ratio=0.8,
        batch_size=4,
        cutoff=3.0,
        max_samples=20  # Limit samples for testing/debugging
    )

    # Initialize model
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

    model = ModelShell(config_model)

    # Define optimizer and learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=cost_function,
        optimizer=optimizer,
        device=device,
        use_comet=False,    # Set to True if you want to use Comet.ml
        live_plot=True,     # Generate plot files during training
        plot_update_freq=1  # Update plot every epoch
    )

    # Train the model
    num_epochs = 200
    save_path = os.path.abspath("./results/best_model.pt")

    model, history = trainer.train(num_epochs, save_path)

    # Test inference with trained model
    model.eval()

    # Get a sample from validation set
    sample_graph = next(iter(val_loader))
    if isinstance(sample_graph, list):
        sample_graph = sample_graph[0]

    # Forward pass
    output_graph = model(sample_graph)

    # Print output
    print("\nInference with trained model:")
    print(f"Output graph keys: {output_graph.keys()}")
    for key in output_graph.keys():
        print(f"{key}: {output_graph[key].shape}")


if __name__ == "__main__":
    main()