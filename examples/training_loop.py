"""
Training model with Comet.ml tracking and Matplotlib live plotting with optimized loss
and advanced learning rate scheduling for faster convergence and lower loss
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

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
    Prepare dataset for training

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

    print("graph generation done!")

    # Split into train and validation
    split_idx = int(len(graph_dataset) * split_ratio)
    train_dataset = graph_dataset[:split_idx]
    val_dataset = graph_dataset[split_idx:]

    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return train_loader, val_loader


def cost_function(pred_graph, target_graph, scale_factor=100.0):
    """
    Calculate loss between predicted and target Hamiltonian and overlap matrices with dynamic weighting.

    Args:
        pred_graph: Dictionary containing predicted edge_description (h_hop) and node_description (s_on_sites)
        target_graph: Dictionary containing target h_hop and s_on_sites values
        scale_factor: Scale factor for target values

    Returns:
        Total loss combining Hamiltonian and overlap matrix losses
    """
    # Extract predictions and targets
    edge_pred = pred_graph["edge_description"]
    node_pred = pred_graph["node_description"]

    edge_target = target_graph["edge_description"] * scale_factor
    node_target = target_graph["node_description"] * scale_factor

    edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_target)
    node_loss = torch.nn.functional.mse_loss(node_pred, node_target)
    # Dynamic weighting based on loss magnitude
    edge_weight = 1.0
    node_weight = 1.0

    # Combined loss
    total_loss = edge_weight * edge_loss + node_weight * node_loss

    # Detect extremely large values and clip
    if total_loss > 1e6:
        print(f"Unusually large loss detected: {total_loss.item()}")
        total_loss = torch.clamp(total_loss, max=1e6)

    return total_loss, {"edge_loss": edge_loss.item(), "node_loss": node_loss.item()}


## TRAINER ##
class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cpu',
                 use_comet=False, live_plot=True, plot_update_freq=1, plot_path="training_plot.png",
                 lr_scheduler=None, grad_clip_value=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.plot_path = plot_path
        self.lr_scheduler = lr_scheduler
        self.grad_clip_value = grad_clip_value

        # Track stats for plateaus
        self.plateau_counter = 0
        self.plateau_patience = 10
        self.min_lr = 1e-6

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
                "device": device,
                "grad_clip_value": grad_clip_value
            })

        # Live plotting setup
        self.live_plot = live_plot
        self.plot_update_freq = plot_update_freq
        if live_plot:
            self.train_losses = []
            self.val_losses = []
            self.epochs = []
            self.learning_rates = []

    def train_epoch(self):
        """Run one epoch of training with gradient clipping for stability"""
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

            # Backward pass
            loss.backward()

            # Apply gradient clipping for stability
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            # Optimize
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
        """Run validation"""
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
                    "edge_index": pred_graph["edge_index"],
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
        plt.figure(figsize=(12, 8))

        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)

        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.learning_rates, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

        print(f"Updated training plot saved to {self.plot_path}")

    def check_for_plateau(self, val_loss, epoch):
        """Check if training has plateaued and adjust learning rate if needed"""
        if epoch > 0 and abs(val_loss - self.val_losses[-1]) < 1e-4:
            self.plateau_counter += 1

            if self.plateau_counter >= self.plateau_patience:
                # Reduce learning rate when plateau is detected
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr > self.min_lr:
                    new_lr = current_lr * 0.5
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Plateau detected! Reducing learning rate from {current_lr} to {new_lr}")
                    self.plateau_counter = 0
                    return True
        else:
            self.plateau_counter = 0
        return False

    def train(self, num_epochs, save_path=None):
        """
        Train the model for specified number of epochs with learning rate scheduling

        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model (optional)

        Returns:
            Trained model and history of training/validation losses
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_edge_loss': [],
            'train_node_loss': [],
            'val_edge_loss': [],
            'val_node_loss': [],
            'learning_rate': []
        }

        start_time = time.time()

        # Create checkpoint directory
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        # Create a separate path for periodic checkpoints
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

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

            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            # Update learning rate scheduler if provided
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            else:
                # Manually check for plateaus
                self.check_for_plateau(val_loss, epoch)

            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time

            # Update tracking for plotting
            if self.live_plot:
                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(current_lr)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} (Edge: {train_components['edge_loss']:.4f}, Node: {train_components['node_loss']:.4f}) - "
                  f"Val Loss: {val_loss:.4f} (Edge: {val_components['edge_loss']:.4f}, Node: {val_components['node_loss']:.4f}) - "
                  f"LR: {current_lr:.6f} - "
                  f"Time: {epoch_time:.2f}s - Total: {elapsed_time / 60:.2f}m")

            # Log to Comet.ml
            if self.use_comet:
                self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
                self.experiment.log_metric("train_edge_loss", train_components["edge_loss"], epoch=epoch)
                self.experiment.log_metric("train_node_loss", train_components["node_loss"], epoch=epoch)
                self.experiment.log_metric("val_loss", val_loss, epoch=epoch)
                self.experiment.log_metric("val_edge_loss", val_components["edge_loss"], epoch=epoch)
                self.experiment.log_metric("val_node_loss", val_components["node_loss"], epoch=epoch)
                self.experiment.log_metric("learning_rate", current_lr, epoch=epoch)
                self.experiment.log_metric("epoch_time", epoch_time, epoch=epoch)

            # Update live plot
            if self.live_plot and (epoch % self.plot_update_freq == 0 or epoch == num_epochs - 1):
                self.update_plot()

            # Save periodic checkpoint (every 50 epochs)
            if epoch % 50 == 0 and epoch > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

            if save_path and train_loss < self.best_train_loss and epoch % 500 == 0:
                self.best_train_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, "train_"+save_path)
                tsp="train_"+save_path
                print(f"New best model saved to {tsp} (val_loss: {train_loss:.4f})")

                if self.use_comet:
                    self.experiment.log_model("best_model_training", save_path)

            # Save best model
            if save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, save_path)
                print(f"New best model saved to {save_path} (val_loss: {val_loss:.4f})")

                if self.use_comet:
                    self.experiment.log_model("best_model_validation", save_path)

        # Final plot update
        if self.live_plot:
            self.update_plot()
            self.create_final_plots(history)

            if self.use_comet and os.path.exists("training_history.png"):
                self.experiment.log_image("training_history.png", name="training_curves")

        # End experiment
        if self.use_comet:
            self.experiment.end()

        return self.model, history

    def create_final_plots(self, history):
        """Create final detailed plots from history"""
        plt.figure(figsize=(15, 10))

        # Main loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale for better visualization of loss improvements

        # Component losses
        plt.subplot(2, 2, 2)
        plt.plot(history['train_edge_loss'], 'b-', label='Train Edge Loss')
        plt.plot(history['val_edge_loss'], 'r-', label='Val Edge Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Edge Loss')
        plt.title('Edge Matrix Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        plt.subplot(2, 2, 3)
        plt.plot(history['train_node_loss'], 'b-', label='Train Node Loss')
        plt.plot(history['val_node_loss'], 'r-', label='Val Node Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Node Loss')
        plt.title('Node Matrix Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        # Learning rate plot
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rate'], 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.close()


def main():
    # Configuration
    dataset_path = "/Users/voicutomut/Documents/GitHub/Hforge/Data/aBN_HSX/nr_atoms_32"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Prepare dataset with larger batch size for better performance
    train_loader, val_loader = prepare_dataset(
        dataset_path=dataset_path,
        orbitals=orbitals,
        split_ratio=0.85,  # Slight increase in training data
        batch_size=8,      # Increased batch size for better gradient estimates
        cutoff=3.0,
        max_samples=None   # Use full dataset for better training
    )

    # Initialize model
    avg_num_neighbors = 8
    config_model = {
        "embedding": {

            'hidden_irreps': "8x0e+8x1o",
            # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1

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
            ## 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            'avg_num_neighbors': avg_num_neighbors,  # need to be computed
            "radial_mlp": [64, 64, 64],
            'num_interactions': 3,
            "correlation": 4,  # correlation order of the messages (body order - 1)
            "num_elements": 2,
            "max_ell": 2,
        },

        "edge_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 800,
            "hidden_dim_matrix_extraction": 800,

        },

        "node_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 800,
            "hidden_dim_matrix_extraction": 800,

        },

    }

    model = ModelShell(config_model)

    # Define optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,           # Lower initial learning rate
        weight_decay=1e-5  # Light L2 regularization
    )

    # Learning rate scheduler with warm-up and cosine annealing
    # This helps find better minima and escape plateaus
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # First restart cycle length
        T_mult=2,  # Increase cycle length after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=cost_function,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        use_comet=False,    # Set to True if you want to use Comet.ml
        live_plot=True,     # Generate plot files during training
        plot_update_freq=5, # Update plot every 5 epochs
        grad_clip_value=0.1 # Tighter gradient clipping for stability
    )

    # Train the model
    num_epochs = 2000
    save_path = "best_model.pt"

    model, history = trainer.train(num_epochs, save_path)

    # Test inference with trained model
    model.eval()

    # Get a sample from validation set
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

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()