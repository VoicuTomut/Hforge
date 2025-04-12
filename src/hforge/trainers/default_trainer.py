"""
Training model with Comet.ml tracking and Matplotlib live plotting with optimized loss
and advanced learning rate scheduling for faster convergence and lower loss
"""

# Imports

import time
import torch
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from comet_ml import Experiment

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Comet ML not installed. Experiment tracking will be disabled.")


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cpu',
                 use_comet=False, live_plot=True, plot_update_freq=1,training_info_path="", lr_scheduler=None, grad_clip_value=1.0, history=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.plot_path = f"{training_info_path}/training_plot_learning_rate.png"
        self.lr_scheduler = lr_scheduler
        self.grad_clip_value = grad_clip_value
        self.history = history
        self.training_info_path=training_info_path

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
                "node_description": batch.h_on_sites
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
                # print("TG:", batch.h_hop.shape)
                # Forward pass
                pred_graph = self.model(batch)

                # Create target graph
                target_graph = {
                    "edge_index": pred_graph["edge_index"],
                    "edge_description": batch.h_hop,
                    "node_description": batch.h_on_sites
                }


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

    def update_plot_learning_rate(self):
        """Update the live plot with new loss values"""
        start_from = 0

        epochs = range(len(self.history["train_loss"]))[start_from:]
        if len(epochs) != len(self.history["val_loss"][start_from:]):
            raise ValueError("The length of train_loss and val_loss do not coincide.")

        # Create a new figure
        plt.figure(figsize=(12, 8))

        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.history['train_loss'][start_from:], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'][start_from:], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)

        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.history['learning_rate'][start_from:], 'g-')
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

    def update_plot(self):
        """Create final detailed plots from history of training/validation losses"""
        start_from = 0

        epochs = range(len(self.history["train_loss"]))[start_from:]
        if len(epochs) != len(self.history["val_loss"][start_from:]):
            raise ValueError("The length of train_loss and val_loss do not coincide.")
        plt.figure(figsize=(12, 8))

        # Main loss plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.history['train_loss'][start_from:], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'][start_from:], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Component losses
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history['train_edge_loss'][start_from:], 'b-', label='Train Edge Loss')
        plt.plot(epochs, self.history['val_edge_loss'][start_from:], 'r-', label='Val Edge Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Edge Loss')
        plt.title('Edge Matrix Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history['train_node_loss'][start_from:], 'b-', label='Train Node Loss')
        plt.plot(epochs, self.history['val_node_loss'][start_from:], 'r-', label='Val Node Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Node Loss')
        plt.title('Node Matrix Loss')
        plt.legend()
        plt.grid(True)

        # TODO: Add flags where each data type training begins

        plt.tight_layout()
        plt.savefig(f'{self.training_info_path}/training_history.png')
        plt.close()

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

    def train(self, num_epochs, filename=None):
        """
        Train the model for specified number of epochs with learning rate scheduling

        Args:
            num_epochs: Number of training epochs
            filename: Path to save the best model (optional)

        Returns:
            model: Trained model
            history (Dict): History of training/validation losses
        """
        folder = self.training_info_path+"/"
        save_path = os.path.abspath(folder +filename) if filename else None

        # Create history if the model is not pretrained
        if self.history is None:
            self.history = {
                # Total losses
                'train_loss': [],
                'val_loss': [],

                # Component losses
                'train_edge_loss': [],
                'train_node_loss': [],
                'val_edge_loss': [],
                'val_node_loss': [],
                'learning_rate': []
            }

        # Track the time of training.
        start_time = time.time()

        # Create checkpoint directory
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        # Create a separate path for periodic checkpoints
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Begin training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_components = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_edge_loss'].append(train_components["edge_loss"])
            self.history['train_node_loss'].append(train_components["node_loss"])

            # Validation phase
            val_loss, val_components = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_edge_loss'].append(val_components["edge_loss"])
            self.history['val_node_loss'].append(val_components["node_loss"])

            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

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
                self.update_plot_learning_rate()

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

            # Save best model based on training loss
            if save_path and train_loss < self.best_train_loss and epoch % 10 == 0:
                self.best_train_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': self.history
                }, folder +"train_ " +filename)
                tsp =folder +"train_ " +filename


                print(f"New best model saved to {tsp} (train_loss: {train_loss:.4f})")

                if self.use_comet:
                    self.experiment.log_model("best_model_training", save_path)

            # Save best model based on validation loss
            if save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                print(f"New best model saved to {save_path} (val_loss: {val_loss:.4f})")

                if self.use_comet:
                    self.experiment.log_model("best_model_validation", save_path)

        # Final plot update
        if self.live_plot:
            self.update_plot()
            # self.create_final_plots()

            if self.use_comet and os.path.exists(f"{self.training_info_path}/training_history.png"):
                self.experiment.log_image(f"{self.training_info_path}/training_history.png", name="training_curves")

        # End experiment
        if self.use_comet:
            self.experiment.end()

        return self.model, self.history

