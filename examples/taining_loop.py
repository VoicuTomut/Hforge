"""
example of training model
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from datasets import load_from_disk
from hforge.graph_dataset import graph_from_row
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell


def prepare_dataset(dataset_path, split_ratio=0.8, batch_size=4, cutoff=3.0):
    """
    Prepare dataset for training

    Args:
        dataset_path: Path to the dataset
        split_ratio: Train/validation split ratio
        batch_size: Batch size for dataloaders
        cutoff: Cutoff distance for graph construction

    Returns:
        train_loader, val_loader
    """
    # Load dataset
    dataset = load_from_disk(dataset_path)

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

    # Convert dataset to graph form
    graph_dataset = []
    for row in dataset:
        graph = graph_from_row(row, orbitals, cutoff=cutoff)
        graph_dataset.append(graph)

    # Split into train and validation
    split_idx = int(len(graph_dataset) * split_ratio)
    train_dataset = graph_dataset[:split_idx]
    val_dataset = graph_dataset[split_idx:]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def cost_function(pred_graph, target_graph):
    """
    Calculate loss between predicted and target Hamiltonian and overlap matrices.

    Args:
        pred_graph: Dictionary containing predicted edge_description (h_hop) and node_description (s_on_sites)
        target_graph: Dictionary containing target h_hop and s_on_sites values

    Returns:
        Total loss combining Hamiltonian and overlap matrix losses
    """
    # Extract predictions and targets
    edge_pred = pred_graph["edge_description"]
    node_pred = pred_graph["node_description"]

    edge_target = target_graph["edge_description"]
    node_target = target_graph["node_description"]

    # Compute MSE loss for both matrices
    edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_target)
    node_loss = torch.nn.functional.mse_loss(node_pred, node_target)

    # Combined loss (can add weights if needed)
    total_loss = edge_loss + node_loss

    return total_loss


## TRAINER ##
class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device


    def train_epoch(self):
        """Run one epoch of training"""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Prepare inputs
            x = batch.x.to(torch.float32)
            edge_index = batch.edge_index.to(torch.int64)

            # Check if edge_attr exists in the batch
            edge_attr = batch.edge_attr.to(torch.float32) if hasattr(batch, 'edge_attr') else None

            # Handle graph state (u) if it exists
            state = batch.u.to(torch.float32) if hasattr(batch, 'u') else None

            # Get batch indices if they exist
            node_batch = batch.batch if hasattr(batch, 'batch') else None
            bond_batch = batch.bond_batch if hasattr(batch, 'bond_batch') else None

            # Forward pass
            pred_graph = self.model(x, edge_index, edge_attr, state, node_batch, bond_batch)

            # Create target graph
            target_graph = {
                "edge_index": pred_graph["edge_index"],
                "edge_description": batch.h_hop,
                "node_description": batch.s_on_sites
            }

            # Calculate loss
            loss = self.loss_fn(pred_graph, target_graph)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Prepare inputs
                x = batch.x.to(torch.float32)
                edge_index = batch.edge_index.to(torch.int64)

                # Check if edge_attr exists in the batch
                edge_attr = batch.edge_attr.to(torch.float32) if hasattr(batch, 'edge_attr') else None

                # Handle graph state (u) if it exists
                state = batch.u.to(torch.float32) if hasattr(batch, 'u') else None

                # Get batch indices if they exist
                node_batch = batch.batch if hasattr(batch, 'batch') else None
                bond_batch = batch.bond_batch if hasattr(batch, 'bond_batch') else None

                # Forward pass
                pred_graph = self.model(x, edge_index, edge_attr, state, node_batch, bond_batch)

                # Create target graph
                target_graph = {
                    "edge_index": pred_graph["edge_index"],
                    "edge_description": batch.h_hop,
                    "node_description": batch.s_on_sites
                }

                # Calculate loss
                loss = self.loss_fn(pred_graph, target_graph)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs, save_path=None):
        """
        Train the model for specified number of epochs

        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model (optional)

        Returns:
            Trained model and history of training/validation losses
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            # Validation phase
            val_loss = self.validate()
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Save best model
            if save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        return self.model, history



def main():
    # Configuration
    dataset_path = "/Users/voicutomut/Documents/GitHub/Hforge/Data/aBN_HSX/nr_atoms_32"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    train_loader, val_loader = prepare_dataset(
        dataset_path=dataset_path,
        split_ratio=0.8,
        batch_size=4,
        cutoff=3.0
    )


    # Initialize model
    avg_num_neighbors = 8
    config_model={
        "embedding":{

            'hidden_irreps': "8x0e+8x1o",# 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1

            "r_max":3,
            "num_bessel":8,
            "num_polynomial_cutoff":6,
            "radial_type":"bessel",
            "distance_transform":None,
            "max_ell": 2,
            "num_elements":2,

        },
        "atomic_descriptors":{
            'hidden_irreps': "8x0e+8x1o", ## 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            'avg_num_neighbors':avg_num_neighbors , # need to be computed
            "radial_mlp" : [64, 64, 64],
            'num_interactions': 2,
            "correlation":3, # correlation order of the messages (body order - 1)
            "num_elements":2,
            "max_ell":2,
        },

        "edge_extraction":{
            "orbitals":orbitals,
            "hidden_dim_message_passing":300,
            "hidden_dim_matrix_extraction":200,

        },

        "node_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 300,
            "hidden_dim_matrix_extraction": 200,

        },

    }
    model=ModelShell(
        config_model
    )

    # Inference results
    output_graph = model(sample_graph)
    print(f"Output graph: {output_graph.keys()}")
    for key in output_graph.keys():
        print(f"{key}: {output_graph[key].shape}")
    print("__________")
    print(sample_graph)

    # Training loop

    # Define optimizer and learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=cost_function,
        optimizer=optimizer,
        device=device
    )

    # Train the model
    num_epochs = 50
    save_path = "best_model.pt"

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
    print("Inference with trained model:")
    print(f"Output graph keys: {output_graph.keys()}")
    for key in output_graph.keys():
        print(f"{key}: {output_graph[key].shape}")


if __name__ == "__main__":
    main()