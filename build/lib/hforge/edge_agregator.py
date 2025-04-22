import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import coalesce


class EdgeAggregator(MessagePassing):
    def __init__(self, edge_dim_radial, edge_dim_angular, hidden_dim, num_layers=3):
        """
        EdgeAggregator for processing radial and angular embeddings through message passing.

        Args:
            edge_dim_radial (int): Dimension of radial edge embeddings
            edge_dim_angular (int): Dimension of angular edge embeddings
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of layers in the MLP
        """
        super().__init__(aggr='add')  # Aggregate edge features using summation

        # MLP for radial embeddings
        self.radial_mlp = nn.ModuleList()
        self.radial_mlp.append(nn.Linear(edge_dim_radial, hidden_dim))
        for _ in range(num_layers - 2):
            self.radial_mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.radial_mlp.append(nn.Linear(hidden_dim, edge_dim_radial))  # Output same dimension as input

        # MLP for angular embeddings
        self.angular_mlp = nn.ModuleList()
        self.angular_mlp.append(nn.Linear(edge_dim_angular, hidden_dim))
        for _ in range(num_layers - 2):
            self.angular_mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.angular_mlp.append(nn.Linear(hidden_dim, edge_dim_angular))  # Output same dimension as input

    def forward(self, radial_embedding, angular_embedding, edge_index):
        """
        Forward pass through the EdgeAggregator.

        Args:
            radial_embedding (torch.Tensor): Radial embeddings of shape [num_edges, edge_dim_radial]
            angular_embedding (torch.Tensor): Angular embeddings of shape [num_edges, edge_dim_angular]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]

        Returns:
            tuple: Processed radial and angular embeddings, and coalesced edge index
        """
        # Process edge embeddings directly with MLPs (no propagation yet)
        radial_out = self._process_embedding(radial_embedding, self.radial_mlp)
        angular_out = self._process_embedding(angular_embedding, self.angular_mlp)

        # Get number of nodes to use for coalescing
        num_nodes = edge_index.max().item() + 1

        # Use coalesce to directly aggregate the embeddings
        clean_edge_index, radial_clean = coalesce(
            edge_index=edge_index,
            edge_attr=radial_out,
            num_nodes=num_nodes,
            reduce="mean"  # Use mean to average duplicate edges
        )

        # Do the same for angular embeddings
        _, angular_clean = coalesce(
            edge_index=edge_index,
            edge_attr=angular_out,
            num_nodes=num_nodes,
            reduce="mean"  # Use mean to average duplicate edges
        )

        return radial_clean, angular_clean, clean_edge_index

    def _process_embedding(self, embedding, mlp):
        """
        Process embeddings through MLP.

        Args:
            embedding (torch.Tensor): Edge embeddings
            mlp (nn.ModuleList): MLP to process the embeddings

        Returns:
            torch.Tensor: Processed embeddings
        """
        x = embedding
        for i, layer in enumerate(mlp):
            x = layer(x)
            if i < len(mlp) - 1:  # Apply non-linearity except for the last layer
                x = F.relu(x)
        return x

    def message(self, embedding, mlp):
        """
        Process embeddings through MLP to compute weights and weighted embeddings.
        This is used during propagation.

        Args:
            embedding (torch.Tensor): Edge embeddings (radial or angular)
            mlp (nn.ModuleList): MLP to process the embeddings

        Returns:
            torch.Tensor: Weighted embeddings
        """
        # Apply MLP: linear -> nonlinear -> linear
        x = embedding
        for i, layer in enumerate(mlp):
            x = layer(x)
            if i < len(mlp) - 1:  # Apply non-linearity except for the last layer
                x = F.relu(x)

        # Output weight scalar for each edge
        weight = x  # Shape: [num_edges, 1]

        # Return weighted embedding
        return weight * embedding


# Example usage
def example():
    import torch
    from torch_geometric.data import Data

    # Example inputs
    radial_embedding = torch.randn(4, 8)  # 4 edges, 8-dimensional radial embedding
    angular_embedding = torch.randn(4, 9)  # 4 edges, 9-dimensional angular embedding
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 1]])  # Edge connections

    # Create the model
    model = EdgeAggregator(
        edge_dim_radial=8,
        edge_dim_angular=9,
        hidden_dim=16,
        num_layers=3
    )

    # Forward pass
    radial_out, angular_out, clean_edge_index = model(radial_embedding, angular_embedding, edge_index)

    print(f"Input shapes:")
    print(f"  Radial embedding: {radial_embedding.shape}")
    print(f"  Angular embedding: {angular_embedding.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Original edge index: {edge_index.tolist()}")

    print(f"\nOutput shapes:")
    print(f"  Processed radial features: {radial_out.shape}")
    print(f"  Processed angular features: {angular_out.shape}")
    print(f"  Clean edge index: {clean_edge_index.tolist()}")


if __name__ == "__main__":
    example()