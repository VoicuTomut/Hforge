import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

#TODO: Add shifts
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
        self.radial_mlp.append(nn.Linear(hidden_dim, 1))

        # MLP for angular embeddings
        self.angular_mlp = nn.ModuleList()
        self.angular_mlp.append(nn.Linear(edge_dim_angular, hidden_dim))
        for _ in range(num_layers - 2):
            self.angular_mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.angular_mlp.append(nn.Linear(hidden_dim, 1))

    def forward(self, radial_embedding, angular_embedding, edge_index):
        """
        Forward pass through the EdgeAggregator.

        Args:
            radial_embedding (torch.Tensor): Radial embeddings of shape [num_edges, edge_dim_radial]
            angular_embedding (torch.Tensor): Angular embeddings of shape [num_edges, edge_dim_angular]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]

        Returns:
            tuple: Aggregated radial and angular features per node, and clean edge index without duplicates
        """
        # Get clean edge index without duplicates (for cases where you need it)
        # This ensures we consider source->target pairs as unique edges
        edge_tuples = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])]
        unique_tuples = list(set(edge_tuples))
        clean_src = [e[0] for e in unique_tuples]
        clean_dst = [e[1] for e in unique_tuples]
        clean_edge_index = torch.tensor([clean_src, clean_dst], dtype=edge_index.dtype, device=edge_index.device)

        # Process edges using message passing (still using original edge_index for proper embedding matching)
        radial_out = self.propagate(edge_index,
                                    x=None,
                                    embedding=radial_embedding,
                                    mlp=self.radial_mlp)

        angular_out = self.propagate(edge_index,
                                     x=None,
                                     embedding=angular_embedding,
                                     mlp=self.angular_mlp)

        return radial_out, angular_out, clean_edge_index

    def message(self, embedding, mlp):
        """
        Process embeddings through MLP to compute weights and weighted embeddings.

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

    def update(self, aggr_out, x=None):
        """
        Update node embeddings.

        Args:
            aggr_out (torch.Tensor): Aggregated features for each node after message passing
            x (torch.Tensor): Original node features (if any)

        Returns:
            torch.Tensor: Updated node embeddings
        """
        # Just return the aggregated output
        return aggr_out


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
    print(f"  Aggregated radial features: {radial_out.shape}")
    print(f"  Aggregated angular features: {angular_out.shape}")
    print(f"  Clean edge index: {clean_edge_index.tolist()}")

    # The expected shapes would be [max_node_idx + 1, edge_dim_*]
    # Here, nodes are [0,1,2], so we expect 3 nodes in the output


if __name__ == "__main__":
    example()