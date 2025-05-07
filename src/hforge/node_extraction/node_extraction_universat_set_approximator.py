import torch
import torch.nn as nn


class MatrixExtractionHead(nn.Module):
    """
    Neural network head that produces a matrix of size (orbitals_i × orbitals_j)
    for a specific atom pair type (i,j).
    """

    def __init__(self, input_dim, num_orbitals_i, num_orbitals_j, hidden_dim=128):
        super(MatrixExtractionHead, self).__init__()
        self.num_orbitals_i = num_orbitals_i
        self.num_orbitals_j = num_orbitals_j
        self.output_dim = num_orbitals_i * num_orbitals_j

        # Neural network to process the environment descriptor and edge embeddings
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Combined vector of node environment and edge features
        Returns:
            Matrix of shape (num_orbitals_i, num_orbitals_j)
        """
        # Process the input through MLP
        output = self.mlp(x)

        # Reshape to matrix form without the extra dimension
        matrix = output.view(self.num_orbitals_i, self.num_orbitals_j)

        return matrix

#TODO: Make it equivariant
class MessagePassing(nn.Module):
    """
    Message passing layer that updates node and edge features.
    Focuses on edge updates as specified in the requirements.
    """

    def __init__(self, node_dim, edge_radial_dim, edge_angular_dim, device='cpu'):
        super(MessagePassing, self).__init__()

        self.device = device

        # Combined edge dimension
        edge_combined_dim = edge_radial_dim + edge_angular_dim
        all_combined_dim = edge_combined_dim + node_dim

        # First MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(all_combined_dim, all_combined_dim),
            nn.Sigmoid(),
        )

        # Second MLP
        dims = torch.linspace(all_combined_dim, node_dim/4, 3, dtype=torch.int32, device=self.device)
        self.mlp2 = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], int(node_dim/2)),
            nn.ReLU(),
            nn.Linear(int(node_dim/2), node_dim),
        )

        self.node_linear_self = nn.Sequential(
            nn.Linear(node_dim, node_dim),
        )

        self.node_linear_neigh = nn.Sequential(
            nn.Linear(node_dim, node_dim),
        )

        self.node_activation_fn = nn.Sequential(
            nn.Sigmoid(),
        )

        # Edge update
        self.edge_update = nn.Sequential(
            nn.Linear(edge_combined_dim + 2*node_dim, edge_combined_dim + 2*node_dim),
            nn.Sigmoid(),
        )

    def forward(self, node_features, edge_radial, edge_angular, edge_index):
        """
        Args:
            node_features: Features for each node [num_nodes, node_dim]
            edge_radial: Radial features for each edge [num_edges, edge_radial_dim]
            edge_angular: Angular features for each edge [num_edges, edge_angular_dim]
            edge_index: Graph connectivity [2, num_edges]
        """
        # Get the number of nodes and edges
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)

        # Get source and target node indices for each edge
        src, dst = edge_index # [num_edges]

        # Get the features of the source and destination nodes
        src_features = node_features[src]  # [num_edges, node_dim]
        dst_features = node_features[dst]  # [num_edges, node_dim]

        # === Node feature update - Universal set approximator. ===

        messages = torch.zeros(num_nodes, node_features.shape[1] + edge_radial.shape[1] + edge_angular.shape[1], device=self.device)  # [num_nodes, node_dim + edge_combined_dim]
        for i in range(num_edges):
            source_node_idx = src[i] # [1]
            target_node_idx = dst[i] # [1]

            # Form the submessage of each node.
            submessage = torch.cat([src_features[source_node_idx], edge_radial[i], edge_angular[i]], dim=-1)  # [node_dim + edge_combined_dim], a vector.

            # Send the submessage through a MLP
            submessage = self.mlp1(submessage)

            # Sum it to the complete message of the target node with proper normalization.
            messages[target_node_idx] += submessage

        # Apply second MLP
        messages = self.mlp2(messages)

        # Update node features
        node_features_updated = self.node_activation_fn(self.node_linear_self(node_features) + self.node_linear_neigh(messages)) + node_features # Residual connection

        # === Edge feature update ===

        # Concatenate source, destination, and edge features
        edge_inputs = torch.cat([src_features, dst_features, edge_radial, edge_angular], dim=-1)

        # Update edge features
        edge_features_updated = self.edge_update(edge_inputs)

        return node_features_updated, edge_features_updated


class NodeExtractionUniversalApproximator(nn.Module):
    def __init__(self, config_routine, device='cpu'):
        super(NodeExtractionUniversalApproximator, self).__init__()

        self.device = device

        # Get the number of layers
        self.mp_layers = config_routine.get("mp_layers", 1)  # Default to 1 if not specified

        # Share parameters across layers or not
        self.share_parameters = config_routine.get("share_parameters", False)  # Default to False if not specified

        # Get the orbital information
        self.orbitals = config_routine["orbitals"]

        # Get the unique atom types
        self.atom_types = list(self.orbitals.keys())

        # Determine input dimension from the atomic environment descriptor
        self.input_dim = config_routine.get("descriptor_dim")

        self.num_orbitals_i = len(self.atom_types)

        # Get edge embedding dimensions for message passing
        self.edge_radial_dim = config_routine.get("edge_radial_dim")
        self.edge_angular_dim = config_routine.get("edge_angular_dim")
        self.edge_combined_dim = self.edge_radial_dim + self.edge_angular_dim

        # Check if parameters should be shared across layers
        if self.share_parameters:
            self.mp_layers = 1

        # Initialize the message passing layers
        self.message_passing_layers = [MessagePassing(
            node_dim=self.input_dim,
            edge_radial_dim=self.edge_radial_dim,
            edge_angular_dim=self.edge_angular_dim,
            device=self.device
        ) for _ in range(self.mp_layers)]

        # Create extraction heads for each atom type (for on-site matrices)
        self.extraction_heads = nn.ModuleDict()
        for atom in self.atom_types:
            # For on-site terms, we need a square matrix of size orbitals[atom] × orbitals[atom]
            num_orbitals = self.orbitals[atom]

            self.extraction_heads[str(atom)] = MatrixExtractionHead(
                input_dim=self.input_dim,  # Only need the node's own features
                num_orbitals_i=num_orbitals,
                num_orbitals_j=num_orbitals,
                hidden_dim=config_routine.get("hidden_dim_matrix_extraction", 128)
            )

    def forward(self, graph_data, embeddings, atomic_env_descriptor):
        """
        Args:
            graph_data: Graph data containing node and edge information
            embeddings: Node and edge embeddings
            atomic_env_descriptor: Environment descriptor for each atom
        Returns:
            Dictionary containing on-site matrices as a tensor
        """
        # Extract node features from the environment descriptor
        node_features = atomic_env_descriptor['nodes']['node_env']

        # Extract edge features from embeddings for message passing
        edge_radial = embeddings['edges']['radial_embedding']
        edge_angular = embeddings['edges']['angular_embedding']

        # Get edge connectivity
        edge_index = graph_data["reduce_edge_index"]


        # 1. Apply message passing to update node and edge features. Several layers of message passing.
        updated_node_features = node_features
        updated_edge_radial = edge_radial
        updated_edge_angular = edge_angular

        # If parameters are shared, use the first layer for all iterations
        # Otherwise, iterate through all layers
        if self.share_parameters:
            for i in range(self.mp_layers):
                updated_node_features, updated_edge_features = self.message_passing_layers[0](
                    updated_node_features, updated_edge_radial, updated_edge_angular, edge_index
                )
                # Split updated edge features back into radial and angular components
                updated_edge_radial = updated_edge_features[:, :self.edge_radial_dim]
                updated_edge_angular = updated_edge_features[:, self.edge_radial_dim:]
        else:
            for layer in self.message_passing_layers:
                updated_node_features, updated_edge_features = layer(
                    updated_node_features, updated_edge_radial, updated_edge_angular, edge_index
                )
                # Split updated edge features back into radial and angular components
                updated_edge_radial = updated_edge_features[:, :self.edge_radial_dim]
                updated_edge_angular = updated_edge_features[:, self.edge_radial_dim:]

        # 2. Generate on-site matrices for each atom
        on_sites = []

        # Iterate through all nodes in the graph
        for i, atom_idx in enumerate(graph_data.x):
            # Get the atom type for this node
            atom_type = self.atom_types[int(atom_idx.item())]

            # Use the extraction head for this atom type
            extraction_head = self.extraction_heads[str(atom_type)]

            # Generate the on-site matrix using only the node's updated features
            on_site_matrix = extraction_head(updated_node_features[i])
            on_sites.append(on_site_matrix)

        # Convert list of on-site matrices to a single tensor
        on_site_tensor = torch.stack(on_sites) if on_sites else torch.tensor([])


        return on_site_tensor