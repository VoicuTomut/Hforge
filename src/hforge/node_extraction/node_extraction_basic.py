import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
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

    def __init__(self, node_dim, edge_radial_dim, edge_angular_dim, hidden_dim=128):
        super(MessagePassing, self).__init__()

        # Combined edge dimension
        edge_combined_dim = edge_radial_dim + edge_angular_dim

        # Node update networks
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update networks (focus on this as per requirements)
        self.edge_update = nn.Sequential(
            nn.Linear(edge_combined_dim + 2 * node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_combined_dim)
        )

    def forward(self, node_features, edge_radial, edge_angular, edge_index):
        """
        Args:
            node_features: Features for each node [num_nodes, node_dim]
            edge_radial: Radial features for each edge [num_edges, edge_radial_dim]
            edge_angular: Angular features for each edge [num_edges, edge_angular_dim]
            edge_index: Graph connectivity [2, num_edges]
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)

        # Combine edge features

        edge_features = torch.cat([edge_radial, edge_angular], dim=-1)

        # Get source and target node indices for each edge
        src, dst = edge_index

        # 1. First update edges using source and destination node features
        src_features = node_features[src]
        dst_features = node_features[dst]

        # Concatenate source, destination, and edge features
        edge_inputs = torch.cat([src_features, dst_features, edge_features], dim=-1)


        # Update edge features
        edge_features_updated = self.edge_update(edge_inputs) + edge_features  # Residual connection

        # 2. Then update nodes using aggregated edge information
        # Initialize updated node features
        node_features_updated = torch.zeros_like(node_features)

        # Aggregate messages for each node
        for i in range(num_edges):
            target_node = dst[i]

            # Aggregate edge features as messages to the target node
            message = torch.cat([node_features[target_node], edge_features_updated[i]], dim=-1)

            # Update node features
            node_update_i = self.node_update(message)

            # Add to the target node (will be normalized later)
            node_features_updated[target_node] += node_update_i

        # Normalize by the number of received messages
        # Count the number of incoming edges for each node
        in_degrees = torch.zeros(num_nodes, device=node_features.device)
        for i in range(num_edges):
            in_degrees[dst[i]] += 1

        # Avoid division by zero
        in_degrees = torch.clamp(in_degrees, min=1)

        # Normalize
        for i in range(num_nodes):
            node_features_updated[i] /= in_degrees[i]

        # Add residual connection
        node_features_updated = node_features_updated + node_features

        return node_features_updated, edge_features_updated


class NodeExtractionBasic(nn.Module):
    def __init__(self, config_routine, device="cpu"):
        super(NodeExtractionBasic, self).__init__()

        # Get the number of layers
        self.n_layers = config_routine.get("n_layers", 1)  # Default to 1 if not specified

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



        # Initialize the message passing layer to update node features
        self.message_passing = MessagePassing(
            node_dim=self.input_dim,
            edge_radial_dim=self.edge_radial_dim,
            edge_angular_dim=self.edge_angular_dim,
            hidden_dim=config_routine.get("hidden_dim_message_passing", 128)
        )

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

        # 1. Apply equivariant message passing to update node features based on their environment
        for i in range(self.n_layers):
            updated_node_features, _ = self.message_passing(
                node_features, edge_radial, edge_angular, edge_index
            )

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