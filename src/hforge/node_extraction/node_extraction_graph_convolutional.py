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
        input_dim = edge_combined_dim + 2*node_dim

        # Node update networks. WEIGHTS MATRIX.
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_combined_dim, node_dim),
            nn.SiLU() #! Change?
        )

         # Edge update networks (focus on this as per requirements)
        self.edge_update = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), int(input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(input_dim/4), int(input_dim/8)),
            nn.ReLU(),
            nn.Linear(int(input_dim/8), int(edge_combined_dim/4)),
            nn.ReLU(),
            nn.Linear(int(edge_combined_dim/4), int(edge_combined_dim/2)),
            nn.ReLU(),
            nn.Linear(int(edge_combined_dim/2), edge_combined_dim)
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

        # === Edge feature update ===

        # Concatenate source, destination, and edge features
        edge_inputs = torch.cat([src_features, dst_features, edge_radial, edge_angular], dim=-1)

        #* Update edge features
        edge_features_updated = self.edge_update(edge_inputs)

        # === Node feature update - Graph convolutional layer. ===

        # Normalize by the number of neighbours
        messages = torch.zeros(num_nodes, node_features.shape[1] + edge_radial.shape[1] + edge_angular.shape[1], device=self.device)  # [num_nodes, node_dim + edge_combined_dim]
        for i in range(num_edges):
            source_node_idx = src[i] # [1]
            target_node_idx = dst[i] # [1]

            # Form the submessage of each node but normalization.
            submessage = torch.cat([src_features[source_node_idx], edge_radial[i], edge_angular[i]], dim=-1)  # [node_dim + edge_combined_dim], a vector.

            # Count the number of neighbours of this current (source) node.
            num_neighbours = torch.sum(edge_index[1] == source_node_idx).item()

            # Sum it to the complete message of the target node with proper normalization.
            messages[target_node_idx] += submessage / num_neighbours

        # "Update" node features (we are still in the aggregation function but we will just add the self submessage to the whole message).
        # Vector with all number of neighbours of each node
        num_neighbours_vector = torch.tensor([torch.sum(edge_index[1] == i) for i in range(num_nodes)], device=self.device) # [num_nodes]
        # Sum to all messages, but only the first node_dim elements (the rest are edge features).
        messages[:, :node_features.size(1)] += (node_features / torch.sqrt(num_neighbours_vector).unsqueeze(1))

        # Normalize by the number of neighbours of each node
        messages /= torch.sqrt(num_neighbours_vector).unsqueeze(1)

        #* Multiply by trainable weights + activation fn.
        node_features_updated = self.node_update(messages)  # [num_nodes, node_dim + edge_combined_dim]

        return node_features_updated, edge_features_updated


class NodeExtractionGraphConvolutional(nn.Module):
    def __init__(self, config_routine, device='cpu'):
        super(NodeExtractionGraphConvolutional, self).__init__()

        self.device = device

        # Get the number of layers
        self.mp_layers = config_routine.get("mp_layers", 1)  # Default to 1 if not specified

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