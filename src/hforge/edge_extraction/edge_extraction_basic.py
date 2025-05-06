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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
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
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.LeakyReLU(),
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

        #* Update edge features
        edge_features_updated = self.edge_update(edge_inputs) + edge_features  # Residual connection

        # 2. Then update nodes using aggregated edge information
        # Initialize updated node features
        node_features_updated = torch.zeros_like(node_features)

        # Aggregate messages for each node
        for i in range(num_edges):
            target_node = dst[i]

            # Aggregate edge features as messages to the target node
            message = torch.cat([node_features[target_node], edge_features_updated[i]], dim=-1)

            #* Update node features
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


class EdgeExtractionBasic(nn.Module):
    def __init__(self, config_routine, device='cpu'):
        super(EdgeExtractionBasic, self).__init__()

        # Get the number of layers
        self.n_layers = config_routine.get("n_layers", 1)  # Default to 1 if not specified

        # Get the orbital information
        self.orbitals = config_routine["orbitals"]

        # Get the unique atom types
        self.atom_types = list(self.orbitals.keys())

        # Determine input dimension from the atomic environment descriptor
        # This should match the dimension you're using in your model
        self.input_dim = config_routine.get("descriptor_dim", 64)

        # Get edge embedding dimensions from the embeddings or config
        self.edge_radial_dim = config_routine.get("edge_radial_dim", 8)
        self.edge_angular_dim = config_routine.get("edge_angular_dim", 9)
        self.edge_combined_dim = self.edge_radial_dim + self.edge_angular_dim

        # Initialize the message passing layer
        self.message_passing = MessagePassing(
            node_dim=self.input_dim,
            edge_radial_dim=self.edge_radial_dim,
            edge_angular_dim=self.edge_angular_dim,
            hidden_dim=config_routine.get("hidden_dim_message_passing", 128)
        )

        # Create extraction heads for each atom-atom pair
        self.extraction_heads = nn.ModuleDict()
        for atom1 in self.atom_types:
            self.extraction_heads[str(atom1)] = nn.ModuleDict()
            for atom2 in self.atom_types:
                # For each atom pair (i,j), we need a head that produces an orbitals_i × orbitals_j matrix

                head_input_dim = 2 * self.input_dim + self.edge_combined_dim  # 2 nodes + edge features
                num_orbitals_i = self.orbitals[atom1]
                num_orbitals_j = self.orbitals[atom2]

                self.extraction_heads[str(atom1)][str(atom2)] = MatrixExtractionHead(
                    input_dim=head_input_dim,
                    num_orbitals_i=num_orbitals_i,
                    num_orbitals_j=num_orbitals_j,
                    hidden_dim=config_routine.get("hidden_dim_matrix_extraction", 128)
                )

    def forward(self, graph_data, embeddings, atomic_env_descriptor):
        """
        Args:
            graph_data: Graph data containing node and edge information
            embeddings: Node and edge embeddings
            atomic_env_descriptor: Environment descriptor for each atom
        Returns:
            Dictionary containing hopping matrices as a tensor
        """
        # Extract node features from the environment descriptor
        node_features = atomic_env_descriptor['nodes']['node_env']

        # Extract edge features from embeddings
        edge_radial = embeddings['edges']['radial_embedding']
        edge_angular = embeddings['edges']['angular_embedding']



        # Get edge connectivity
        edge_index = graph_data["reduce_edge_index"]


        # 1. Apply equivariant message passing to update node and edge features.
        # Several layers of message passing.
        for i in range(self.n_layers):
            updated_node_features, updated_edge_features = self.message_passing(
                node_features, edge_radial, edge_angular, edge_index
            )

        # Split updated edge features back into radial and angular components
        updated_edge_radial = updated_edge_features[:, :self.edge_radial_dim]
        updated_edge_angular = updated_edge_features[:, self.edge_radial_dim:]

        # 2. Generate hopping matrices for each edge
        hoppings = []

        # Process each edge in the graph
        for edge_idx in range(edge_index.size(1)):
            # Get the source and target nodes
            src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]

            # Get the atom types
            src_atom_type = self.atom_types[int(graph_data.x[src].item())]
            dst_atom_type = self.atom_types[int(graph_data.x[dst].item())]

            # Concatenate source node, destination node, and edge features
            edge_input = torch.cat([
                updated_node_features[src],
                updated_node_features[dst],
                updated_edge_radial[edge_idx],
                updated_edge_angular[edge_idx]
            ], dim=-1)

            # Use the appropriate extraction head for this atom pair
            hopping_head = self.extraction_heads[str(src_atom_type)][str(dst_atom_type)]

            # Generate the hopping matrix
            hopping_matrix = hopping_head(edge_input)
            hoppings.append(hopping_matrix)

        # Convert list of hopping matrices to a single tensor
        # This stacks all matrices along a new first dimension
        hopping_tensor = torch.stack(hoppings) if hoppings else torch.tensor([])

        return  hopping_tensor