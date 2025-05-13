"""
example of model inferance
"""


import torch
import os

from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell
from hforge.plots.plot_matrix import plot_comparison_matrices, reconstruct_matrix

from examples.training_loop.training_loop import prepare_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def load_best_model(model, optimizer=None, path="./EXAMPLE_info/best_model.pt", device='cpu'):
    """
    Load the best model checkpoint from the saved file

    Args:
        model: Model instance to load the weights into
        optimizer: Optional optimizer to load state (for continued training)
        path: Path to the saved model checkpoint
        device: Device to load the model to ('cpu' or 'cuda')

    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (if provided)
        epoch: The epoch at which the model was saved
        history: Training history dictionary
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return model, optimizer, 0, {}

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    epoch = checkpoint['epoch']
    history = checkpoint.get('history', {})

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from {path} (saved at epoch {epoch + 1})")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model, optimizer, epoch, history


def main():
    # Load the dataset and extract a toy sample
    # dataset = load_from_disk("./Data/aBN_HSX/nr_atoms_3")
    # features: ['nr_atoms', 'atomic_types_z', 'atomic_positions', 'lattice_nsc', 'lattice_origin',
    #            'lattice_vectors', 'boundary_condition', 'h_matrix', 's_matrix']

    dataset_path="./Data/aBN_HSX/nr_atoms_3"
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
    dataset, _ = prepare_dataset(
        dataset_path=dataset_path,
        orbitals=orbitals,
        split_ratio=0.85,  # Slight increase in training data
        batch_size=1,  # Increased batch size for better gradient estimates
        cutoff=3.0,
        max_samples=None  # Use full dataset for better training
    )
    print(dataset)
    # Playing_row
    dataset = list(iter(dataset))
    row_index=10
    for row_index in range(20):

        sample_graph = dataset[row_index]  # Replace 'train' with the correct split if applicable



        # Initialize model
        avg_num_neighbors = 80
        nr_bits=10
        config_model = {
            "embedding": {

                'hidden_irreps': "8x0e+8x1o",
                # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1

                "r_max": 3,
                "num_bessel": 8,
                "num_polynomial_cutoff": 6,
                "radial_type": "bessel",
                "distance_transform": None,
                "max_ell": 3,
                "num_elements": nr_bits,
                "orbitals": orbitals,
                "nr_bits": nr_bits

            },
            "atomic_descriptors": {
                'hidden_irreps': "8x0e+8x1o",
                ## 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
                "interaction_cls_first": RealAgnosticResidualInteractionBlock,
                "interaction_cls": RealAgnosticResidualInteractionBlock,
                'avg_num_neighbors': avg_num_neighbors,  # need to be computed
                "radial_mlp": [64, 64, 64],
                'num_interactions': 2,
                "correlation": 3,  # correlation order of the messages (body order - 1)
                "num_elements": nr_bits,
                "max_ell": 3,
            },

            "edge_extraction": {
                "orbitals": orbitals,
                "hidden_dim_message_passing": 900,
                "hidden_dim_matrix_extraction": 900,

            },

            "node_extraction": {
                "orbitals": orbitals,
                "hidden_dim_message_passing": 900,
                "hidden_dim_matrix_extraction": 900,

            },

        }



        model = ModelShell(config_model)

        # Load the saved weights
        model, _, _, _ = load_best_model(model, path="train_best_model.pt", device=device)

        # Set to evaluation mode
        model.eval()

        # Inference results
        print("__________")
        print(sample_graph)
        print("inference:", sample_graph)
        print("X:",sample_graph.x)
        output_graph=model(sample_graph)
        print(f"Output graph: {output_graph.keys()}")
        for key in output_graph.keys():
            print(f"{key}: {output_graph[key].shape}")
        print("__________")
        print(sample_graph)


        predicted_h=reconstruct_matrix(output_graph["edge_description"],output_graph["node_description"], output_graph["edge_index"])
        original_h=reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])


        edge_loss = torch.sum(torch.abs(sample_graph["h_hop"] - output_graph["edge_description"]) / (torch.abs(sample_graph["h_hop"]) + 0.1))
        node_loss=torch.sum(torch.abs(sample_graph["h_on_sites"] - output_graph["node_description"]) / (torch.abs(sample_graph["h_on_sites"]) + 0.1))


        h_loss = torch.sum(torch.abs(original_h - predicted_h) / (torch.abs(original_h) + 0.1))
        print("h_loss:", h_loss)
        print("edge_loss:", edge_loss)
        print("node_loss:", node_loss)


        # Create the 4-panel comparison plots
        fig = plot_comparison_matrices(original_h*100, predicted_h, save_path="./EXAMPLE_info/matrix_comparison_New_{i}.html")

        # Display the plots
        fig.show()
if __name__ == "__main__":
    main()