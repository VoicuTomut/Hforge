"""
example of model inferance
"""


import torch
import os
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np

from datasets import load_from_disk
from hforge.graph_dataset import graph_from_row
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_comparison_matrices(original_h, predicted_h, save_path=None):
    """
    Creates a simple 2x2 subplot with original matrix, predicted matrix,
    difference, and percentage difference.

    Parameters:
    original_h (torch.Tensor): Original Hamiltonian matrix
    predicted_h (torch.Tensor): Predicted Hamiltonian matrix
    save_path (str, optional): Path to save the HTML output

    Returns:
    plotly.graph_objects.Figure: The plotly figure object with 4 subplots
    """
    # Convert to numpy if they're PyTorch tensors
    if isinstance(original_h, torch.Tensor):
        original_h = original_h.detach().cpu().numpy()
    if isinstance(predicted_h, torch.Tensor):
        predicted_h = predicted_h.detach().cpu().numpy()

    # Calculate differences
    diff = original_h - predicted_h
    diff_min = np.min(diff)
    diff_max = np.max(diff)

    # Calculate percentage difference with careful handling of small values
    epsilon = 1e-10  # Small value to prevent division by zero
    # Where original is close to zero, use a small value as denominator
    mask_small = np.abs(original_h) < epsilon

    # Initialize percent_diff with zeros
    percent_diff = np.zeros_like(original_h, dtype=float)

    # Calculate percentage only where original is not too small
    np.divide(diff, np.abs(original_h), out=percent_diff, where=~mask_small)
    percent_diff = percent_diff * 100  # Convert to percentage

    # For values where original is very small, set to either max or min
    # depending on whether diff is positive or negative
    if np.any(mask_small):
        max_percent = 100  # Cap at 100% difference for better visualization
        percent_diff[mask_small & (diff > 0)] = max_percent
        percent_diff[mask_small & (diff < 0)] = -max_percent
        percent_diff[mask_small & (diff == 0)] = 0

    # Clip extreme values for better visualization
    percent_diff = np.clip(percent_diff, -100, 100)

    percent_diff_min = np.min(percent_diff)
    percent_diff_max = np.max(percent_diff)

    # Print extrema for differences
    print(f"Difference - Min: {diff_min:.6f}, Max: {diff_max:.6f}")
    print(f"Percentage Difference - Min: {percent_diff_min:.2f}%, Max: {percent_diff_max:.2f}% (capped at ±100%)")

    # Determine max values for consistent scaling
    orig_abs_max = max(abs(np.max(original_h)), abs(np.min(original_h)))
    pred_abs_max = max(abs(np.max(predicted_h)), abs(np.min(predicted_h)))
    max_val = max(orig_abs_max, pred_abs_max)
    diff_abs_max = max(abs(diff_min), abs(diff_max))

    # Create row and column labels
    n_rows, n_cols = original_h.shape
    row_labels = list(range(n_rows))
    col_labels = list(range(n_cols))

    # Create simple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Original Hamiltonian",
            "Predicted Hamiltonian",
            f"Difference (Min: {diff_min:.2f}, Max: {diff_max:.2f})",
            f"Percentage Difference (Capped at ±100%)"
        ),
        vertical_spacing=0.2,  # Increased spacing
        horizontal_spacing=0.15  # Increased spacing
    )

    # Add original hamiltonian (top-left)
    fig.add_trace(
        go.Heatmap(
            z=original_h,
            x=col_labels,
            y=row_labels,
            colorscale='RdBu',
            zmin=-max_val,
            zmax=max_val,
            zmid=0,
            colorbar=dict(
                title="Value",
                thickness=15,
                len=0.4,
                y=0.8,  # Position in the middle of the top row
                x=0.85  # Position to the right of the first column
            )
        ),
        row=1, col=1
    )

    # Add predicted hamiltonian (top-right)
    fig.add_trace(
        go.Heatmap(
            z=predicted_h,
            x=col_labels,
            y=row_labels,
            colorscale='RdBu',
            zmin=-max_val,
            zmax=max_val,
            zmid=0,
            showscale=False  # No colorbar to avoid duplication
        ),
        row=1, col=2
    )

    # Add difference (bottom-left)
    fig.add_trace(
        go.Heatmap(
            z=diff,
            x=col_labels,
            y=row_labels,
            colorscale='RdBu',
            zmin=-diff_abs_max,
            zmax=diff_abs_max,
            zmid=0,
            colorbar=dict(
                title="Difference",
                thickness=15,
                len=0.4,
                y=0.2,  # Position in the middle of the bottom row
                x=0.85  # Position to the right of the first column
            )
        ),
        row=2, col=1
    )

    # Add percentage difference (bottom-right)
    fig.add_trace(
        go.Heatmap(
            z=percent_diff,
            x=col_labels,
            y=row_labels,
            colorscale='RdBu',
            zmin=-100,
            zmax=100,
            zmid=0,
            colorbar=dict(
                title="% Difference",
                thickness=15,
                len=0.4,
                y=0.2,  # Position in the middle of the bottom row
                x=1.05,  # Position to the right of the second column
                tickvals=[-100, -50, 0, 50, 100],
                ticktext=["-100%", "-50%", "0%", "50%", "100%"]
            )
        ),
        row=2, col=2
    )

    # Update axis labels for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Orbital Index", row=i, col=j)
            fig.update_yaxes(title_text="Orbital Index", row=i, col=j, autorange="reversed")

    # Set the overall layout
    fig.update_layout(
        title={
            'text': "Hamiltonian Matrix Comparison",
            'x': 0.5,
            'y': 0.98,
            'font': {'size': 24}
        },
        width=1400,
        height=1200,
        margin=dict(l=80, r=120, t=120, b=80),
        plot_bgcolor='rgba(240,240,240,0.95)',
        paper_bgcolor='rgba(250,250,250,0.95)',
        font=dict(size=14)
    )

    # Save to HTML if path provided
    if save_path:
        fig.write_html(save_path)

    return fig



def reconstruct_matrix(hop, onsite, reduce_edge_index):
    """
    Reconstructs a block matrix from hopping matrices, on-site energy matrices, and edge index coordinates.

    Parameters:
    hop (list): List of hopping matrices, each representing interaction between sites
    onsite (list): List of on-site energy matrices for the main diagonal blocks
    reduce_edge_index (list): List of edge indices representing connections between sites
                             Each pair of consecutive elements represents (row, column) coordinates

    Returns:
    torch.Tensor: The reconstructed block matrix
    """
    # Determine the number of sites
    num_sites = len(onsite)

    # Determine the size of each block from the first onsite matrix
    block_size = onsite[0].shape[0]

    # Initialize the full matrix with zeros
    full_matrix = torch.zeros((num_sites * block_size, num_sites * block_size), dtype=torch.float)

    # Fill the main diagonal blocks with on-site energy matrices
    for i in range(num_sites):
        row_start = i * block_size
        row_end = (i + 1) * block_size
        full_matrix[row_start:row_end, row_start:row_end] = onsite[i]

    # Convert edge_index to a tensor and reshape to pairs for easier handling
    edge_pairs = torch.tensor(reduce_edge_index).reshape(-1, 2)

    # Fill the off-diagonal blocks based on edge_index and hop matrices
    for idx, (site_i, site_j) in enumerate(edge_pairs):
        if idx < len(hop):
            # Calculate the block positions
            row_start_i = site_i * block_size
            row_end_i = (site_i + 1) * block_size
            row_start_j = site_j * block_size
            row_end_j = (site_j + 1) * block_size

            # Place the hopping matrix in the appropriate block
            full_matrix[row_start_i:row_end_i, row_start_j:row_end_j] = hop[idx]

            # For Hermitian matrices, also fill the symmetric block with the conjugate transpose
            full_matrix[row_start_j:row_end_j, row_start_i:row_end_i] = hop[idx].transpose(-1, -2).conj()

    return full_matrix

def load_best_model(model, optimizer=None, path="best_model.pt", device='cpu'):
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
    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return model, optimizer, 0, {}

    checkpoint = torch.load(path, map_location=device)

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
    dataset = load_from_disk("/Users/voicutomut/Documents/GitHub/Hforge/Data/aBN_HSX/nr_atoms_32")
    # features: ['nr_atoms', 'atomic_types_z', 'atomic_positions', 'lattice_nsc', 'lattice_origin',
    #            'lattice_vectors', 'boundary_condition', 'h_matrix', 's_matrix']
    print(dataset)
    # Playing_row
    row_index = 4
    sample = dataset[row_index]  # Replace 'train' with the correct split if applicable

    # Preproces the sample to graph form:
    orbitals={
        1:13,
        2:13,
        3:13,
        4:13,
        5:13,
        6:13,
        7:13,
        8:13,}
    sample_graph=graph_from_row(sample,orbitals, cutoff=4.0)
    print(sample_graph)


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



    model = ModelShell(config_model)

    # Load the saved weights
    model, _, _, _ = load_best_model(model, path="best_model.pt", device=device)

    # Set to evaluation mode
    model.eval()

    # Inference results
    print("__________")
    print(sample_graph)
    output_graph=model(sample_graph)
    print(f"Output graph: {output_graph.keys()}")
    for key in output_graph.keys():
        print(f"{key}: {output_graph[key].shape}")
    print("__________")
    print(sample_graph)


    original_h=reconstruct_matrix(output_graph["edge_description"],output_graph["node_description"], output_graph["edge_index"])
    predicted_h=reconstruct_matrix(sample_graph["h_hop"], sample_graph["h_on_sites"], output_graph["edge_index"])

    print("original_h:", original_h.shape)
    print("predicted_h:", predicted_h.shape)

    # Create the 4-panel comparison plot
    fig = plot_comparison_matrices(original_h, predicted_h, save_path="matrix_comparison.html")

    # Display the plot
    fig.show()
if __name__ == "__main__":
    main()