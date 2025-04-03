"""
example of model inferance
"""


import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np







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
    np.divide(diff, np.abs(original_h)+0.1, out=percent_diff, where=~mask_small)
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
                y=0.80,  # Position in the middle of the top row
                x=0.425  # Position to the right of the first column
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
            colorbar=dict(
                title="Value",
                thickness=15,
                len=0.4,
                y=0.80,  # Position in the middle of the top row
                x=1.0  # Position to the right of the first column
            )
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
                y=0.20,  # Position in the middle of the bottom row
                x=0.425  # Position to the right of the first column
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
                y=0.20,  # Position in the middle of the bottom row
                x=1.0,  # Position to the right of the second column
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
