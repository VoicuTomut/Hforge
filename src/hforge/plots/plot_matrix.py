"""
example of model inferance
"""

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt







def plot_comparison_matrices(original_h, predicted_h, save_path=None, want_png_percent_diff=False):
    """
    Creates a simple 2x2 subplot with original matrix, predicted matrix,
    difference, and percentage difference.
    Percentage difference is capped at 100% (extreme values are clipped for better visualization), but higher values are physical.

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
    mask_small = np.abs(original_h) < epsilon # Boolean matrix for small values

    # Initialize percent_diff with zeros
    percent_diff = np.zeros_like(original_h, dtype=float)

    # Calculate percentage only where original is not too small
    #// np.divide(diff, np.abs(original_h)+0.1, out=percent_diff, where=~mask_small)
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
    # print(f"Difference - Min: {diff_min:.6f}, Max: {diff_max:.6f}")
    # print(f"Percentage Difference - Min: {percent_diff_min:.2f}%, Max: {percent_diff_max:.2f}% (capped at ±100%)")

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
    # Create a separate figure for the percentage difference
    fig_percent_diff = go.Figure()
    fig_percent_diff.add_trace(
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
        )
    )
    # Add the percentage difference trace to the main figure
    for trace in fig_percent_diff.data:
        fig.add_trace(trace, row=2, col=2)

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

    # Save the percent_diff figure separately if sepcified
    if want_png_percent_diff:
        fig_percent_diff.write_image(save_path.replace('.html', '_percent_diff.png'))

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
    # print("len(hop)=", len(hop))
    # print("len(onsite)=", len(onsite))
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
    edge_pairs = reduce_edge_index.clone().detach().reshape(-1, 2)

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

def plot_error_matrices(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, n_atoms=None, absolute_error_cbar_limit=None):
    """
    Saves as .png the plots of the true matrix, predicted matrix, absolute and relatives errors with additional stats.

    Args:
        true_matrix: The ground truth matrix
        predicted_matrix: The matrix you want to compare.
        matrix_label (str): The matrix type of the true and prediced matrices, to use in the title. (Optional)
        figure_title (str): Title of the figure. (Optional)
        predicted_matrix_text (str): Text under the predicted matrix. (Optional)
        filepath (str): Path to save the figure. (Optional)
        n_atoms (int): Number of atoms of the hamiltonian to plot a grid dividing different orbitals. (Optional)
    """

    # === Error matrices computation ===

    # Absolute error matrix
    absolute_error_matrix = true_matrix - predicted_matrix

    # Relative error matrix (in %): e_rel = e_abs / x
    epsilon = 0.001 # * Define your resolution here.
    relative_error_matrix = absolute_error_matrix / (true_matrix + epsilon)*100 # Sum epsilon to avoid divergences

    # // Optionally, clip extreme values to prevent overflow or underflow
    # // relative_error_matrix = np.clip(relative_error_matrix, -1e100, 1e100)

    # Compute the limits of the true and predicted matrices:
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    cbar_limits = [np.max([np.abs(vmin), np.abs(vmax)])for _ in range(2)] # Put the zero in the middle of the colorbar

    # Compute the limits of the absolute error matrix:
    if absolute_error_cbar_limit is None:
        vmin = np.min(absolute_error_matrix)
        vmax = np.max(absolute_error_matrix)
        cbar_limits.append(np.max([np.abs(vmin), np.abs(vmax)]))
    else:
        cbar_limits.append(absolute_error_cbar_limit)

    # Compute the limits of the relative error matrix:
    max_error = 100.0 # %
    cbar_limits.append(max_error)

    # === Set titles ===
    if matrix_label is None:
        matrix_label = ''
    titles = ["True " + matrix_label, "Predicted " + matrix_label, 'Absolute error (A-B)', f'Relative error (A-B)/(A+{epsilon})']
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Plots ===

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cmap_string = 'RdYlBu'

    # Plot the matrices:
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    for i, axis in enumerate(axes.flat):
        image = axis.imshow(matrices[i], cmap=cmap_string, vmin=-cbar_limits[i], vmax=cbar_limits[i])
        cbar = fig.colorbar(image, fraction=0.046, pad=0.04)
        cbar.ax.set_title(cbar_titles[i])
        axis.set_title(titles[i])

        axis.set_xlabel("Matrix element")
        axis.set_ylabel("Matrix element")
        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")

        # Matrix blocks divisor
        # TODO: Now is set to a ctt nº of orbitals, 13. We will need to allow flexible nº of orbitals, depending on the config file.
        if n_atoms is not None and n_atoms <= 8: # When n_atoms is too high, the grid makes the matrix ilegible.
            n_orbitals = 13
            minor_ticks = np.arange(-0.5, n_orbitals*n_atoms, n_orbitals)
            axes.flat[i].set_xticks(minor_ticks, minor=True)
            axes.flat[i].set_yticks(minor_ticks, minor=True)
            axes.flat[i].grid(which='minor', color='black', linestyle='-', linewidth=1)
            axes.flat[i].tick_params(which='minor', left=False, top=False)
    fig.tight_layout()

    # === Show predicted matrix text ===
    if predicted_matrix_text is not None:
        axes.flat[1].text(1.1, -0.05, predicted_matrix_text, ha='right', va='top', transform=axes.flat[1].transAxes, fontsize=12)

    # === Max and min absolute error ===
    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    axes.flat[2].text(0.5, -0.1, f'max = {max_absolute_error:.2f} eV,  min = {min_absolute_error:.2f} eV', ha='center', va='center', transform=axes.flat[2].transAxes, fontsize=12)

    # === Max and min relative error ===
    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    axes.flat[3].text(0.5, -0.1, f'max = {max_relative_error:.2f}%,  min = {min_relative_error:.2f}%', ha='center', va='center', transform=axes.flat[3].transAxes, fontsize=12)

    # === Title of the figure ===
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16)
        make_space_above(axes, topmargin=0.8)

    # === Save figure ===
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
    plt.close(fig)

def make_space_above(axes, topmargin=1):
    """Increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_error_matrices_interactive(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, n_atoms=None, absolute_error_cbar_limit=None):
    """Interactive Plotly visualization of error matrices."""

    # === Error matrices computation ===
    absolute_error_matrix = true_matrix - predicted_matrix
    epsilon = 0.001
    relative_error_matrix = absolute_error_matrix / (true_matrix + epsilon)*100

    # === Colorbar limits ===
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(abs(vmin), abs(vmax))

    if absolute_error_cbar_limit is None:
        lim_abs = np.max(np.abs(absolute_error_matrix))
    else:
        lim_abs = absolute_error_cbar_limit

    lim_rel = 100.0  # %

    cbar_limits = [lim_data, lim_data, lim_abs, lim_rel]

    # === Titles ===
    if matrix_label is None:
        matrix_label = ''
    titles = [
        "True " + matrix_label,
        "Predicted " + matrix_label,
        "Absolute error (A-B)",
        f"Relative error (A-B)/(A+{epsilon})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Figure ===
    cbar_positions = [0.44, 1, 0.44, 1]
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig = make_subplots(
        rows=2, cols=2,
        # subplot_titles=titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.17
    )

    for i, matrix in enumerate(matrices):
        row = i // 2 + 1
        col = i % 2 + 1

        heatmap = go.Heatmap(
            z=matrix,
            colorscale='RdYlBu',
            zmin=-cbar_limits[i],
            zmax=cbar_limits[i],
            colorbar=dict(title=cbar_titles[i], len=0.475, yanchor="middle", y=0.807 - 0.585*(row-1)),
            colorbar_x = cbar_positions[i]
        )
        fig.add_trace(heatmap, row=row, col=col)

    # === Subplot titles ===
    fig.update_layout(
        xaxis1=dict(side="top", title_text=titles[0]), yaxis1=dict(autorange="reversed"),
        xaxis2=dict(side="top", title_text=titles[1]), yaxis2=dict(autorange="reversed"),
        xaxis3=dict(side="top", title_text=titles[2]), yaxis3=dict(autorange="reversed"),
        xaxis4=dict(side="top", title_text=titles[3]), yaxis4=dict(autorange="reversed"),
        margin={"l":0,
                "r":0,
                "t":0,
                "b":0}
    )

    # === Atomic orbitals blocks grid ===
    if n_atoms is not None:
        n_orbitals = 13
        minor_ticks = np.arange(-0.5, n_orbitals * n_atoms, n_orbitals)

        for i, matrix in enumerate(matrices):
            row = i // 2 + 1
            col = i % 2 + 1  # Ensure shapes are added to the correct subplot

            grid_lines = [
                # Vertical grid lines
                dict(type="line", x0=x, x1=x, y0=-0.5, y1=n_orbitals * n_atoms - 0.5, line=dict(color="black", width=1))
                for x in minor_ticks
            ] + [
                # Horizontal grid lines
                dict(type="line", y0=y, y1=y, x0=-0.5, x1=n_orbitals * n_atoms - 0.5, line=dict(color="black", width=1))
                for y in minor_ticks
            ]

            # Add each grid line to the corresponding subplot
            for line in grid_lines:
                fig.add_shape(line, row=row, col=col)

    # === Text annotations ===

    # Text under predicted matrix
    if predicted_matrix_text:
        fig.add_annotation(
            text=predicted_matrix_text,
            xref='x2 domain', yref='y2 domain',
            x=1.1, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align='right'
        )

    # Absolute error stats
    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    fig.add_annotation(
        text=f"max = {max_absolute_error:.2f} eV, min = {min_absolute_error:.2f} eV",
        xref='x3 domain', yref='y3 domain',
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    # Relative error stats
    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    fig.add_annotation(
        text=f"max = {max_relative_error:.2f}%, min = {min_relative_error:.2f}%",
        xref='x4 domain', yref='y4 domain',
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    # === Layout of the whole figure ===
    fig.update_layout(
        height=850,
        width=800,
        title_text=figure_title if figure_title else "Matrix Comparison and Errors",
        title_x=0.5,
        title_y=0.99,
        margin=dict(t=100, b=20)
    )

    # === Output ===
    if filepath:
        fig.write_html(filepath)
    else:
        fig.show()
