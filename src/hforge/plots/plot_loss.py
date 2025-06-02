import matplotlib.pyplot as plt
import os

def plot_loss_from_history(history, directory_to_save_in, start_from=None, end_in=None, start_from_last_epochs=None, plot_validation=True, tail=None):
    """ Creates a .png with the plot of the history of loss.

    Args:
        history (Dict): The loss history of the model
        directory_to_save_in: Directory where the model is saved
        start_from (int, Optional): The first epoch to start the plot from. Default: None (first epoch)
        end_in (int, Optional): The last epoch to stop the plot from. Default: None (last epoch)
        start_from_last_epochs (int, Optional): Use if want to plot only last specified number of epochs. Default: None
        plot_validation (bool, Optional): Also plot the validation loss. Default: True
        tail (str, Optional): Additional text to add at the end of the filename. Default: None
    """

    # === Previous checks ===
    # Check correctness of inputs
    if start_from is not None and start_from_last_epochs is not None:
        raise ValueError("start_from and start_from_last_epochs cannot be both specified.")

    # Check consistency of validation and training x axes
    if len(history["train_loss"][start_from:end_in]) != len(history["val_loss"][start_from:end_in]):
        raise ValueError("The length of train_loss and val_loss do not coincide.")

    # Plot only the last epochs if specified
    last_epoch = len(history["train_loss"])
    if start_from_last_epochs is not None:
        start_from = last_epoch - start_from_last_epochs

    # Get all the epochs
    epochs = range(len(history["train_loss"]))[start_from:end_in]

    # === Subplots ===
    plt.figure(figsize=(12, 8))

    # Learning rate
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history['learning_rate'][start_from:end_in], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')

    # Total loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history['train_loss'][start_from:end_in], 'b-', label='Training Loss')
    if plot_validation:
        plt.plot(epochs, history['val_loss'][start_from:end_in], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Components
    # Train loss
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history['train_edge_loss'][start_from:end_in], 'b-', label='Train Edge Loss')
    if plot_validation:
        plt.plot(epochs, history['val_edge_loss'][start_from:end_in], 'r-', label='Val Edge Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Edge Loss')
    plt.title('Edge Matrix Loss')
    plt.legend()
    plt.grid(True)

    # Validation loss
    plt.subplot(3, 2, 6)
    plt.plot(epochs, history['train_node_loss'][start_from:end_in], 'b-', label='Train Node Loss')
    if plot_validation:
        plt.plot(epochs, history['val_node_loss'][start_from:end_in], 'r-', label='Val Node Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Node Loss')
    plt.title('Node Matrix Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # === Save figure ===
    # If plotted only last epochs
    if start_from_last_epochs is not None:
        filename = f'plot_last_{start_from_last_epochs}epochs.png'

    # If did not start from 0
    elif start_from is not None:
        filename = f'plot_training_history_startfrom_{start_from}epochs.png'

    # Default
    else:
        # Important to use the same name in all epochs to enable overwritting
        filename = f'plot_training_history.png'

    # Add the tail to the name
    if tail is not None:
        filename = filename.replace(".png", f"_{tail}.png")

    # Save the figure in the specified directory
    plt.savefig(f'{directory_to_save_in}/{filename}')
    print(f"Plot saved as {directory_to_save_in}/{filename}")

    plt.close()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_loss_from_history_interactive(
    history,
    directory_to_save_in,
    start_from=None,
    end_in=None,
    start_from_last_epochs=None,
    plot_validation=True,
    tail=None
):
    """
    Creates an HTML file with interactive Plotly plots of training history.

    Args:
        history (Dict): The loss history of the model.
        directory_to_save_in (str): Directory where the plot should be saved.
        start_from (int, Optional): First epoch to plot from.
        end_in (int, Optional): Last epoch to plot to (non-inclusive).
        start_from_last_epochs (int, Optional): Plot only the last specified number of epochs.
        plot_validation (bool, Optional): Whether to include validation curves.
        tail (str, Optional): Extra string for output file naming.
    """
    if start_from is not None and start_from_last_epochs is not None:
        raise ValueError("start_from and start_from_last_epochs cannot be both specified.")

    last_epoch = len(history["train_loss"])
    if start_from_last_epochs is not None:
        start_from = last_epoch - start_from_last_epochs

    start_from = start_from or 0
    end_in = end_in or last_epoch

    epochs = list(range(start_from, end_in))

    if plot_validation and len(history["train_loss"][start_from:end_in]) != len(history["val_loss"][start_from:end_in]):
        raise ValueError("The length of train_loss and val_loss do not coincide.")

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"colspan": 2}, None],
               [{"colspan": 2}, None],
               [{}, {}]],
        subplot_titles=("Learning Rate", "Training and Validation Loss",
                        "Edge Matrix Loss", "Node Matrix Loss")
    )

    # Learning Rate
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["learning_rate"][start_from:end_in],
        mode='lines',
        name="Learning Rate",
        line=dict(color="green")
    ), row=1, col=1)

    # Training and Validation Loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["train_loss"][start_from:end_in],
        mode='lines',
        name="Training Loss",
        line=dict(color="blue")
    ), row=2, col=1)

    if plot_validation:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history["val_loss"][start_from:end_in],
            mode='lines',
            name="Validation Loss",
            line=dict(color="red")
        ), row=2, col=1)

    # Edge Matrix Loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["train_edge_loss"][start_from:end_in],
        mode='lines',
        name="Train Edge Loss",
        line=dict(color="blue")
    ), row=3, col=1)

    if plot_validation:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history["val_edge_loss"][start_from:end_in],
            mode='lines',
            name="Val Edge Loss",
            line=dict(color="red")
        ), row=3, col=1)

    # Node Matrix Loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["train_node_loss"][start_from:end_in],
        mode='lines',
        name="Train Node Loss",
        line=dict(color="blue")
    ), row=3, col=2)

    if plot_validation:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history["val_node_loss"][start_from:end_in],
            mode='lines',
            name="Val Node Loss",
            line=dict(color="red")
        ), row=3, col=2)

    fig.update_layout(
        height=900,
        width=1200,
        title_text="Training History",
        showlegend=True
    )

    # Y-axis log scale for Learning Rate subplot
    fig.update_yaxes(type="log", row=1, col=1)

    # === Save the figure ===
    filename = (
        f"plot_last_{start_from_last_epochs}epochs.html" if start_from_last_epochs is not None else
        f"plot_training_history_startfrom_{start_from}epochs.html" if start_from > 0 else
        "plot_training_history.html"
    )

    if tail:
        filename = filename.replace(".html", f"_{tail}.html")

    filepath = os.path.join(directory_to_save_in, filename)
    fig.write_html(filepath)
    print(f"Interactive plot saved to: {filepath}")
