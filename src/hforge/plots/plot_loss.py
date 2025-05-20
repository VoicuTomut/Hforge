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
        filename = f'training_history_startfrom_{start_from}epochs.png'

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