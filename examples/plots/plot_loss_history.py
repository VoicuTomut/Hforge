from hforge.model_shell import ModelShell
import yaml
from hforge.utils import load_model, load_model_and_dataset_from_directory
import os
from hforge.utils import create_directory, get_object_from_module
from hforge.plots import plot_loss_from_history, plot_loss_from_history_interactive


# TODO: Write in the title the nr of atoms, if it's train or val sample and the minimum loss.

def main():
    # === Device setup ===
    device = "cpu"

    # === Load configuration ===
    # * Write the parent directory where all the models are located
    directory = r"C:\Users\angel\Documents\GitHub\Hforge\example_results"
    plots_folder_name = "plot_loss_history"
    plots_directory = os.path.join(directory, plots_folder_name)
    create_directory(plots_directory)

    for path, folders, files in os.walk(directory):
        for folder in folders:
            # Avoid plots folder
            if folder == plots_folder_name:
                continue

            # Get the model directory
            model_directory = os.path.join(directory, folder)

            # Load the history
            _, history, _ = load_model_and_dataset_from_directory(model_directory, "train_best_model.pt")

            # === Plots ===
            # Plot the last epochs
            plot_loss_from_history(history, model_directory, start_from_last_epochs=100, plot_validation=True, tail=folder)
            plot_loss_from_history(history, plots_directory, start_from_last_epochs=100, plot_validation=True, tail=folder)

            # Plot the full history
            plot_loss_from_history(history, model_directory, start_from=5, tail=folder)
            plot_loss_from_history(history, plots_directory, start_from=5, tail=folder)

            # Plotly
            plot_loss_from_history_interactive(history, model_directory, tail=folder)

        # Prevent from looking inside the folders
        break

if __name__ == "__main__":
    main()