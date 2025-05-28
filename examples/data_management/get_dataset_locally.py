"""
Download ehe aBN dataset.
"""
import os

from datasets import load_dataset

from hforge.data_management.data_processing import preprocess_and_save_graphs
from hforge.data_management.dataset_load import read_datasets_list_from_parent_dir
from hforge.utils import load_config


def main():
    # Load the dataset
    dataset = load_dataset("AndreiVoicuT/aBN_HSX")
    print(dataset)

    # Save the dataset to disk
    parent_dir = "Data/aBN_HSX"
    dataset.save_to_disk(parent_dir)

    # Convert it to graphs
    config = load_config()
    orbitals = config["orbitals"]
    dataset_config = config["dataset"]
    cutoff=dataset_config["cutoff"]
    seed=dataset_config["seed"]

    datasets = read_datasets_list_from_parent_dir(parent_dir, max_samples=None, seed=seed)

    graph_dataset_dir = os.path.dirname(parent_dir)
    graph_foldername = 'aBN_HSX_graphs'
    graph_dataset_dir = os.path.join(graph_dataset_dir, graph_foldername)
    preprocess_and_save_graphs(datasets, orbitals, graph_dataset_dir, cutoff=cutoff)

if __name__ == "__main__":
    main()