"""
Script to convert the HuggingFace dataset into graphs. Take into account that the graphs depends on some training config: orbitals and cutoff radious, so each time these parameters change, you should delete the graphs folder and recompute them.
"""
import os

from hforge.utils import load_config
from hforge.data_management.dataset_load import read_datasets_list_from_parent_dir
from hforge.data_management.data_processing import preprocess_and_save_graphs

def main():
    # === Load configuration ===
    config = load_config()
    dataset_config = config["dataset"]
    orbitals = config["orbitals"]
    cutoff=dataset_config["cutoff"]

    # === Load ALL aBN dataset and convert it to graph ===

    # Load datasets
    parent_dir = dataset_config["path"]
    seed = dataset_config["seed"]
    datasets = read_datasets_list_from_parent_dir(parent_dir, max_samples=None, seed=seed)

    # Convert them to graphs and save locally
    graph_dataset_dir = os.path.dirname(parent_dir)
    graph_foldername = 'aBN_HSX_graphs'
    graph_dataset_dir = os.path.join(graph_dataset_dir, graph_foldername)
    preprocess_and_save_graphs(datasets, orbitals, graph_dataset_dir, cutoff=cutoff)

if __name__ == "__main__":
    main()