"""Script to study the data before training."""

from prettytable import PrettyTable

from hforge.data_management.dataset_load import prepare_dataset_from_parent_dir
from hforge.utils import load_config


# from hforge.utils import prepare_dataset, load_config


def main():
    # Load the dataset
    dataset_directory = "Data/aBN_HSX/nr_atoms_2"
    config_path = "examples/training_loop/training_loop_config.yaml"
    config = load_config(config_path)
    dataset_config = config["dataset"]
    # dataset = prepare_dataset(
    #     dataset_path=dataset_directory,
    #     orbitals=config["orbitals"],
    #     training_split_ratio=dataset_config["split_ratio"],
    #     cutoff=dataset_config["cutoff"],
    #     # max_samples=dataset_config["max_samples"],
    #     max_samples=None,
    #     load_other_nr_atoms=dataset_config["load_other_nr_atoms"],
    #     split=False
    # )

    # train_dataset, val_dataset, test_dataset = prepare_dataset(
    #     dataset_path=dataset_directory,
    #     orbitals=config["orbitals"],
    #     training_split_ratio=dataset_config["split_ratio"],
    #     cutoff=dataset_config["cutoff"],
    #     max_samples=dataset_config["max_samples"],
    #     # max_samples=None,
    #     load_other_nr_atoms=dataset_config["load_other_nr_atoms"],
    #     split=True
    # )

    parent_dir = "Data/aBN_HSX"
    dataset = prepare_dataset_from_parent_dir(parent_dir, orbitals=config["orbitals"], cutoff=dataset_config["cutoff"], max_samples=dataset_config["max_samples"], seed=42)

    # # * change this
    # dataset = val_dataset

    # Count the number of each sample
    unique_n_atoms_list = find_unique_n_atoms(dataset)
    counts_n_atoms = []
    for n_atoms in unique_n_atoms_list:
        counts_n_atoms.append(count_samples(dataset, n_atoms))


    # === Print some stats ===
    table = PrettyTable(["n_atoms", "Amount"])
    for i, n_atoms in enumerate(unique_n_atoms_list):
        table.add_row([n_atoms, counts_n_atoms[i]])

    # n_train_samples = len(train_dataset)
    # n_val_samples = len(val_dataset)
    # n_test_samples = len(test_dataset)
    # print(f"There are {n_train_samples}/{n_val_samples}/{n_test_samples} training/validation/test samples.")
    print(f"There are {len(dataset)} samples in the dataset.")
    print(table)

def find_unique_n_atoms(dataset):
    """Finds all the unique number of atoms graph types that there are in the dataset."""
    list_n_atoms = [len(sample["x"]) for sample in dataset]
    unique_n_atoms = list(dict.fromkeys(list_n_atoms))
    return unique_n_atoms

def count_samples(dataset, n_atoms_query):
    counter = 0
    for sample in dataset:
        n_atoms = len(sample["x"])
        if n_atoms == n_atoms_query:
            counter += 1
    return counter

if __name__ == '__main__':
    main()