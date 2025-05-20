from prettytable import PrettyTable
import os

def main():
    table = PrettyTable(["Type", "Last model save", "New prediction"])
    print(table)

    a = "angel"
    end = None
    start = None
    print(a[start:end])

    folder = os.path.basename(os.path.normpath('/folderA/folderB/folderC/folderD/'))
    print(folder)

    import torch

    torch.manual_seed(42)  # Set a fixed seed

    subset_size = 10  # Example subset size
    dataset_size = 100  # Example dataset size

    # Generate random indices twice with the same seed
    indices1 = torch.randperm(dataset_size)[:subset_size]
    torch.manual_seed(42)
    indices2 = torch.randperm(dataset_size)[:subset_size]

    print("First set of indices:", indices1)
    print("Second set of indices:", indices2)

    # Check if the generated indices are identical
    print("Reproducible:", torch.equal(indices1, indices2))

if __name__ == "__main__":
    main()