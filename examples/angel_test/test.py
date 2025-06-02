from prettytable import PrettyTable
import os

def main():
    print(4 / 2)
    print( 5 / 2)
    print( 5 // 2)
    print( 5 % 2)

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

    a = torch.tensor([1, 2, 3], device="cuda")
    b = torch.tensor([4, 5, 6])
    print(a.device)


if __name__ == "__main__":
    main()