"""
Download ehe aBN dataset.
"""
from datasets import load_dataset


def main():
    # Load the dataset
    dataset = load_dataset("AndreiVoicuT/aBN_HSX")
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk("Data/aBN_HSX")

if __name__ == "__main__":
    main()