import torch

def z_one_hot(z, orbitals, nr_bits):
    """
    Generate one-hot encodings from a list of single-value tensors.

    Args:
        z (list of torch.Tensor): A list of single-value tensors, e.g., [[2], [3], [4], [2], [2], ...].
        orbitals (dict): A dictionary mapping numbers to their corresponding values.
        nr_bits (int): The number of bits for one-hot encoding.

    Returns:
        torch.Tensor: A tensor containing the one-hot encodings.
    """

    # Extract values from the list of single-value tensors
    node_map={}
    k=0
    for key in orbitals.keys():
        node_map[key]=k
        k+=1

    indices = [tensor.item() for tensor in z]
   # print("indices:", indices)

    # Create an empty tensor for one-hot encoding
    one_hot = torch.zeros(len(indices), nr_bits)

    # Fill in the one-hot encoding based on the indices
    for i, idx in enumerate(indices):
        if idx in orbitals:  # Ensure the index exists in orbitals
            one_hot[i, int(node_map[idx])] = 1  # Set the corresponding bit to 1
        else:
            raise ValueError(f"Index {idx} not found in orbitals.")

    return one_hot

