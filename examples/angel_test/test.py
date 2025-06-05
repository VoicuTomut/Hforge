import torch

def main():
    import torch
    loss = torch.nn.MSELoss()
    input = torch.tensor([[[0, 2, 0], [3, 0, 0]], [[1, 1, 0], [3, 0, 0]]], dtype=torch.float)
    target = torch.tensor([[[1, 0, 0], [3, 0, 0]], [[1, 2, 0], [3, 0, 0]]], dtype=torch.float)
    output = loss(input, target)
    print(output.item())



    a
    ####################################### COO:
    import torch

    # Create a dense tensor and convert to sparse
    a = torch.tensor([[0, 2, 0], [3, 0, 0]])
    b = torch.tensor([[1, 0, 0], [3, 0, 0]])
    c = torch.tensor([[0, 0, 0], [0, 0, 0]])
    sparse_a = a.to_sparse()
    sparse_b = b.to_sparse()
    sparse_a = c.to_sparse()

    # print(a-b)
    # print(sparse_a - sparse_b)
    print(sparse_a.to_dense())
    acccw

    # Print sparse tensor attributes explicitly
    print(f"indices:\n{sparse_a.indices()}")
    print(f"values:\n{sparse_a.values()}")
    print(f"size:\n{sparse_a.size()}")
    print(f"nnz: {sparse_a._nnz()}")
    print(f"layout: {sparse_a.layout}")

    ####################################### cSR:

    import torch

    # Dense tensor
    a = torch.tensor([[0, 2, 0], [3, 0, 0]])

    # Convert to CSR sparse format
    csr_a = a.to_sparse_csr()
    csr_b = b.to_sparse_csr()
    print(a - b)
    print(csr_a - csr_b)

    # Print CSR tensor attributes
    print(f"crow_indices (compressed row pointers):\n{csr_a.crow_indices()}")
    print(f"col_indices (column indices per value):\n{csr_a.col_indices()}")
    print(f"values:\n{csr_a.values()}")
    print(f"size:\n{csr_a.size()}")
    print(f"nnz: {csr_a._nnz()}")
    print(f"layout: {csr_a.layout}")

if __name__ == "__main__":
    main()