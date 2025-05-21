import torch

def print_graph_device(graph):
    for key, value in graph:
        if torch.is_tensor(value):
            print(f"{key}: {value.device}")

def print_data_loader_device(data_loader):
    for i, batch in enumerate(data_loader):
        print(f"Batch {i}")
        for attr in dir(batch):
            if not attr.startswith('_'):
                val = getattr(batch, attr)
                if torch.is_tensor(val):
                    print(f"  {attr}: {val.device}, shape={val.shape}")
        if i == 2:  # Only print first 3 batches
            break