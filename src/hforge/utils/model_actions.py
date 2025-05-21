"""Functions to use the model"""

# Standard library imports
import torch

def generate_prediction(model, input_graph):
    # Automatically use the model's device. Otherwise it will raise an error.
    device = next(model.parameters()).device
    input_graph = input_graph.to(device)

    model.eval()
    with torch.no_grad():
        # Generate prediction
        return model(input_graph)