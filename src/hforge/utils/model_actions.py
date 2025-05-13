"""Functions to use the model"""

# Standard library imports
import torch

def generate_prediction(model, input_graph):
    model.eval()
    with torch.no_grad():
        # Generate prediction
        return model(input_graph)