"""Functions to use the model"""

# Standard library imports
import torch

from hforge.plots.plot_matrix import reconstruct_matrix


def generate_prediction(model, input_graph):
    # Automatically use the model's device. Otherwise it will raise an error.
    device = next(model.parameters()).device
    input_graph = input_graph.to(device)

    model.eval()
    with torch.no_grad():
        # Generate prediction
        return model(input_graph)

def generate_hamiltonian_prediction(model, input_graph, loss_fn):
    output_graph = generate_prediction(model, input_graph)
    target_graph = {
        "edge_index": output_graph["edge_index"],
        "edge_description": input_graph.h_hop,
        "node_description": input_graph.h_on_sites
    }

    loss, _ = loss_fn(output_graph, target_graph)

    print("\n\nn_atoms: ", len(input_graph["x"]))

    print("\npred_graph:")
    print("len(hop): ", len(output_graph["edge_description"]))
    print("len(onsite): ", len(output_graph["node_description"]))

    # print("\ntarget_graph:")
    # print("len(hop): ", len(target_graph["edge_description"]))  
    # print("len(onsite): ", len(target_graph["node_description"]))
    predicted_h = reconstruct_matrix(output_graph["edge_description"], output_graph["node_description"], output_graph["edge_index"])
    original_h = reconstruct_matrix(target_graph["edge_description"], target_graph["node_description"], output_graph["edge_index"])
    

    return loss, predicted_h, original_h