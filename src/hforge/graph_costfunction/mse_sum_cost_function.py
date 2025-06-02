import torch

def mse_sum_cost_function(pred_graph, target_graph, scale_factor=100.0):
    """
    Calculate loss between predicted and target Hamiltonian and overlap matrices with dynamic weighting.

    Args:
        pred_graph: Dictionary containing predicted edge_description (h_hop) and node_description (s_on_sites)
        target_graph: Dictionary containing target h_hop and s_on_sites values
        scale_factor: Scale factor for target values

    Returns:
        total_loss: The sum of edge_loss and node_loss.
        (Dict): Dictionary containing edge_loss and node_loss separately.
    """
    # Extract predictions and targets
    edge_pred = pred_graph["edge_description"]
    node_pred = pred_graph["node_description"]

    edge_target = target_graph["edge_description"] * scale_factor
    node_target = target_graph["node_description"] * scale_factor

    # Compute MSE loss for both matrices
    edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_target, reduction="sum")
    node_loss = torch.nn.functional.mse_loss(node_pred, node_target, reduction="sum")

    # Dynamic weighting based on loss magnitude
    edge_weight = 1.0
    node_weight = 1.0

    # Combined loss
    total_loss = edge_weight * edge_loss + node_weight * node_loss

    # Detect extremely large values and clip
    if total_loss > 1e12:
        print(f"Unusually large loss detected: {total_loss.item()}")
        total_loss = torch.clamp(total_loss, max=1e12)

    return total_loss, {"edge_loss": edge_loss.item(), "node_loss": node_loss.item()}
