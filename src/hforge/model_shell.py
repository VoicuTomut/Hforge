"""
Model shell is a wrapper around the cor part of the model.
Why is this necessary?
This is used because it keeps the input to th emodel simple so the only inputs of the model are:
- atom type and position
- edges
"""

import torch
from e3nn import o3
from hforge.edge_agregator import EdgeAggregator
from hforge.mace.mace_descriptor import MACEDescriptor
from hforge.encodings import EmbeddingBase
from hforge.utils.importing_facilities import get_object_from_module


def compute_mace_output_shape(config):
    """
    Compute the output shape of MACE descriptor based on configuration and number of atoms.

    Parameters:
    -----------
    config : dict
        Dictionary containing configuration parameters:
        - hidden_irreps: str, e.g. "8x0e+8x1o"
        - num_interactions: int, number of interaction layers

    Returns:
    --------
    tuple
        Output shape as (num_atoms, features)

    int
        Total number of features per atom
    """
    # Parse hidden irreps
    irreps = o3.Irreps(config['hidden_irreps'])

    # Calculate dimensions per irrep type
    scalar_features = irreps.count(o3.Irrep(0, 1))
    vector_features = irreps.count(o3.Irrep(1, -1))

    # Total components (each vector has 3 components)
    total_components = scalar_features + (vector_features * 3)

    # Total features across all interactions
    total_features = total_components * config['num_interactions']


    return total_features


class ModelShell(torch.nn.Module):
    # ! Once the model is initialized, you cannot move it to a different device!

    def __init__(self, model_config, device='cpu'):
        super(ModelShell, self).__init__()

        self.embedding=EmbeddingBase(model_config["embedding"])

        model_config["atomic_descriptors"]["radial_embedding.out_dim"]=self.embedding.radial_embedding.out_dim
        max_ell=model_config["embedding"]["max_ell"]
        model_config["atomic_descriptors"]["angular_embedding.out_dim"] = sum(2 * l + 1 for l in range(max_ell + 1))
        self.atomic_descriptors=MACEDescriptor(model_config["atomic_descriptors"])


        self.edge_aggregator=EdgeAggregator(edge_dim_radial=model_config["atomic_descriptors"]["radial_embedding.out_dim"],
                                          edge_dim_angular= model_config["atomic_descriptors"]["angular_embedding.out_dim"],
                                          hidden_dim=5,
                                          num_layers=3)

        model_config["edge_extraction"]["edge_radial_dim"]=model_config["atomic_descriptors"]["radial_embedding.out_dim"]
        model_config["edge_extraction"]["edge_angular_dim"] = model_config["atomic_descriptors"][
            "angular_embedding.out_dim"]
        features = compute_mace_output_shape( model_config["atomic_descriptors"])
        model_config["edge_extraction"]["descriptor_dim"]=features


        # === Initialize edge extraction ===
        self.edge_extraction = get_object_from_module(model_config["edge_extraction"].get("edge_extraction_class", "EdgeExtractionBasic"), 'hforge.edge_extraction')(model_config["edge_extraction"], device = device)
        
        # === Initialize node extraction ===
        model_config["node_extraction"]["edge_radial_dim"] = model_config["atomic_descriptors"]["radial_embedding.out_dim"]
        model_config["node_extraction"]["edge_angular_dim"] = model_config["atomic_descriptors"]["angular_embedding.out_dim"]
        model_config["node_extraction"]["descriptor_dim"] = features

        model_config["node_extraction"]["node_extraction_class"] = model_config["node_extraction"].get("node_extraction_class", "NodeExtractionBasic")

        self.node_extraction = get_object_from_module(model_config["node_extraction"]["node_extraction_class"], 'hforge.node_extraction')(model_config["node_extraction"], device = device)

        self.global_extraction=None


    def forward(self, graph_data):
        # Embeddings
        embeddings=self.embedding(graph_data)

        # interaction :
        atomic_env_descriptor=self.atomic_descriptors(
            graph_data=graph_data,
            embeddings=embeddings,
        )

        # Edge agregator:
        if self.edge_aggregator is not None:
            embeddings['edges']['radial_embedding'], embeddings['edges']['angular_embedding'], graph_data["reduce_edge_index"]=self.edge_aggregator(
                edge_index=graph_data["edge_index"],
                radial_embedding=embeddings['edges']['radial_embedding'],
                angular_embedding=embeddings['edges']['angular_embedding']
            )

        # Message Passing and extract information:
        model_results= {"edge_index":graph_data["reduce_edge_index"]}
        if self.edge_extraction is not None:
            edge_description=self.edge_extraction(graph_data,embeddings,atomic_env_descriptor)
            model_results["edge_description"]=edge_description
        if self.node_extraction is not None:
            node_description=self.node_extraction(graph_data,embeddings,atomic_env_descriptor)
            model_results["node_description"]=node_description
        if self.global_extraction is not None:
            global_info=self.global_extraction(graph_data,embeddings,atomic_env_descriptor)
            model_results["global_info"]=global_info

        # Eliminate self loops from edges:
        model_results["edge_index"] , model_results["edge_description"] = remove_self_loops(model_results["edge_index"], model_results["edge_description"])
        # print("model_results.keys()= ", model_results.keys()) # dict_keys(['edge_index', 'edge_description', 'node_description'])
        # print("model_results[edge_index].shape]= ", model_results["edge_index"].shape)
        # print("model_results[edge_description].shape]= ", model_results["edge_description"].shape)
        # print("model_results[node_description].shape]= ", model_results["node_description"].shape)
        # print("graph_data.keys()= ", graph_data.keys())
        # print("graph_data[pos].shape= ", graph_data["pos"].shape)
        # print("graph_data[h_on_sites].shape= ", graph_data["h_on_sites"].shape)

        return model_results



def remove_self_loops(edge_index, edge_desc):
    """
    Removes self-loop edges (like 0-0, 1-1) from edge_index and edge_desc.

    Args:
        edge_index (torch.Tensor): Tensor of shape [2, num_edges]
        edge_desc (torch.Tensor): Tensor of shape [num_edges, ...]

    Returns:
        filtered_edge_index (torch.Tensor): Edge index without self-loops
        filtered_edge_desc (torch.Tensor): Edge desc without self-loop edges
    """
    # Mask for non-self-loop edges
    mask = edge_index[0] != edge_index[1]

    # Apply mask
    filtered_edge_index = edge_index[:, mask]
    filtered_edge_desc = edge_desc[mask]

    return filtered_edge_index, filtered_edge_desc





