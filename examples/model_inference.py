"""
example of model inferance
"""


import numpy as np
from datasets import load_from_disk
from hforge.graph_dataset import graph_from_row
from hforge.mace.modules import RealAgnosticResidualInteractionBlock
from hforge.model_shell import ModelShell


def main():
    # Load the dataset and extract a toy sample
    dataset = load_from_disk("/Users/voicutomut/Documents/GitHub/Hforge/Data/aBN_HSX/nr_atoms_3")
    # features: ['nr_atoms', 'atomic_types_z', 'atomic_positions', 'lattice_nsc', 'lattice_origin',
    #            'lattice_vectors', 'boundary_condition', 'h_matrix', 's_matrix']
    print(dataset)
    # Playing_row
    row_index = 4
    sample = dataset[row_index]  # Replace 'train' with the correct split if applicable

    # Preproces the sample to graph form:
    orbitals={
        1:13,
        2:13,
        3:13,
        4:13,
        5:13,
        6:13,
        7:13,
        8:13,}
    sample_graph=graph_from_row(sample,orbitals, cutoff=3.0)
    print(sample_graph)


    # Initialize model
    avg_num_neighbors = 8
    config_model={
        "embedding":{

            'hidden_irreps': "8x0e+8x1o",# 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1

            "r_max":3,
            "num_bessel":8,
            "num_polynomial_cutoff":6,
            "radial_type":"bessel",
            "distance_transform":None,
            "max_ell": 2,
            "num_elements":2,

        },
        "atomic_descriptors":{
            'hidden_irreps': "8x0e+8x1o", ## 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            'avg_num_neighbors':avg_num_neighbors , # need to be computed
            "radial_mlp" : [64, 64, 64],
            'num_interactions': 4,
            "correlation":4, # correlation order of the messages (body order - 1)
            "num_elements":2,
            "max_ell":2,
        },

        "edge_extraction":{
            "orbitals":orbitals,
            "hidden_dim_message_passing":300,
            "hidden_dim_matrix_extraction":200,

        },

        "node_extraction": {
            "orbitals": orbitals,
            "hidden_dim_message_passing": 300,
            "hidden_dim_matrix_extraction": 200,

        },

    }
    model=ModelShell(
        config_model
    )

    # Inference results
    print("__________")
    print(sample_graph)
    output_graph=model(sample_graph)
    print(f"Output graph: {output_graph.keys()}")
    for key in output_graph.keys():
        print(f"{key}: {output_graph[key].shape}")
    print("__________")
    print(sample_graph)




if __name__ == "__main__":
    main()