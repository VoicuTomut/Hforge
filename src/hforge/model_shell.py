"""
Model shell is a wrapper around the cor part of the model.
Why is this necessary?
This is used because it keeps the input to th emodel simple so the only inputs of the model are:
- atom type and position
- edges
"""

import torch

from hforge.pretty import pretty_print_dict
from hforge.mace.modules.blocks import RadialEmbeddingBlock, EquivariantProductBasisBlock
from hforge.mace.modules.utils import get_edge_vectors_and_lengths
from hforge.edge_agregator import EdgeAggregator
from hforge.edge_extraction import EdgeExtractionBasic
from hforge.node_extraction import NodeExtractionBasic

def z_one_hot(z):

    """
    Generate one-hot encodings from a list of single-value tensors.

    Args:
        z (list of torch.Tensor): A list of single-value tensors, e.g., [[2], [3], [4], [2], [2], ...].

    Returns:
        torch.Tensor: A tensor containing the one-hot encodings.
    """
    # Flatten and extract unique values, sort in ascending order
    unique_values = torch.unique(torch.tensor(z).flatten())
    unique_values = torch.sort(unique_values).values

    # Create a mapping from value to index
    value_to_index = {value.item(): idx for idx, value in enumerate(unique_values)}

    # Create the one-hot encoded tensor
    num_classes = len(unique_values)
    indices = torch.tensor([value_to_index[val.item()] for val in torch.tensor(z).flatten()])
    one_hot = torch.nn.functional.one_hot(indices, num_classes=num_classes).to(torch.float32)

    return one_hot


def compute_mace_output_shape( config):
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

    def __init__(self, config_routine):
        super(ModelShell, self).__init__()

        self.embedding=EmbeddingBase(config_routine["embedding"])

        config_routine["atomic_descriptors"]["radial_embedding.out_dim"]=self.embedding.radial_embedding.out_dim
        max_ell=config_routine["embedding"]["max_ell"]
        config_routine["atomic_descriptors"]["angular_embedding.out_dim"] = sum(2 * l + 1 for l in range(max_ell + 1))
        self.atomic_descriptors=MACEDescriptor(config_routine["atomic_descriptors"])


        self.edge_aggregator=EdgeAggregator(edge_dim_radial=config_routine["atomic_descriptors"]["radial_embedding.out_dim"],
                                          edge_dim_angular= config_routine["atomic_descriptors"]["angular_embedding.out_dim"],
                                          hidden_dim=5,
                                          num_layers=3)





        config_routine["edge_extraction"]["edge_radial_dim"]=config_routine["atomic_descriptors"]["radial_embedding.out_dim"]
        config_routine["edge_extraction"]["edge_angular_dim"] = config_routine["atomic_descriptors"][
            "angular_embedding.out_dim"]
        features = compute_mace_output_shape( config_routine["atomic_descriptors"])
        config_routine["edge_extraction"]["descriptor_dim"]=features

        self.edge_extraction=EdgeExtractionBasic(
            config_routine["edge_extraction"]
        )
        self.node_extraction = NodeExtractionBasic(config_routine["node_extraction"])
        self.global_extraction=None


    def forward(self, graph_data):

        # Embeddings
        embeddings=self.embedding(graph_data)
        print("##! embeddings ##")
        #pretty_print_dict(embeddings)
        print("## embeddings !##")
        #/Embeddings


        # interaction :
        atomic_env_descriptor=self.atomic_descriptors(
            graph_data=graph_data,
            embeddings=embeddings,

        )
        print("##! descriptor ##")
        #pretty_print_dict(atomic_env_descriptor)
        print("## descriptor !##")
        # /interaction :


        # Edge agregator:
        print("edge_index:", graph_data["edge_index"])
        if self.edge_aggregator is not None:
            embeddings['edges']['radial_embedding'], embeddings['edges']['angular_embedding'], graph_data["reduce_edge_index"]=self.edge_aggregator(
                edge_index=graph_data["edge_index"],
                radial_embedding=embeddings['edges']['radial_embedding'],
                angular_embedding=embeddings['edges']['angular_embedding']
            )





        # Extract information:
        print("# Extract information:")
        model_results= {"edge_indes":graph_data["edge_index"]}
        if self.edge_extraction is not None:
            edge_description=self.edge_extraction(graph_data,embeddings,atomic_env_descriptor )
            model_results["edge_description"]=edge_description
        if self.node_extraction is not None:
            node_description=self.node_extraction(graph_data,embeddings,atomic_env_descriptor)
            model_results["node_description"]=node_description
        if self.global_extraction is not None:
            global_info=self.global_extraction(graph_data,embeddings,atomic_env_descriptor)
            model_results["global_info"]=global_info


        return model_results


"""
! Embeddings o3
"""
from e3nn import o3
class EmbeddingBase(torch.nn.Module):

    def __init__(self,config_routine):
        super(EmbeddingBase, self).__init__()

        print("##! embeddings-config ##")
        pretty_print_dict(config_routine)


        node_attr_irreps = o3.Irreps([(config_routine["num_elements"], (0, 1))])
        #print(f"mace shell {node_attr_irreps=}")

        hidden_irreps=o3.Irreps(config_routine["hidden_irreps"])
        node_feats_irreps =o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])

        #print(f"mace shell {node_feats_irreps=}")

        self.node_embedding = o3.Linear(
            node_attr_irreps,
            node_feats_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=config_routine["r_max"],
            num_bessel=config_routine["num_bessel"],
            num_polynomial_cutoff=config_routine["num_polynomial_cutoff"],
            radial_type=config_routine["radial_type"],
            distance_transform=config_routine["distance_transform"],
        )
        sh_irreps=o3.Irreps.spherical_harmonics(config_routine["max_ell"])
        #print(f"mace shell {sh_irreps=}")
        self.angular_embedding = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")


    def forward(self, graph_data):

            # Embeddings
           # print("graph_data passed to embedding:", graph_data)

            one_hot_z=z_one_hot(graph_data.x)



            # Atomic numbers to binery encodding
            node_description=one_hot_z
            print("one hot z shapes :",one_hot_z.shape)
            # Atomic distances:
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=graph_data.pos,
                edge_index=graph_data.edge_index,
                shifts=graph_data.shifts,
            )
            #print("ss shape",self.node_embedding(node_description).shape)

            #print(" shell node atr sapes :", node_description.shape)
            node_feats=self.node_embedding(node_description)
            #print("node_feats:", node_feats.shape)
            radial_embedding=self.radial_embedding( lengths,
                                                              node_description,
                                                              graph_data.edge_index,
                                                              graph_data.x)
            angular_embedding=self.angular_embedding(vectors)

            embedding_collection={
                "nodes":{
                    "one_hot":one_hot_z,
                    "node_features":node_feats,

                },
                "edges":{
                    "radial_embedding":radial_embedding, #atomic numbers here is abit of uncertenty
                    "angular_embedding":angular_embedding,
                }
            }

            embedding_collection_shapes = {
                "nodes":{
                    "one_hot":one_hot_z.shape,
                    "node_features":node_feats.shape,

                },
                "edges":{
                    "radial_embedding":radial_embedding.shape, #atomic numbers here is abit of uncertenty
                    "angular_embedding":angular_embedding.shape,
                }
            }

            #print("embedding_collection_shapes:")
            #print(embedding_collection_shapes)
            return embedding_collection

"""
Embeddings o3 !
"""


"""
! MACEDescriptor
"""


class MACEDescriptor(torch.nn.Module):

    def __init__(self, config_routine):
        super(MACEDescriptor, self).__init__()


        # get the input output representations for etch step:


        node_attr_irreps = o3.Irreps([(config_routine["num_elements"], (0, 1))])

        node_feats_irreps = o3.Irreps([(o3.Irreps(config_routine["hidden_irreps"]).count(o3.Irrep(0, 1)), (0, 1))])
        sh_irreps = o3.Irreps.spherical_harmonics(config_routine["max_ell"])
        radial_embedding_out_dim=config_routine["radial_embedding.out_dim"]
        edge_feats_irreps = o3.Irreps(f"{radial_embedding_out_dim}x0e")
        hidden_irreps=o3.Irreps(config_routine["hidden_irreps"])
        hidden_irreps_out = o3.Irreps(config_routine["hidden_irreps"])

        num_features = hidden_irreps.count(str(o3.Irrep(0, 1)))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()


        if isinstance(config_routine["correlation"], int):
            correlation = [config_routine["correlation"]] *config_routine["num_interactions"]



        # First interaction and the first product:

        inter = config_routine["interaction_cls_first"](
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=config_routine["avg_num_neighbors"],
            radial_MLP=config_routine["radial_mlp"],
            cueq_config=None,
        )

        config_dict_interact = {
            "node_attrs_irreps": node_attr_irreps,
            "node_feats_irreps": node_feats_irreps,
            "edge_attrs_irreps": sh_irreps,
            "edge_feats_irreps": edge_feats_irreps,
            "target_irreps": interaction_irreps,
            "hidden_irreps": hidden_irreps,
            "avg_num_neighbors": config_routine["avg_num_neighbors"],
            "radial_MLP": config_routine["radial_mlp"],
            "cueq_config": None,
        }
        #print("interaction class seting:", config_dict_interact)

        self.interactions = torch.nn.ModuleList([inter])
        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(config_routine["interaction_cls_first"]):
            use_sc_first = True

        prod = EquivariantProductBasisBlock(
            node_feats_irreps=inter.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=config_routine["num_elements"],
            use_sc=use_sc_first,
            cueq_config=None,
        )
        self.products = torch.nn.ModuleList([prod])

        # Chain  of interaction and product:
        num_interactions=config_routine["num_interactions"]

        for i in range(num_interactions - 1):

            inter = config_routine["interaction_cls"](
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=config_routine["avg_num_neighbors"],
                radial_MLP=config_routine["radial_mlp"],
                cueq_config=None,
            )


            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=config_routine["num_elements"],
                use_sc=True,
                cueq_config=None,
            )
            self.products.append(prod)



    def forward(self, graph_data, embeddings):


        node_feats_list=[]
        k=1
        node_feats = embeddings["nodes"]["node_features"]
        for interaction, product in zip(
            self.interactions, self.products
        ):
            #print("Interaction level:", k)
            k+=1

            node_feats, sc = interaction(
                node_attrs=embeddings["nodes"]["one_hot"],
                node_feats=node_feats,
                edge_attrs=embeddings["edges"]["angular_embedding"],
                edge_feats=embeddings["edges"]["radial_embedding"],
                edge_index=graph_data.edge_index,
            )

            #graph_dict = {
             #   "node_attrs": embeddings["nodes"]["one_hot"].shape,
              #  "node_feats":node_feats.shape,
              #  "edge_attrs": embeddings["edges"]["angular_embedding"].shape,
               # "edge_feats": embeddings["edges"]["radial_embedding"].shape,
                #"edge_index": graph_data.edge_index,
            #}
           # print("interaction input shape:", graph_dict)

            #o_hot=embeddings["nodes"]["one_hot"]
            #print(f"product_inputs:{node_feats.dtype=}\n {sc.dtype=}\n {o_hot.dtype=}\n", )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=embeddings["nodes"]["one_hot"],
            )
            node_feats_list.append(node_feats)

            # TODO: here we can tray to add some aditional trnasofrmation.
            #node_transformation = readout(node_feats, node_heads)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)


        # Outputs
        # TODO: here I think we can try an environment extraction some attention base mechanism between all the interactions
        node_env=node_feats_out
        descriptors = {"nodes":{ "node_env": node_env,}}

       # print("Node descriptor:", node_env)
        return descriptors

"""
MACEDescriptor !
"""



