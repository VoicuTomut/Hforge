"""
! Embeddings o3
"""
import torch
from e3nn import o3
from hforge.mace.modules.blocks import RadialEmbeddingBlock, EquivariantProductBasisBlock
from hforge.mace.modules.utils import get_edge_vectors_and_lengths
from hforge.one_hot import z_one_hot

class EmbeddingBase(torch.nn.Module):

    def __init__(self,config_routine):
        super(EmbeddingBase, self).__init__()





        node_attr_irreps = o3.Irreps([(config_routine["num_elements"], (0, 1))])


        hidden_irreps=o3.Irreps(config_routine["hidden_irreps"])
        node_feats_irreps =o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])



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

        self.angular_embedding = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        self.orbitals=config_routine["orbitals"]
        self.nr_bit=config_routine["nr_bits"]

    def forward(self, graph_data):

            # Embeddings

            device = graph_data.x.device
            one_hot_z=z_one_hot(graph_data.x, orbitals=self.orbitals, nr_bits=self.nr_bit).to(device)



            # Atomic numbers to binery encodding
            node_description=one_hot_z

            # Atomic distances:
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=graph_data.pos,
                edge_index=graph_data.edge_index,
                shifts=graph_data.shifts,
            )


            node_feats=self.node_embedding(node_description)
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
                    "radial_embedding":radial_embedding, #atomic numbers here is a bit of uncertenty
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

            return embedding_collection

"""
Embeddings o3 !
"""
