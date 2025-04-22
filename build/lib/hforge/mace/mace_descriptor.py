"""
! MACEDescriptor
"""

import torch
from e3nn import o3
from hforge.mace.modules.blocks import  EquivariantProductBasisBlock


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


            #o_hot=embeddings["nodes"]["one_hot"]

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
        # TODO: here I think we can try an enviroment extraction some attention base mechanism between all the interactions
        node_env=node_feats_out
        descriptors = {"nodes":{ "node_env": node_env,}}

        return descriptors

"""
MACEDescriptor !
"""