import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from matplotlib import pyplot as plt
import ase.io
from ase.visualize import view
from scipy.spatial.transform import Rotation
from hforge.mace import data, modules, tools
from hforge.mace.tools import torch_geometric
torch.set_default_dtype(torch.float64)
import warnings
warnings.filterwarnings("ignore")


# setup some default prameters
z_table = tools.AtomicNumberTable([1, 6, 8])
atomic_energies = np.array([-1.0, -3.0, -5.0], dtype=float)
cutoff = 3

default_model_config = dict(
        num_elements=3,  # number of chemical elements
        atomic_energies=atomic_energies,  # atomic energies used for normalisation
        avg_num_neighbors=8,  # avg number of neighbours of the atoms, used for internal normalisation of messages
        atomic_numbers=z_table.zs,  # atomic numbers, used to specify chemical element embeddings of the model
        r_max=cutoff,  # cutoff
        num_bessel=8,  # number of radial features
        num_polynomial_cutoff=6,  # smoothness of the radial cutoff
        max_ell=2,  # expansion order of spherical harmonic adge attributes
        num_interactions=2,  # number of layers, typically 2
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interation block of first layer
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interaction block of subsequent layers
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),  # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
        correlation=3,  # correlation order of the messages (body order - 1)
        MLP_irreps=o3.Irreps("16x0e"),  # number of hidden dimensions of last layer readout MLP
        gate=torch.nn.functional.silu,  # nonlinearity used in last layer readout MLP
    )
default_model = modules.MACE(**default_model_config)

# build mace model
model = modules.MACE(**default_model_config)

single_molecule = ase.io.read('data2/solvent_rotated.xyz', index='0')

Rcut = 3.0 # cutoff radius
z_table = tools.AtomicNumberTable([1, 6, 8])

config = data.Configuration(
    atomic_numbers=single_molecule.numbers,
    positions=single_molecule.positions
)

# we handle configurations using the AtomicData class
batch = data.AtomicData.from_config(config, z_table=z_table, cutoff=Rcut)
print("positions:", batch.positions)
print("node_attrs:", batch.node_attrs)
print("edge_index:", batch.edge_index)


vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
    positions=batch["positions"],
    edge_index=batch["edge_index"],
    shifts=batch["shifts"],
)


print("node atr sapes :",batch.node_attrs.shape)
initial_node_features = model.node_embedding(batch.node_attrs)
print("node_feats:", initial_node_features.shape)

edge_features = model.radial_embedding(lengths, batch["node_attrs"], batch["edge_index"], z_table)
edge_attributes = model.spherical_harmonics(vectors)

print('initial_node_features is (num_atoms, num_channels):', initial_node_features.shape)
print('edge_features is (num_edge, num_bessel_func):', edge_features.shape)
print('edge_attributes is (num_edge, dimension of spherical harmonics):', edge_attributes.shape)
print(
    '\nInitial node features. Note that they are the same for each chemical element\n',
    initial_node_features
)

## let's go to interaction

Interaction = model.interactions[0]

intermediate_node_features, sc = Interaction(
    node_feats=initial_node_features,
    node_attrs=batch.node_attrs,
    edge_feats=edge_features,
    edge_attrs=edge_attributes,
    edge_index=batch.edge_index,
)

graph_dict = {
                "node_attrs": batch.node_attrs.shape,
                "node_feats": initial_node_features.shape,
                "edge_attrs": edge_attributes.shape,
                "edge_feats": edge_features.shape,
                "edge_index": batch.edge_index,
            }
print("interaction input shape:", graph_dict)
print("the output of the interaction is (num_atoms, channels, dim. spherical harmonics):", intermediate_node_features.shape)