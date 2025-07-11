# Hforge

A framework for extracting the Hamiltonian (H) and Overlap (S) matrices from atomic structures.

## Setup

Set up the environment:
```bash
pip install -e .
```
If you need cuda support, then you have to install torch separatelly. Overwriting previous installation is OK, you just have to run the appropiate code according to the [PyTorch documentation | PyTorch](https://pytorch.org/get-started/locally/) after the previous command. For example:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## First use
First you need to download the dataset from HuggingFace. This is done by executing 
```bash
python examples/get_dataset_locally.py
```
However, this will also convert the dataset to graphs. This conversion depends on the orbitals and the cutoff radius that you want for your model. Thus, everytime you want to change these two parameters you will need to change them in the training_loop_config.yaml file, delete the 'aBN_HSX_graphs' folder and rerun this script.

## Examples

### How to use the model in different contexts:

- Run simple inference:
  ```bash
  python examples/deprecated_model_inference.py
  ```

- Run a training loop:
  ```bash
  python examples/training_loop/training_loop.py
  ```

- Test the model:
  ```bash
  python examples/deprecated_model_show.py
  ```

### Data Management

Download the dataset locally:
```bash
python examples/get_dataset_locally.py
```

### Graph Conversion

Test conversion to graph:
```bash
python examples/deprecated_conversion_to_graph.py
```

## Current Status

At the moment, we are focusing only on Hamiltonian generation. Once we reach good performance, the overlap matrix (S) will be relatively easy to extract afterward.

## Todo

- [x] Add support to train with different nr of atoms (Angel)
- [x] Try to download the new data
- [x] Visualization of the eigenvalues of the Intergral Transfer Matrix of H. Use the script of thomas.
- [ ] Study how does the "two (or more) interaction" affect to the convergence of the training (whatever is this interaction term)
- [ ] Improve package importing time PLEASE
- [x] Check if shifts are used inside the MACE implementation (Angel) and add if needed (Andrei)
- [ ] Improve the comments throughout the code (Angel)
- [x] Save the output of the example in a directory call EXAMPLE_info and add it to gitignore (issue+branch*push) (Angel)
- [x] Tune the hyperparameters and see how it affects the results (Angel)
- [X] Try to fix the batching problem when training with different numbers of atoms, limit to 32 (Andrei)
- [ ] Benchmark with other H generating codes (Ange) (maybe around middle of april)
- [ ] Investigate if training on 32 atoms can generalize to 64 atoms (similar to 8→32 generalization) ?
- [ ] Try other PyTorch graph layers to se improve performance ?
- [ ] Figure out why, when startig training an already trained model, it does not resume the same loss value.
- -------------------------------------------------------- (from  21 aprilie)
- [ ] Use E3NN for onsite and hopping extractions
- [ ] Implement mixed training approach

## Lifecare Todo tasks
- [x] Implement a parameter to select if you want to begin training from a checkpoint or not
- [x] Implement facility to resume plotting history when loading previously trained models
- [ ] Make a different checkpoint directory for each training

### Additional Tasks

- [ ] Add shifts when aggregating edges

## Project structure:
This project provides tools and models for extracting quantum mechanical matrix representations from atomic structure data.

    hforge/
    ├── README.md                     # Main documentation file
    ├── setup.py                      # Package installation file
    ├── requirements.txt                      # requirements file file
    ├── src/                          # Source code directory
    │   └── hforge/                   # Main package directory
    │       ├── __init__.py
    │       ├── edge_extraction/      # Edge extraction module
    │       │   ├── edge_extraction_basic.py
    │       │   └── __init__.py
    │       ├── mace/                 # MACE implementation
    │       │   ├── ....
    │       │   └── __init__.py
    │       ├──node_extraction/      # Node extraction module
    │       │   ├── node_extraction_basic.py
    │       │   └── __init__.py
    │       ├── _init_.py
    │       ├── data_preproces.py
    │       ├── edge_agregator.py
    │       ├── graph_dataset.py
    │       ├── model_shell.py
    │       ├── neighbours.py
    │       ├── pretty.py
    │       └── radial.py
    ├── examples/                     # Example scripts directory
    │   ├── conversion_to_graph.py
    │   ├── get_dataset_localy.py
    │   ├── model_inference.py
    │   ├── model_show.py
    │   ├── reshape_with_data.py
    │   ├── to_compare_model_inference.py
    │   └── training_loop.py
    ├── Data/                         # Data directory
    │   └── ...
    ├── EXAMPLE_info/                 # Results of the examples directory
    │   └── ...
    └── .gitignore