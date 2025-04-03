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
HOWEVER CUDA IS NOT WORKING YET

## Examples

### How to use the model in different contexts:

- Run simple inference:
  ```bash
  python examples/model_inference.py
  ```

- Run a training loop:
  ```bash
  python examples/training_loop.py
  ```

- Test the model:
  ```bash
  python examples/model_show.py
  ```

### Data Management

Download the dataset locally:
```bash
python examples/get_dataset_localy.py
```

### Graph Conversion

Test conversion to graph:
```bash
python examples/conversion_to_graph.py
```

## Current Status

At the moment, we are focusing only on Hamiltonian generation. Once we reach good performance, the overlap matrix (S) will be relatively easy to extract afterward.

## Todo

- [ ] WHY IS THE TRAINING SLOWER WITH CUDA
- [ ] Improve package importing time PLEASE
- [x] Check if shifts are used inside the MACE implementation (Angel) and add if needed (Andrei)
- [ ] Improve the comments throughout the code (Angel)
- [x] Save the output of the example in a directory call EXAMPLE_info and add it to gitignore (issue+branch*push) (Angel)
- [ ] Tune the hyperparameters and see how it affects the results (Angel)
- [X] Try to fix the batching problem when training with different numbers of atoms, limit to 32 (Andrei)
- [ ] Benchmark with other H generating codes (Ange) (maybe around middle of april)
- [ ] Investigate if training on 32 atoms can generalize to 64 atoms (similar to 8→32 generalization) ?
- [ ] Try other PyTorch graph layers to se improve performance ?
- -------------------------------------------------------- (from  21 aprilie)
- [ ] Use E3NN for onsite and hopping extractions
- [ ] Implement mixed training approach

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