# Graph Attention Networks (GAT) Implementation

This repository provides three Jupyter notebooks that move from a from-scratch NumPy implementation to practical PyTorch GAT training on citation datasets.

## Notebooks

### 1) Numpy_Single_Head_GAT.ipynb

Purpose: Learn the core GAT math with a minimal, framework-light implementation.

Includes:
- Single-head GAT pipeline in NumPy
- Custom `softmax` and `Leaky_RELU`
- `custom_GAT()` demo on a small hand-crafted graph
- Step-by-step attention flow:
  1. Linear projection
  2. Attention score computation
  3. Softmax normalization
  4. Neighbor aggregation

Best for:
- Understanding GAT internals
- Debugging intuition on tiny graphs

### 2) Single_Head_using_Pytorch.ipynb

Purpose: Build and train a single-head GAT in PyTorch.

Includes:
- `_sparse_softmax()` for edge-wise normalization
- `GAT_Single_Head` layer
- `SingleHeadGAT_Classifier` (2-layer model)
- Training + evaluation on Planetoid datasets:
  - Cora
  - CiteSeer
  - PubMed
- Loss/accuracy plots across epochs

Best for:
- Practical single-head GAT training
- Baseline results on benchmark citation graphs

### 3) Multihead_GAT.ipynb

Purpose: Extend to multi-head attention and compare concat vs average head aggregation.

Includes:
- `GAT_Multihead_Attention` layer
- Head combination modes:
  - Concatenation for hidden layers
  - Averaging for output layer
- `MultiHeadGAT_Classifier`
- Training + evaluation on:
  - Cora
  - CiteSeer
  - PubMed

Best for:
- Multi-head GAT experiments
- Stronger representations via multiple attention heads

## Core GAT Equations

For each node $i$:

1. Linear transform: $h_i = W x_i$
2. Attention score: $e_{ij} = \text{LeakyReLU}(a^T[h_i \Vert h_j])$
3. Normalize over neighbors:
   $$
   \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
   $$
4. Aggregate:
   $$
   h'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h_j
   $$

Where $\mathcal{N}(i)$ includes neighbors (and usually self-loop), and $\Vert$ means concatenation.

## Requirements

- Python 3.x
- NumPy
- PyTorch
- torch-geometric
- matplotlib

Install core packages:

```bash
pip install numpy torch matplotlib torch_geometric
```

## How To Run

1. Open a notebook in Jupyter/Colab/VS Code notebooks.
2. Run cells top-to-bottom.
3. For training sections, keep internet access enabled for dataset download (Planetoid).
4. Tune epochs, dropout, number of heads, and learning rate for experiments.

## Suggested Learning Order

1. `Numpy_Single_Head_GAT.ipynb`
2. `Single_Head_using_Pytorch.ipynb`
3. `Multihead_GAT.ipynb`

## Reference

- Veličković et al., Graph Attention Networks, ICLR 2018.

