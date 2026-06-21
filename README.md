# Graph Attention Networks (GAT) Implementation

This repository contains two Jupyter notebooks demonstrating the implementation and understanding of Graph Attention Networks (GAT), a powerful deep learning architecture for graph-structured data.

---

## 📚 Notebooks Overview

### 1. **Numpy_Single_Head_GAT.ipynb**

**Purpose**: Understand GAT from scratch using pure NumPy implementation.

**What it consists of:**
- Complete mathematical explanation of the GAT mechanism
- Custom Single-Head GAT implementation using only NumPy (no deep learning frameworks)
- Step-by-step implementation of core components:
  - **Linear Transformation**: Feature transformation using weight matrix `W`
  - **Attention Coefficient Calculation**: Compute attention scores `e_ij` between node `i` and its neighbors `j`
  - **Softmax Normalization**: Normalize attention scores using softmax to get `α_ij`
  - **Feature Aggregation**: Aggregate neighbor features weighted by attention coefficients
- Custom implementations of:
  - `softmax()` function
  - `Leaky_ReLU()` activation function
  - `custom_GAT()` function demonstrating the complete workflow

**Key Features:**
- Educational focus on understanding the underlying mathematics
- No external dependencies beyond NumPy
- Visual understanding of how attention works in graph neural networks
- Practical example with a simple 4-node graph

**Basic Application:**
- Learning the fundamentals of graph attention mechanisms
- Understanding how attention weights are computed and applied in graph neural networks
- Foundation for building more complex GAT variants

---

### 2. **Single_Head_&__Multi_Head_&Full_GAT.ipynb**

**Purpose**: Implement GAT using PyTorch with progression from single-head to multi-head to full GAT architecture.

**What it consists of:**
- Complete mathematical explanation of GAT mechanisms
- **Single-Head GAT Implementation** (PyTorch):
  - Basic attention mechanism using PyTorch
  - Sparse softmax implementation for efficient computation
  - Attention coefficient calculation with LeakyReLU activation
  
- **Multi-Head Attention** (PyTorch):
  - Parallel attention heads for richer feature representations
  - Concatenation of multiple attention head outputs
  - Improved learning capacity through ensemble of attention mechanisms

- **Full GAT Implementation**:
  - Complete graph attention layer combining all techniques
  - Training on graph datasets
  - Performance evaluation and visualization

**Key Components:**
- `_sparse_softmax()` function: Efficient sparse softmax for graph operations
- Multi-head attention modules
- Full layer implementations with PyTorch `nn.Module`
- Training loops and evaluation metrics

**Key Features:**
- Production-ready PyTorch implementation
- Support for different attention head configurations
- Efficient computation using sparse operations
- Scalable to larger graph datasets

**Basic Application:**
- Node classification on graph-structured data
- Learning node representations with attention mechanisms
- Building deeper GNN architectures
- Fine-tuning for specific graph learning tasks
- Transfer learning on graph datasets

---

## 🧮 Mathematical Foundation

Both notebooks implement the following GAT computation:

For each node $i$:

1. **Linear Transformation**: $h_i = Wx_i$

2. **Attention Score**: $e_{ij} = \text{LeakyReLU}(a^T[h_i || h_j])$

3. **Normalization**: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$

4. **Aggregation**: $h_i' = \sum_{j \in \mathcal{N}_i} \alpha_{ij} h_j$

Where:
- $W$ is the weight matrix for linear transformation
- $a^T$ is the attention mechanism vector
- $\mathcal{N}_i$ is the set of neighbors of node $i$ (including self-loops)
- $||$ denotes concatenation
- $\alpha_{ij}$ represents the attention weight from node $j$ to node $i$

---

## 🎯 Use Cases & Applications

### When to use which notebook:

- **Numpy_Single_Head_GAT.ipynb**: 
  - Understanding GAT theory and mechanics
  - Educational purposes and tutorials
  - Debugging and understanding attention flows
  - Small-scale graph problems

- **Single_Head_&__Multi_Head_&Full_GAT.ipynb**:
  - Real-world graph datasets
  - Production implementations
  - High-performance requirements
  - Complex graph learning tasks

### Real-world Applications:

1. **Node Classification**: Predicting node labels (e.g., paper categories in citation networks)
2. **Graph Classification**: Classifying entire graphs
3. **Link Prediction**: Predicting missing edges in networks
4. **Social Network Analysis**: Understanding influence and connections
5. **Molecular Graphs**: Predicting molecular properties
6. **Knowledge Graphs**: Reasoning and knowledge completion
7. **Recommendation Systems**: Learning user-item interactions with attention

---

## ⚙️ Requirements

### For Numpy_Single_Head_GAT.ipynb:
- NumPy
- Python 3.x

### For Single_Head_&__Multi_Head_&Full_GAT.ipynb:
- PyTorch
- NumPy
- Python 3.x

---

## 🚀 Getting Started

1. Open either notebook in Jupyter/JupyterLab
2. Run cells sequentially to understand the implementation
3. Modify hyperparameters to experiment with different configurations
4. Apply the concepts to your own graph datasets

---

## 📖 References

Graph Attention Networks are based on the paper:
- Veličković et al. (2017): "Graph Attention Networks" - ICLR 2018

---

## 📝 Notes

- Both implementations follow the same mathematical principles
- The NumPy version is ideal for educational purposes
- The PyTorch version is optimized for practical applications
- Multi-head attention improves learning by using multiple attention mechanisms in parallel
- Sparse softmax operations make the PyTorch implementation scalable to larger graphs

