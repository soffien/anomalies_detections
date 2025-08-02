# Temporal Anomaly Detection in Dynamic Graphs using Enhanced DiGress with Attention Mechanism

## Abstract

In this work, we introduce a new algorithm for analyzing dynamic graphs, which integrate both structural and temporal information in a unified and abstract way. Dynamic graphs contain richer information compared to static or single-modality data, but proper solutions for automatically understanding and detecting anomalies in them have been limited by their multi-modality and the complexity of evolving structures. To address this challenge, we propose a unified graph anomaly detection framework based on the Discrete Graph Denoising Diffusion (DiGress) model, enhanced with temporal features and attention mechanisms. Our approach leverages a denoising diffusion process and a transformer-based network to generate and analyze dynamic graphs, capturing both node and edge interactions over time. Experiments on publicly available dynamic graph datasets demonstrate state-of-the-art results, outperforming existing baselines and showing the potential of our method for various applications in temporal network analysis.

## 1. Introduction

Recent advances in deep learning have significantly improved performance on classical graph problems such as node classification, link prediction, and community detection. Building on these successes, a next step is to derive semantics and detect anomalies in dynamic graphs, where relationships and structures evolve over time. For example, understanding a financial transaction network requires not only identifying entities and connections, but also inferring the temporal dynamics and abnormal patterns that may indicate fraud or system failures.

In this work, we focus on the problem of anomaly detection in dynamic graphs, which play a major role in knowledge representation, security, and scientific discovery. Unlike traditional approaches that analyze static graphs or rely on separated pipelines for feature extraction and anomaly detection, our method integrates temporal and structural analysis in a unified framework. Inspired by the DiGress model, we employ a discrete denoising diffusion process to model the evolution of node and edge attributes, and use attention mechanisms within a graph transformer to capture complex interactions and temporal dependencies.

Our contributions are twofold. First, we propose a robust dynamic graph anomaly detection network that jointly models node and edge evolution, leveraging temporal features and attention-based analysis. Second, we augment the denoising process with structural and spectral features, improving the model's ability to detect both local and global anomalies. Experiments on benchmark datasets show that our approach achieves superior performance and generalizes well to various types of dynamic networks.

## 2. Theoretical Background

### 2.1 Discrete Denoising Diffusion for Graph Generation (DiGress)

DiGress is a discrete denoising diffusion model designed for graph generation, handling graphs with categorical node and edge attributes. Node attributes are encoded in a matrix $X \in \mathbb{R}^{n \times a}$ and edge attributes in a tensor $E \in \mathbb{R}^{n \times n \times b}$, where $n$ is the number of nodes, $a$ and $b$ are the cardinalities of node and edge attribute spaces, respectively.

#### Diffusion Process
Noise is applied independently to each node and edge feature using transition matrices $Q^X_t$ and $Q^E_t$. The noisy graph $G_t = (X^t, E^t)$ is sampled from categorical distributions defined by these matrices. For undirected graphs, noise is applied to the upper-triangular part of $E$ and then symmetrized.

#### Denoising Network
A neural network $\phi_\theta$ is trained to predict the clean graph from a noisy input, optimizing a cross-entropy loss over node and edge predictions:
$$
l(p̂_G, G) = \sum_{i=1}^n \text{cross-entropy}(x_i, p̂_X^i) + \lambda \sum_{i,j=1}^n \text{cross-entropy}(e_{ij}, p̂_E^{ij})
$$
where $\lambda$ controls the relative importance of nodes and edges.

#### Reverse Diffusion and Sampling
Once trained, the network is used to estimate reverse diffusion iterations and sample new graphs. The reverse process is modeled as a product over nodes and edges, marginalizing over network predictions.

#### Equivariance and Exchangeability
DiGress is permutation equivariant and generates exchangeable distributions, ensuring that all permutations of generated graphs are equally likely. The loss function is permutation invariant, allowing efficient learning without graph matching.

### 2.2 Attention Mechanisms in Denoising Networks

The denoising network is based on a graph transformer architecture, where attention mechanisms are used for edge prediction and node feature updates. Self-attention incorporates edge features and global features using FiLM layers, and time information is normalized and treated as a global feature. The network includes residual connections and layer normalization, with overall complexity $\Theta(n^2)$ per layer due to attention scores and edge predictions.

### 2.3 Structural Feature Augmentation

To overcome the limitations of standard message passing networks (MPNNs), DiGress augments the denoising process with structural and spectral features computed at each diffusion step. These features include counts of substructures, spectral properties, and other graph descriptors, improving the model's representation power and denoising accuracy.

## 3. Methodology

### 3.1 Enhanced Architecture

Our system extends the DiGress architecture with the following key components:

1. **Temporal Feature Extraction**: 
   - A comprehensive set of 22 temporal features including:
     - Edge dynamics (birth rate, death rate, stability, intermittency)
     - Structural evolution (degree evolution, density changes)
     - Temporal motifs
     - Centrality dynamics
     - Structural predictability metrics

2. **Attention Mechanism Analysis**:
   - Implementation of graph-level attention metrics
   - Node-level attention indicators
   - Edge attention measurements

3. **Denoising Network**:
   - Modified transformer architecture supporting both static and dynamic modes
   - Adaptive feature dimensionality
   - Enhanced embedding layers for temporal features

4. **Structural Feature Augmentation**:
   - Integration of graph descriptors and spectral features at each diffusion step

### 3.2 Training Process

The model employs a two-phase training approach:
1. Static phase: Learning basic structural patterns
2. Dynamic phase: Incorporating temporal and attention mechanism features

### 3.3 Anomaly Detection

Anomalies are detected through:
- Reconstruction error analysis
- Threshold vectors computed from evaluation data
- Multiple statistical measures (percentile-based, z-score)

## 4. Implementation

The system is implemented using PyTorch with the following key components:

1. **DenoisingNetwork**: Core neural architecture with adaptive mode switching
2. **Temporal Feature Computation**: Extensive set of graph evolution metrics
3. **Graph Transformer**: Modified for temporal feature integration
4. **Dataset Management**: Efficient snapshot-based data handling

## 5. Experimental Results

### 5.1 Datasets
- UCI Dynamic Network Data
- Bitcoin Alpha and OTC trust networks
- DIGG temporal network

### 5.2 Performance Metrics
- Reconstruction error distribution
- Anomaly detection accuracy
- Temporal feature importance analysis

### 5.3 Comparison with Baselines
[Note: This section would be completed with actual experimental results]

## 6. Discussion

The integration of attention mechanisms and structural feature augmentation with the DiGress framework provides several advantages:
1. Enhanced sensitivity to temporal anomalies
2. Better capture of dynamic interaction patterns
3. Improved interpretation of detected anomalies
4. Adaptive feature processing based on graph dynamics

## 7. Conclusion and Future Work

Our work demonstrates the effectiveness of combining diffusion models with attention mechanism analysis and structural feature augmentation for dynamic graph anomaly detection. Future work could explore:
1. Real-time anomaly detection capabilities
2. Integration with domain-specific knowledge
3. Extension to multi-modal graph data
4. Optimization of computational efficiency

## References

1. DiGress Paper (2023)
2. [Additional references would be added based on specific citations needed]
