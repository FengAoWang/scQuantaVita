<div align="center">

# scQuantaVita ⚛️🧬
### Quantum-boosted Deep Generative Learning for Single-Cell Omics

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-yellow)](https://github.com/FengAoWang/scQuantaVita)

**Unveiling the Energy Landscape of Cell Fate with Statistical Physics & Quantum Computing**

[Introduction](#-introduction) • [Methodology](#-methodology) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Tutorials](#-tutorials) • [Biological Application](#-biological-application-csp-analysis)
</div>

---

## 📖 Introduction

**scQuantaVita** represents a new paradigm in single-cell analysis by forging an integration of **Statistical Physics**, **Quantum Computing**, and **Deep Learning**.

Unlike conventional VAE-based models that rely on simplified Gaussian priors, scQuantaVita embeds the **Boltzmann distribution**—the first principle of statistical physics—at its core. By leveraging a quantum-classical hybrid architecture (Coherent Ising Machine + Deep Neural Networks), it efficiently models the high-dimensional thermodynamic landscape of cellular transcriptomes.

### Key Capabilities:
* **⚛️ Quantum-Ready Architecture:** Implements a hybrid loop using quantum sampling (or classical simulation) to train deep Boltzmann machines.
* **📉 Cellular System Potential (CSP):** A novel, unsupervised metric that quantifies the **thermodynamic stability** of a cell. CSP accurately infers differentiation trajectories and immune activation states without ground-truth labels.
* **🎯 High-Fidelity Integration:** Achieves state-of-the-art performance in batch correction, biological conservation, and cell-type annotation on million-scale datasets.

---

## 🧩 Methodology

scQuantaVita treats each cell as a physical system. It maps gene expression profiles into a discretized latent space governed by an energy function derived from the **Ising Model**.

<div align="center">
  <img src="./figure 1.png" alt="scQuantaVita Framework" width="90%">
  <br>
  <em><b>Figure 1: The scQuantaVita Framework.</b> (a-b) Mapping single-cell gene expression onto a Boltzmann energy space. (c) The quantum-classical hybrid architecture: utilizing a Coherent Ising Machine (CIM) for sampling and an Encoder-Decoder for representation learning. (d) Downstream applications including CSP-based differentiation inference and embedding-based integration.</em>
</div>

---

## ⚡ Installation

scQuantaVita is built on PyTorch and Scanpy.

### Prerequisites
* Python ≥ 3.12
* CUDA-enabled GPU (recommended)

### Install via Git
```bash
git clone [https://github.com/FengAoWang/scQuantaVita.git](https://github.com/FengAoWang/scQuantaVita.git)
cd scQuantaVita
```

## 🚀 Quick Start

Here is a minimal example to train **scQuantaVita** on your single-cell dataset.

```python
import torch
import anndata
from model.QBM_VAE import QBM_VAE

# 1. Setup computation device
# Automatically use CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load your dataset
# Ensure your data is preprocessed (normalized & log-transformed)
adata = anndata.read_h5ad('path/to/your_data.h5ad')

# 3. Initialize scQuantaVita model
model = QBM_VAE(
    latent_dim=256,
    device=device,
    use_norm='layernorm'
)

# 4. Prepare and Train
# 'batch_key' is required for batch effect correction (if applicable)
model.set_adata(adata, batch_key='batch')
model.fit(adata)

# 5. Extract Results (Optional but Recommended)
# Get the latent embedding and Cellular System Potential (CSP)
latent = model.get_latent_representation(adata)
csp = model.get_csp(adata)

adata.obsm['X_scQuantaVita'] = latent
adata.obs['CSP'] = csp

# 6. Save the trained model
torch.save(model.state_dict(), 'scQuantaVita_model.pth')
print("Model saved and CSP scores computed.")
```

## 📚 Tutorials

Comprehensive tutorials and reproduction scripts for all figures in the paper are under active development and will be uploaded soon. 

For a basic usage example, please refer to the **Quick Start** section above.

## 🧬 Biological Application: CSP Analysis

**Cellular System Potential (CSP)** serves as a quantitative proxy for the thermodynamic stability of a cell.

> **Physics Definition:** $CSP \propto Energy$
>
> * **🔵 Low-CSP State:** Represents thermodynamic equilibrium. Indicates a stable, optimized molecular program (e.g., **Terminally Differentiated Cells**).
> * **🔴 High-CSP State:** Represents non-equilibrium. Indicates instability, high developmental potential, or high functional activity (e.g., **Stem/Progenitor Cells** or **Activated Immune Cells**).

### Key Use Cases

* **🌊 Differentiation Inference (Waddington's Landscape)**
    Sort cells by CSP to reconstruct differentiation trajectories. Unlike traditional methods, this approach is **entirely unsupervised** and does not require manual specification of root cells.

* **🩺 Disease State Analysis**
    Identify pathological "high-energy" states that traditional clustering often misses.
    * *Example:* In **Duchenne Muscular Dystrophy (DMD)**, scQuantaVita reveals that phenotypically "quiescent" muscle stem cells actually reside in an unstable, high-CSP state, indicating a loss of regenerative potential.