# Fairness and Benchmarking in Graph Neural Networks

This repository contains the code for our project completed as part of the **Graphical Models: Discrete Inference and Learning** course of the **MVA Master's Program** at ENS Paris-Saclay.

**Authors**: Naïl Khelifa & Tom Rossa  
**Course**: Graphical Models – MVA 2024–2025

---

## 📚 Project Description

This project explores the performance and fairness of Graph Neural Networks (GNNs), inspired by the following foundational papers:

- [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982)
- [Towards Fair Graph Neural Networks via Graph Counterfactual](https://arxiv.org/abs/2307.04937)

Our work includes benchmarking several GNN architectures on graph classification tasks and investigating fairness criteria via counterfactual methods. The experiments are performed on standard graph datasets such as IMDB-BINARY, DD, PROTEINS, and MNIST-derived graphs.

---

## 📂 Folder Overview

- **data/**: Contains raw datasets and preprocessing utilities for graph classification tasks.
- **data/datasets/**: Includes scripts to load, format, and manage various datasets used in the experiments.
- **evaluation/**: Provides scripts for evaluating models using techniques like K-fold cross-validation and hyperparameter selection.
- **experiments/**: Stores experiment configurations and results from benchmarking and fairness evaluations.
- **Pipeline/**: Implements the full training and evaluation pipelines used to run model assessments.
- **src/**: Holds the core GNN model definitions, custom layers, pooling mechanisms, and graph-related operations.
- **utils/**: Contains helper functions and utilities for training, evaluation, and logging.


# 🚀 How to Run
You can reproduce the main experiments using the notebook:

- `Training_Main_Exp.ipynb`

For a minimal toy example:

- `toy_examples.ipynb`

You can also run the training scripts from the command line:

`python Pipeline/train.py`

## 📊 Datasets

We use the following datasets:

- IMDB-BINARY
- DD
- PROTEINS
- MNIST (transformed into graphs)

Dataset loaders and preprocessing scripts can be found under `data/datasets/`.

## 🧠 Models
Implemented models include:

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Graph Isomorphism Network (GIN)
- GraphSAGE
- Molecular Fingerprint-inspired networks (baseline)

Models are defined in `src/models.py`, with additional layers in `src/layer.py`.
