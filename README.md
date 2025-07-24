# 🧠 Credit Card Fraud Detection via Knowledge Graphs & Centrality-Based Features 💳🔍

This repository contains the full implementation of our **bachelor thesis research** on enhancing Credit Card Fraud Detection (CCFD) through **graph-based analysis** and **machine learning**.

By modeling transactions as a **Knowledge Graph**, we extracted **centrality-based features**, including weighted variants, to enrich traditional feature sets. These were then fed into a variety of classifiers, including **XGBoost**, **Bagging**, and others, achieving state-of-the-art performance on a real-world dataset.

---

## 📂 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `thesisTradML.ipynb` | Initial ML experiments without graph-based features (baseline). |
| `Main.py` | Main execution script for graph-enhanced experiments. Change `clfx` parameter on lines 128 & 131 to run different classifiers. |
| `Train_Test.py` | Contains all reusable ML training and evaluation functions. |
| `graph_utils/` | Utility functions for generating Knowledge Graphs, applying Neo4j queries, and computing centrality measures. |
| `data/` | Placeholder for datasets (not included in repo). |
| `results/` | Contains result logs, evaluation metrics, plots, and serialized models. |
| `assets/` | Visual assets such as pipeline diagrams and charts for documentation or publication. |

---

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/dimou-gk/KG-Fraud-Detection.git
   cd KG-Fraud-Detection
