# Monocular 3D Reconstruction Pipeline 

## Contributors (Group 15)
- **Almira Dyussenova** - Architecture, Core Modeling, & Data Pipeline
- **Livia Zoebeli** - MLOps Infrastructure & DVC
- **Kanhaiya Kanhaiya** - Experiment Tracking & Monitoring
[cite_start]This repository contains an end-to-end MLOps pipeline designed for a reproducible machine learning workflow[cite: 86]. [cite_start]The project focuses on **Monocular 3D Reconstruction**, converting single 2D images into interactive 3D environments using deep learning[cite: 88, 91].

## Project Goal
[cite_start]The primary objective is to implement a scalable ML environment that integrates data/model versioning, experiment tracking, and automated deployment[cite: 86, 87]. This framework enables real-world applications in:
- [cite_start]**VR/AR:** Immersive simulations[cite: 90, 91].
- [cite_start]**Gaming:** Rapid 3D asset generation[cite: 92, 93].
- [cite_start]**Architecture:** Layout visualization from photos[cite: 94].
- [cite_start]**E-commerce:** Digital twins of products[cite: 95].

## Pipeline Stages
[cite_start]The project is divided into four main operational stages[cite: 96, 109]:

1. [cite_start]**Data Preparation:** Data collection (Kaggle), cleaning, and version control using **DVC** to ensure full reproducibility[cite: 97, 98, 99].
2. [cite_start]**Model Development:** Experimentation in **PyTorch** with **Weights & Biases (W&B)** for tracking metrics, logs, and hyperparameters[cite: 100, 101, 102].
3. [cite_start]**CI/CD & Registry:** Storing best artifacts in the W&B Model Registry, with **GitHub Actions** for automation and **Docker** for containerization[cite: 103, 104, 105].
4. [cite_start]**Serving & Monitoring:** Local deployment via **FastAPI** and **Streamlit**, with system monitoring supported by **Grafana**[cite: 106, 107, 108].

## Tech Stack & Tooling
| Tool | Purpose |
| :--- | :--- |
| **PyTorch** | [cite_start]Deep learning framework and pretrained models[cite: 154]. |
| **DVC** | [cite_start]Data and artifact versioning[cite: 154]. |
| **W&B** | [cite_start]Experiment tracking and model registry[cite: 154]. |
| **Docker** | [cite_start]Consistent runtime environments[cite: 154]. |
| **FastAPI** | [cite_start]High-performance backend inference[cite: 154]. |
| **Streamlit** | [cite_start]Interactive user interface[cite: 154]. |

## Getting Started

### 1. Environment Setup
```bash
conda activate machle
pip install -r requirements.txt
