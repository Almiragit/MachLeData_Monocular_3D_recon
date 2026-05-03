# Monocular 3D Reconstruction Pipeline 

## Contributors (Group 15)
- **Almira Dyussenova** 
- **Kanhaiya Kanhaiya**
- **Livia Zoebeli**


This repository contains an end-to-end MLOps pipeline designed for a reproducible machine learning workflow. The project focuses on **Monocular 3D Reconstruction**, converting single 2D images into interactive 3D environments using deep learning.

## Project Goal
The primary objective is to implement a scalable ML environment that integrates data/model versioning, experiment tracking, and automated deployment. This framework enables real-world applications in:
- **VR/AR:** Immersive simulations and environments.
- **Gaming:** Rapid 3D asset generation from images.
- **Architecture:** Layout visualization and planning from photos.
- **E-commerce:** Generating digital twins of products.

##  Pipeline Stages
The project is divided into four main operational stages:

1. **Data Preparation:** Data collection (Kaggle), cleaning, and version control using **DVC** to ensure full reproducibility.
2. **Model Development:** Experimentation in **PyTorch** with **Weights & Biases (W&B)** for tracking metrics, logs, and hyperparameters.
3. **CI/CD & Registry:** Storing best artifacts in the W&B Model Registry, with **GitHub Actions** for automation and **Docker** for containerization.
4. **Serving & Monitoring:** Local deployment via **FastAPI** and **Streamlit**, with system monitoring supported by **Grafana**.

##  Tech Stack & Tooling
| Tool | Purpose |
| :--- | :--- |
| **PyTorch** | Deep learning framework and pretrained models |
| **DVC** | Data and artifact versioning |
| **W&B** | Experiment tracking and model registry |
| **Docker** | Consistent runtime environments |
| **FastAPI** | High-performance backend inference |
| **Streamlit** | Interactive user interface |

##  Getting Started

### 1. Environment Setup
Clone the repository and activate the environment:
```bash
conda activate machle
pip install -r requirements.txt
run notebooks/DaV2_PtCloud.ipynb file 


