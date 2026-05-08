# Monocular 3D Reconstruction Pipeline

**Group 15** — Almira Dyussenova · Kanhaiya Kanhaiya · Livia Zoebeli

End-to-end MLOps pipeline converting single RGB images into depth maps and interactive 3D point clouds.

---

## MLOps Pipeline

```
1. DATA PREPARATION     →  2. MODEL DEVELOPMENT      →  3. MODEL REGISTRY & CI/CD    →  4. LOCAL SERVING
   Kaggle + DVC               PyTorch + W&B               W&B Registry                    FastAPI + Streamlit
                              Git (this repo)              GitHub Actions                  Grafana monitoring
                                                           Docker                          
```

### Stage 1 — Data Preparation
| Tool | Role |
|---|---|
| **Kaggle** | Dataset source |
| **DVC** | Data versioning & pipeline reproducibility |

```bash
dvc repro download     # download dataset from Kaggle
dvc repro preprocess   # clean, resize, split (80/10/10)
```

### Stage 2 — Model Development & Experimentation
| Tool | Role |
|---|---|
| **PyTorch** | ResNet encoder + decoder depth model (no pretrained weights) |
| **Git** | Code versioning |
| **W&B** | Experiment tracking: metrics, loss curves, model artifacts |

```bash
# Quick sanity check (3 epochs)
python src/training/train.py --debug

# Full training run
python src/training/train.py

# Evaluate best checkpoint on test set
python src/training/evaluate.py --split test
```

### Stage 3 — Model Registry & CI/CD
| Tool | Role |
|---|---|
| **W&B Model Registry** | Versioned best-model artifacts |
| **GitHub Actions** | Automated lint → train → registry → Docker build |
| **Docker** | Containerised inference server |

```bash
# Push best model to W&B Registry manually
python src/registry/push_model.py

# Build Docker image locally
docker build -t machle-api .
```

**GitHub Secrets required:**
- `WANDB_API_KEY`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### Stage 4 — Local Model Serving & Deployment
| Tool | Role | URL |
|---|---|---|
| **FastAPI** | Inference API (`POST /predict`) | http://localhost:8000 |
| **Streamlit** | Upload UI + 3D visualisation | http://localhost:8501 |
| **Prometheus** | Metrics scraping | http://localhost:9090 |
| **Grafana** | Monitoring dashboard | http://localhost:3000 |

```bash
# Start all services with Docker Compose
WANDB_API_KEY=<your-key> docker-compose up --build

# Or run individually (development)
uvicorn app.api.main:app --reload --port 8000
streamlit run app/frontend/app.py
```

Grafana credentials: `admin` / `machle2025`

---

## Project Structure

```
MachLeData/
├── src/
│   ├── data/
│   │   ├── download.py        # Kaggle download
│   │   ├── preprocess.py      # Clean, resize, split
│   │   └── dataset.py         # PyTorch Dataset & DataLoaders
│   ├── models/
│   │   ├── model.py           # DepthResNet + build_model factory
│   │   └── losses.py          # SILog, BerHu losses
│   ├── training/
│   │   ├── train.py           # Full training loop + W&B
│   │   └── evaluate.py        # Evaluation script
│   ├── registry/
│   │   └── push_model.py      # W&B Model Registry push
│   └── utils.py               # Shared helpers
├── app/
│   ├── api/main.py            # FastAPI inference server
│   ├── frontend/app.py        # Streamlit UI
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/provisioning/
├── configs/
│   ├── paths.yaml             # Data paths
│   └── train.yaml             # Training hyperparameters
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
├── Dockerfile                 # API container
├── docker-compose.yml         # All services
├── dvc.yaml                   # DVC pipeline stages
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
conda activate machle
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env   # add WANDB_API_KEY, Kaggle credentials

# 3. Run full DVC pipeline
dvc repro

# 4. Start serving stack
docker-compose up --build
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **PyTorch** | Depth estimation model (from scratch) |
| **DVC** | Data + artifact versioning |
| **W&B** | Experiment tracking & model registry |
| **GitHub Actions** | CI/CD automation |
| **Docker / Compose** | Containerisation & orchestration |
| **FastAPI** | High-performance inference API |
| **Streamlit** | Interactive web frontend |
| **Prometheus + Grafana** | System & API monitoring |
