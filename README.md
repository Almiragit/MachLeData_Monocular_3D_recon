# Monocular 3D Reconstruction Pipeline

**Group 15** — Almira Dyussenova · Kanhaiya Kanhaiya · Livia Zoebeli

End-to-end MLOps pipeline: **single RGB image → depth map → interactive 3D point cloud**,
with model fine-tuning, experiment tracking, CI/CD, and drift monitoring.

---

## Architecture

```
DATA                    TRAIN                   REGISTRY & CI/CD        SERVE                   MONITOR
─────────               ─────────               ─────────────────       ─────────               ─────────
NYU Depth V2            DaV2 Hybrid             W&B Registry            FastAPI                 Drift Detection
  │                     frozen DINOv2             │                    /predict, /drift          │
  │                     trainable DPTHead          │                    Streamlit UI            Prometheus
  ▼                     SILogLoss + W&B            │                    + 3D Point Cloud         │
download_nyu             │                        GitHub Actions        │                       Grafana
  │                     train                      │                    docker-compose           │
  ▼                      │                        Lint → Tests          │                       Retrain Loop
prepare_data             ▼                        DVC Check              ▼                       │
  │                    evaluate                    Docker Build          User uploads photo        │
  ▼                      │                                                 │                     │
train ─────────────────► │                        │                       │                     │
  │                      ▼                        │                       ▼                     │
  │ ← ─ ─ ─ ─ ─ ─ ─ ─ best_model.pth ─ ─ ─ ─ ─► push_registry          User sees depth + 3D ──► Drift check
  │                                                                                               │
  └── compute_baseline (drift reference from val split) ──────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python **3.11**
- Docker Desktop (Docker Engine + Docker Compose)
- DVC installed (`pip install dvc`)
- Optional but recommended: Conda/Mamba env

```bash
# 1. Clone and activate environment
git clone <repo-url>
cd <project-folder>
conda activate <your-env-name>
pip install -r requirements.txt

# 2. Login to W&B (for experiment tracking)
wandb login

# 3. Download NYU Depth V2 dataset + prepare .pt files
dvc repro download_nyu prepare_data

# 4. Fine-tune DaV2 Hybrid Decoder (10 epochs, frozen DINOv2)
dvc repro train

# 5. Compute drift baseline from validation split
dvc repro compute_baseline

# 6. Evaluate best checkpoint
dvc repro evaluate

# 7. Push to W&B Registry
dvc repro push_registry

# 8. (Optional) Run retrain trigger once
python src/training/retrain_trigger.py --once

# Or run everything:
dvc repro
```

### Windows (cmd) quick demo start

```bat
docker compose up -d --build
docker compose ps
start http://localhost:8501
start http://localhost:8000/docs
start http://localhost:9090
start http://localhost:3000
```

---

## Pipeline Stages (6 Stages)

| Stage | Command | Description |
|---|---|---|
| `download_nyu` | `dvc repro download_nyu` | Download NYU Depth V2 dataset (.mat file) |
| `prepare_data` | `dvc repro prepare_data` | Convert .mat → .pt files, split train/val/test |
| `train` | `dvc repro train` | Fine-tune DPTHead decoder (frozen DINOv2) + W&B logging |
| `compute_baseline` | `dvc repro compute_baseline` | Compute drift reference stats from val split |
| `evaluate` | `dvc repro evaluate` | Evaluate best model on test split (MAE, RMSE, δ₁-δ₃) |
| `push_registry` | `dvc repro push_registry` | Push checkpoint to W&B Model Registry |

---

## Model Architecture: DaV2 Hybrid

**Depth-Anything-V2 (ViT-B)** with **Frozen DINOv2 Encoder** + **Trainable DPTHead Decoder**:

| Component | Parameters | Trainable |
|---|---|---|
| DINOv2 Encoder (ViT-B) | ~300M | ❌ Frozen |
| DPTHead Decoder | ~5M | ✅ Yes |
| **Total** | **~305M** | **~5M trainable** |

- **Loss:** SILogLoss (Scale-Invariant Logarithmic Loss) — standard for monocular depth
- **Optimizer:** Adam (lr=1e-4)
- **Tracking:** W&B logs batch_loss, epoch_loss, val_loss per epoch
- **Checkpointing:** Best model saved when val_loss improves

---

## Serving Stack

| Service | URL | Purpose |
|---|---|---|
| **FastAPI** | http://localhost:8000 | Inference: `/predict`, `/predict/json`, `/health`, `/drift`, `/metrics` |
| **Streamlit** | http://localhost:8501 | Upload UI + Depth map + Interactive 3D Point Cloud (Plotly) |
| **Prometheus** | http://localhost:9090 | Metrics scraping (8 drift gauges) |
| **Grafana** | http://localhost:3000 | Drift dashboard (admin / machle2025) |

```bash
# Start all services
docker compose up -d --build

# Or run individually (development)
uvicorn app.api.main:app --reload --port 8000
streamlit run app/frontend/app.py
```

---

## Monitoring & Drift Detection

### How it works:
1. **Training** → `compute_baseline.py` extracts image statistics from val split (brightness, contrast, blur, depth stats)
2. **FastAPI startup** → loads `baseline.json` as reference distribution
3. **Each prediction** → extracts stats from incoming image → updates rolling window (100 samples)
4. **Background thread (60s interval)** → computes PSI scores + KS-test against reference
5. **Prometheus** scrapes 8 drift metrics from `/metrics`
6. **Grafana** visualises: PSI Brightness, Contrast, Blur (gauges), Alert Status, Depth Mean, Invalid Ratio

### Alert thresholds:
| Metric | Warning (PSI > 0.10) | Alert (PSI > 0.25) |
|---|---|---|
| Brightness | 🟡 Monitor | 🔴 Trigger |
| Contrast | 🟡 Monitor | 🔴 Trigger |
| Blur | 🟡 Monitor | 🔴 Trigger |
| Invalid Depth Ratio | — | 🔴 > 0.30 |

### Retraining loop:
```
Grafana Alert → Prometheus → retrain_trigger.py → dvc repro → new outputs
```

### Retrain trigger usage
```bash
# One-shot check (CI/cron/manual validation)
python src/training/retrain_trigger.py --once

# Continuous loop (default check interval: 300s)
python src/training/retrain_trigger.py

# Dry-run simulation (no real dvc repro), force alert for demo/testing
python src/training/retrain_trigger.py --once --dry-run --force-alert --alerts-needed 1

# Auto-loop simulation (demonstrates automatic trigger path every few seconds)
python src/training/retrain_trigger.py --dry-run --force-alert --alerts-needed 3 --interval 5
```

Requirements:
- Prometheus reachable at `http://localhost:9090`
- Drift metric `drift_alert_triggered` available via Prometheus scrape of `/metrics`

---

## CI/CD Pipeline (GitHub Actions)

| Job | Runs on | What it does |
|---|---|---|
| `lint` | push to main/dev | Flake8 + YAML config validation |
| `test` | push to main/dev | 17 pytest tests: unit + smoke/integration (API + retrain trigger) |
| `dvc-check` | push to main (or manual) | `dvc dag` + pipeline syntax check |
| `docker-build` | push to main | Build + push Docker image to Docker Hub |

---

## Demo Runbook (Presentation-Ready)

### 1) Validate code quality quickly
```bash
python -m pytest tests -q
```

### 2) Start full serving/monitoring stack
```bash
docker compose up -d --build
docker compose ps
```

Open:
- Streamlit: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### 3) Build drift baseline (fast debug run)
```bash
python src/monitoring/compute_baseline.py --debug
```

### 4) Show retrain trigger behavior without expensive retraining

Healthy check:
```bash
python src/training/retrain_trigger.py --once
```

Forced trigger path (safe simulation):
```bash
python src/training/retrain_trigger.py --once --dry-run --force-alert --alerts-needed 1
```

### 5) (Optional) Show automatic trigger loop behavior
```bash
python src/training/retrain_trigger.py --dry-run --force-alert --alerts-needed 3 --interval 5
```

Stop with `Ctrl + C`.

---

## Project Structure

```
MachLeData/
├── src/
│   ├── data/
│   │   ├── downloadNYU.py         # NYU Depth V2 download
│   │   ├── prepare_split_data.py  # .mat → .pt conversion
│   │   └── dataset.py             # PyTorch Dataset
│   ├── models/
│   │   ├── model.py               # DepthAnythingWrapper + HybridDepthModel
│   │   └── losses.py              # SILogLoss, BerHuLoss
│   ├── training/
│   │   ├── train.py               # Fine-tuning loop + W&B
│   │   ├── evaluate.py            # Evaluation metrics (MAE, RMSE, δ₁-δ₃)
│   │   └── retrain_trigger.py     # Automated retraining on drift
│   ├── registry/
│   │   └── push_model.py          # W&B Model Registry push
│   ├── monitoring/
│   │   └── compute_baseline.py    # Drift reference statistics
│   └── utils.py                   # Shared helpers
├── app/
│   ├── api/main.py                # FastAPI: /predict, /drift, /health
│   ├── frontend/app.py            # Streamlit: upload + 3D visualisation
│   └── monitoring/
│       ├── drift_detector.py      # PSI + KS drift detection
│       ├── prometheus.yml
│       └── grafana/provisioning/  # Auto-provisioned dashboards
├── configs/
│   ├── paths.yaml                 # Data paths
│   └── train.yaml                 # Training hyperparameters
├── tests/
│   ├── test_basic.py              # unit/config/loss/drift tests
│   └── test_integration_smoke.py  # smoke tests for API + retrain trigger
├── .github/workflows/ci.yml       # CI/CD: lint → test → dvc → docker
├── Dockerfile + docker-compose.yml
├── dvc.yaml                       # 6-stage DVC pipeline
└── requirements.txt
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Depth-Anything-V2 (ViT-B)** | Pretrained depth estimation model (frozen DINOv2) |
| **PyTorch** | Fine-tuning + DataLoaders |
| **DVC** | Data + pipeline versioning |
| **W&B** | Experiment tracking + Model Registry |
| **GitHub Actions** | CI/CD: lint, test, dvc check, docker build |
| **Docker / Compose** | Containerised serving stack |
| **FastAPI** | Inference API with Prometheus metrics |
| **Streamlit** | Web frontend + 3D Plotly visualisation |
| **Prometheus + Grafana** | Drift monitoring + alerting |
| **pytest** | Unit tests (loss functions, drift, configs) |

---

## Troubleshooting (quick)

If a service does not load, run:

```bash
docker compose ps
docker compose logs --tail=120 api frontend
```

Common checks:
- API should be `healthy`
- Frontend should be `Up`
- `http://localhost:8000/health` should return status `ok`