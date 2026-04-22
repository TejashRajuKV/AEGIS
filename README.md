# AEGIS - AI Fairness Audit System V6 Ultimate

> *"Every team here will tell you AI is biased. We are the only team that fixes it — live, in real time, without touching the model, for any AI system on earth."*

## What is AEGIS?

AEGIS is a production-grade AI fairness audit system that detects, measures, and automatically mitigates bias in machine learning models. It combines cutting-edge causal discovery, reinforcement learning, neural counterfactual generation, and real-time drift monitoring into a single deployable system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AEGIS Backend (FastAPI)                   │
├────────────┬────────────┬────────────┬────────────┬────────────┤
│   Causal   │  Fairness  │    RL      │   Drift    │   Text     │
│ Discovery  │  Audit     │  Autopilot │  Monitor   │   Bias     │
│            │            │            │            │            │
│ DAG-GNN    │ Demo Par.  │    PPO     │   CUSUM    │ Embedding  │
│ PC-Algo    │ Eq. Odds   │  Pareto    │ Wasserst.  │ Cosine     │
│ Proxy      │ Calibrat.  │  Goodhart  │  Ensemble  │ StereoSet  │
│ Scoring    │ Subgroup   │  Shaped    │  Temporal  │ Prompt     │
├────────────┴────────────┴────────────┴────────────┴────────────┤
│                     Neural / Counterfactual                      │
│           Conditional VAE + Latent Interpolation                 │
├─────────────────────────────────────────────────────────────────┤
│                     Services & Pipelines                         │
│  Auto-Fix Generator │ Model Registry │ Task Queue │ WebSocket  │
├─────────────────────────────────────────────────────────────────┤
│                  Model-Agnostic Wrappers                         │
│    sklearn │ xgboost │ pytorch │ tensorflow │ lightgbm          │
└─────────────────────────────────────────────────────────────────┘
```

## Five Key Differentiators

### 1. DAG-GNN for Causal Discovery
Learns causal structure from raw data using Directed Acyclic Graph Graph Neural Networks. Discovers proxy chains in high-dimensional datasets that standard PC-algorithm completely misses. No assumptions about functional form.

### 2. Multi-Modal Bias Coverage
Extends fairness auditing from traditional tabular ML to Large Language Models. Runs demographic-framed prompts through embedding layers and measures cosine distance in latent space. The only tool that audits both traditional ML models AND LLMs.

### 3. PPO with Custom Shaped Reward
Uses Proximal Policy Optimization with continuous action space for fairness autopilot. The shaped reward combines accuracy + three fairness metrics simultaneously with Pareto-domination checks to avoid Goodhart's Law.

### 4. CUSUM + Wasserstein Ensemble
CUSUM catches mean drift. Wasserstein-1 distance catches distribution shift including variance and tail changes. Using both together means near-zero false negatives for drift detection.

### 5. Auto-Fix Code Generator
When AEGIS finds bias, it generates actual Python code to fix it. "Add these 3 lines to your preprocessing pipeline and your equalized odds gap drops from 28% to 6%."

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and run all services
docker-compose up --build

# Backend available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: Direct Python

```bash
cd aegis-backend

# Install dependencies
pip install -r requirements.txt

# Set up datasets
python ../aegis-dataset-setup.py

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Make

```bash
make install    # Install dependencies
make setup      # Download and prepare datasets
make run        # Start the development server
make test       # Run the test suite
```

## API Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Health** | `GET /health` | System health check |
| **Datasets** | `GET /api/datasets/list` | List available datasets |
| | `POST /api/datasets/load` | Load a dataset |
| | `GET /api/datasets/{name}/schema` | Get dataset schema |
| **Models** | `POST /api/models/register` | Register a model |
| | `GET /api/models/list` | List registered models |
| **Fairness** | `POST /api/fairness/audit` | Run fairness audit |
| | `GET /api/fairness/metrics` | Get fairness metrics |
| **Causal** | `POST /api/causal/discover` | Run causal discovery |
| | `POST /api/causal/proxy-chains` | Detect proxy chains |
| **Drift** | `POST /api/drift/monitor` | Monitor data drift |
| | `GET /api/drift/alerts` | Get drift alerts |
| **RL Autopilot** | `POST /api/autopilot/start` | Start autopilot |
| | `GET /api/autopilot/status/{task_id}` | Check task status |
| | `GET /api/autopilot/results/{task_id}` | Get task results |
| **Counterfactual** | `POST /api/counterfactual/generate` | Generate counterfactuals |
| | `POST /api/counterfactual/interpolate` | Interpolate in latent space |
| **Text Bias** | `POST /api/text-bias/audit` | Audit text model bias |
| **Auto-Fix** | `POST /api/code-fix/generate` | Generate bias fix code |
| **WebSocket** | `WS /ws/{session_id}` | Real-time audit streaming |

## Performance Numbers

On the Adult Census Income dataset:
- **Baseline model:** 34% demographic parity gap
- **After AEGIS RL autopilot:** 6% gap
- **Accuracy cost:** 1.8%

On the COMPAS recidivism dataset:
- **False positive rate gap** between Black and White defendants: 21% → 4%
- **Accuracy cost:** 2.1%

## Project Structure

```
aegis-backend/
├── app/
│   ├── api/                  # FastAPI routes (12 route modules)
│   ├── data/                 # Dataset loading, preprocessing, schemas
│   ├── ml/
│   │   ├── causal/           # DAG-GNN, PC Algorithm, proxy detection
│   │   ├── drift/            # CUSUM, Wasserstein, ensemble detectors
│   │   ├── fairness/         # Demographic parity, equalized odds, calibration
│   │   ├── gnn/              # Graph neural network layers & trainers
│   │   ├── neural/           # Conditional VAE, counterfactual generation
│   │   ├── rl/               # PPO agent, Pareto reward, Goodhart guard
│   │   └── text_bias/        # LLM embedding bias auditor
│   ├── models/               # Database models, Pydantic schemas
│   ├── pipeline/             # Orchestration pipelines
│   ├── services/             # Business logic services
│   │   └── wrappers/         # Model-agnostic wrappers (sklearn, xgboost, pytorch, tf)
│   ├── utils/                # Math utilities, validation, logging
│   ├── config.py             # 140+ configuration settings
│   ├── main.py               # FastAPI app with create_app() factory
│   └── exceptions.py         # 15 domain-specific exceptions
├── tests/                    # Comprehensive test suite
├── scripts/                  # Standalone runners
├── notebooks/                # 6 Jupyter experiment notebooks
├── requirements.txt          # Pinned dependencies
└── Dockerfile.backend        # Production Docker image

aegis-shared/
├── configs/                  # Dataset schemas, ML hyperparameters
└── datasets/                 # CSV datasets + download scripts
```

## Supported Datasets

- **Adult Census Income** - Income prediction with gender/race bias
- **COMPAS Recidivism** - Criminal justice with racial bias
- **German Credit** - Credit scoring with age/gender bias
- **Synthetic** - Generated data for testing

## Supported Models

AEGIS is model-agnostic through its wrapper system:
- scikit-learn (RandomForest, LogisticRegression, GradientBoosting, etc.)
- XGBoost
- LightGBM
- PyTorch models
- TensorFlow/Keras models

## Team Division (4-Person)

| Person | Role | Modules |
|--------|------|---------|
| A | ML Core | Causal graph, RL agent, text bias scanner |
| B | Neural Networks | CVAE, GNN trainer, temporal drift |
| C | Systems | FastAPI server, WebSocket, model wrappers, LLM API |
| D | Interface | Dashboard, heatmap, Pareto navigator, demo |

## Tech Stack

- **Backend:** FastAPI, Python 3.11+, SQLAlchemy (async), Pydantic V2
- **ML:** PyTorch, scikit-learn, XGBoost, NetworkX
- **NLP:** Sentence-Transformers, HuggingFace Transformers
- **Deployment:** Docker, docker-compose
- **Testing:** pytest, pytest-asyncio
