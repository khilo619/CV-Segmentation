# CV Stream Segmentation Engine

[![CI Pipeline](https://github.com/khilo619/CV-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/khilo619/CV-Segmentation/actions)
[![Docker Image](https://img.shields.io/badge/GHCR-Docker%20Image-blue)](https://github.com/khilo619/CV-Segmentation/pkgs/container/CV-Segmentation)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://python.org)
[![License: Proprietary / Enterprise](https://img.shields.io/badge/License-Proprietary-red.svg)]()

High-throughput, automated boundary detection and zero-copy lossless splitting engine for bulk-merged resume/CV PDF streams (ATS exports, university career portals, and recruitment databases).

---

## Master Architecture & Engineering Ledger
For the complete technical specification, mathematical formulations, hardware benchmarks, and team task distributions, read:
👉 **[LEDGER.md](LEDGER.md)**

---

## 1. The Problem
Enterprise ATS platforms (Workday, Greenhouse, Taleo, Bullhorn) export candidate pools by concatenating hundreds of multi-page resumes into a single monolithic PDF without boundary markers (e.g., 500 pages). Feeding these streams into automated parsers creates corrupt "PII chimeras" (mixing Candidate A's phone with Candidate B's degrees) or triggers HTTP 413 / timeout crashes.

This engine automatically identifies candidate boundaries and cleanly slices the stream into discrete single-candidate PDF files.

---

## 2. High-Level Architecture (v1.0 CPU-Optimized Cascade)

```
Merged Bulk PDF Stream
         │
         ▼
[Tier 0: Fast Ingestion (PyMuPDF)] ──(No text?)──► [Conditional OCR (PaddleOCR)]
         │
         ▼
[Tier 1: Heuristic Gating] (PII density, font dominance, pagination, sentence fracture)
         ├── High-confidence split (p >= 0.98) ──► Fast-Path Boundary (y=1)
         ├── High-confidence same (p <= 0.02)  ──► Fast-Path Continuation (y=0)
         └── Ambiguous pairs (0.02 < p < 0.98)
                  │
                  ▼
[Tier 2: LightGBM Pairwise Classifier] (36 tabular features, CPU-only, 2ms/page)
                  │
                  ▼
[Tier 3: Global Sequence Optimizer (Viterbi HMM)] (Enforces real CV length priors)
                  │
                  ▼
[Lossless Slicing Engine (pikepdf)] ──► candidate_001.pdf, candidate_002.pdf, manifest.json
```

### Key Performance Targets:
* **Boundary Recall ($R_B$):** $\ge 98.5\%$
* **Boundary Precision ($P_B$):** $\ge 97.0\%$
* **Clean Document Accuracy:** $\ge 96.0\%$
* **Latency:** $< 15\text{ ms / page}$ on 100% standard CPU (a 500-page batch processes in ~7 seconds).

---

## 3. Repository Structure

```
cv-stream-segmenter/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Lint, mypy, pytest with coverage gate
│   │   └── docker-publish.yml     # Multi-stage Docker build & push to GHCR
│   └── pull_request_template.md   # Standardized PR checklist
├── configs/
│   ├── heuristics.yaml            # Rule thresholds and regex patterns
│   ├── lightgbm_params.yaml       # Hyperparameters & feature list
│   └── hmm_priors.yaml            # Transition probabilities
├── docker/
│   ├── Dockerfile                 # Multi-stage production build (CPU-optimized)
│   └── .dockerignore
├── notebooks/                     # Exploratory research & visualization
│   ├── 01_data_synthesis_check.ipynb
│   ├── 02_heuristics_tuning.ipynb
│   ├── 03_feature_engineering_lgbm.ipynb
│   └── 04_viterbi_trellis_eval.ipynb
├── scripts/
│   ├── download_datasets.py       # Fetches open-source Kaggle/HF CVs
│   ├── generate_synthetic.py      # Stitches streams with anti-leakage
│   └── evaluate_benchmark.py      # Computes WindowDiff, P_k, F1 boundary metrics
├── src/
│   └── cv_segment/
│       ├── __init__.py
│       ├── schemas.py             # Shared dataclasses & type definitions
│       ├── ingestion/             # PyMuPDF fast digital extraction & OCR triage
│       ├── heuristics/            # PII, font hierarchy, pagination, syntax rules
│       ├── features/              # 36-dimensional tabular pairwise feature extractor
│       ├── models/                # LightGBM classifier & Viterbi HMM decoder
│       ├── slicer/                # pikepdf zero-copy lossless splitting engine
│       ├── pipeline.py            # Master pipeline orchestrator
│       └── api/                   # FastAPI microservice
├── tests/
│   ├── unit/                      # Unit test suites
│   └── integration/               # End-to-end pipeline tests
├── pyproject.toml                 # Package dependencies and tool configs
└── README.md
```

---

## 4. Team Roles & Branching Strategy

Our 3-engineer team operates on **Trunk-Based Development** with protected `main`:

* **Engineer 1 (Repo Manager / Orchestrator):**
  * Branch: `feat/eng1-ingestion-slicer-cicd`
  * Focus: CI/CD, Docker/GHCR, Ingestion (`PyMuPDF`), Slicer (`pikepdf`), API/CLI.
* **Engineer 2 (Heuristics & Features Lead):**
  * Branch: `feat/eng2-heuristics-features`
  * Focus: Rule gating engine (PII, font, pagination, syntax) & 36 pairwise features.
* **Engineer 3 (ML & Sequence Modeling Lead):**
  * Branch: `feat/eng3-synthetic-lightgbm-viterbi`
  * Focus: Synthetic data generator with anti-leakage, LightGBM training, Viterbi HMM.

---

## 5. Quickstart & Local Development

### 5.1 Installation (Using `uv` or `pip`)
```bash
# Clone the repository
git clone <REMOTE_REPO_URL>
cd CV_Segmentation

# Install dependencies in editable mode with development tools
pip install uv
uv pip install --system -e ".[dev]"
```

### 5.2 Running Quality Checks & Tests
```bash
# Run linter
ruff check .
ruff format --check .

# Run type checker
mypy src/

# Run unit tests
pytest tests/ -v --cov=src
```

### 5.3 Segmenting a PDF via CLI
```bash
python -m cv_segment.pipeline --input sample_stream.pdf --output ./output_candidates/
```

### 5.4 Running the FastAPI Microservice
```bash
uvicorn cv_segment.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---
*Maintained by the AI Document Architecture Team.*
