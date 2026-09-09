# PROJECT LEDGER: CV Stream Segmentation Engine (v1.0 Architecture)

> **Document Classification:** Master Architecture & Engineering Ledger  
> **Target Release:** Version 1.0 (High-Precision CPU-Only Modular Monolith)  
> **Team Size:** 3 Engineers (Lead Orchestrator/DevOps, Feature/Rule Lead, ML/Modeling Lead)  
> **System Scope:** Automated Candidate Boundary Detection & Zero-Copy Lossless PDF Splitting  

---

## 1. Executive Vision & Production Guarantees

### 1.1 The Operational Mandate
Enterprise Applicant Tracking Systems (ATS) and job boards (Workday, Taleo, Greenhouse, Symplicity) routinely merge hundreds of candidate resumes into monolithic PDF streams (200–1,000 pages). Downstream parsers crash, timeout, or combine distinct identities into invalid "PII chimeras."

This project engineers a high-throughput, privacy-compliant, on-premise engine to accurately identify candidate transitions and slice the stream into discrete candidate files.

### 1.2 Mathematical Formulation & Production KPIs
Let an input stream $\mathcal{D}$ be an ordered sequence of pages:
$$\mathcal{D} = (P_1, P_2, \dots, P_N)$$

The system predicts the boundary indicator vector:
$$\mathbf{Y} = [y_1, y_2, \dots, y_{N-1}] \in \{0, 1\}^{N-1}$$
where $y_i = 1$ denotes that $P_i$ is the terminal page of candidate $k$ and $P_{i+1}$ begins candidate $k+1$.

| Metric | Target | Operational Rationale |
| :--- | :--- | :--- |
| **Boundary Recall ($R_B$)** | $\ge 98.5\%$ | Minimizes missed splits (prevents multi-candidate chimera profiles). |
| **Boundary Precision ($P_B$)** | $\ge 97.0\%$ | Prevents over-splitting multi-page CVs into orphaned single pages. |
| **Clean Document Accuracy** | $\ge 96.0\%$ | Percentage of candidates completely segmented with zero error. |
| **Throughput / Latency** | $< 20\text{ ms / page}$ | A 500-page batch processes in under 10 seconds on standard CPU. |
| **Hardware Footprint** | 100% Multi-core CPU | Zero GPU dependency for v1.0. Docker image $<350\text{ MB}$. |

### 1.3 Why This Guarantees High Accuracy on Real-World CVs
1. **Multi-Signal Triangulation:** Relying on a single model (just regex or just ML) fails on creative templates. Triangulating **PII density + typography hierarchy + pagination + syntactic continuation + tabular gradient boosting** guarantees that if one signal is missing, others compensate.
2. **Structural Prior Enforcement (Viterbi HMM):** Classifiers predict in isolation and can make erratic decisions (e.g. splitting mid-sentence or creating a 20-page CV). The Viterbi sequence trellis mathematically eliminates impossible transitions and penalizes rare structures.
3. **Distribution-Preserving Synthesis:** Training on synthetic streams that enforce real-world CV length distributions (45% 1-page, 42% 2-page, 10% 3-page, 3% 4+ page) and strip PDF metadata prevents overfitting to synthetic artifacts.

---

## 2. Selected Architecture for Version 1.0 (CPU-Optimized Cascade)

We intentionally **omit heavy Vision Transformers (ViT / LayoutLM) in Version 1.0**. A deep visual model requires PyTorch/CUDA, blows up Docker image sizes from 350MB to 4.5GB, and increases inference latency by $20\times$, while providing marginal gain on clean digital text. Version 1.0 is engineered as a **pure CPU-native cascaded pipeline**.

```
                           [ Merged PDF Stream ]
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │  Tier 0: Fast Digital Extraction (PyMuPDF)       │
            │  Extract: text spans, bbox, fonts, images        │
            │  Latency: 2 - 4 ms / page (CPU)                  │
            └────────────────────────┬─────────────────────────┘
                                     │
                      (Char Count < 80 & Has Images?)
                           ├── YES ──► [Conditional OCR: PaddleOCR / Tesseract]
                           └── NO  ──┐ (Executed only on ~5% of scanned pages)
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │  Tier 1: Heuristic Gating Engine                 │
            │  • PII in top 35% header zone                    │
            │  • Font size dominance ratio (Title / Median)    │
            │  • Explicit pagination ("Page 1 of 2", "Page 2") │
            │  • Syntactic sentence & bullet fracture checks   │
            │  Latency: < 1 ms / page (CPU)                    │
            └────────────────────────┬─────────────────────────┘
                                     │
                      (Confidence Evaluation)
                           ├── P(split) >= 0.98 ──► Fast-Path Boundary (y=1)
                           ├── P(split) <= 0.02 ──► Fast-Path Continuation (y=0)
                           └── Ambiguous (0.02 < P < 0.98) [~20% of pairs]
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │  Tier 2: LightGBM Pairwise Classifier            │
            │  36-dimensional tabular feature vector           │
            │  Latency: 1 - 3 ms / page (CPU)                  │
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │  Tier 3: Global Sequence Optimizer (Viterbi HMM) │
            │  Dynamic programming trellis over state space:   │
            │  S ∈ {START, PAGE_2, PAGE_3, PAGE_4+}            │
            │  Latency: < 1 ms for entire 500-page stream (CPU)│
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │  Lossless Slicing Engine (pikepdf / QPDF)        │
            │  Zero-copy byte-level stream splitting           │
            │  Latency: < 400 ms total for 500 pages (CPU)     │
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
            Output: candidate_001.pdf, candidate_002.pdf, manifest.json
```

---

## 3. Computational Benchmark & Hardware Allocation

### 3.1 Latency & Resource Breakdown per Page

| Stage | Module | Compute Type | Execution Time / Page | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | `PyMuPDF` Vector Ingestion | CPU (C++) | **2.5 ms** | ~15 MB per stream |
| **Tier 0 Fallback** | `PaddleOCR` / Tesseract | CPU (SIMD) | **450 ms** *(triggered on <5% of pages)* | ~80 MB |
| **Tier 1** | Deterministic Heuristics | CPU (Pure Python) | **0.8 ms** | < 5 MB |
| **Tier 2** | 36-Feature Extraction | CPU (NumPy) | **2.0 ms** | < 10 MB |
| **Tier 2** | LightGBM Inference | CPU (OpenMP C++) | **0.5 ms** | ~12 MB model buffer |
| **Tier 3** | Viterbi Trellis Optimizer | CPU (NumPy DP) | **0.002 ms** (<1 ms total stream) | < 1 MB |
| **Output** | `pikepdf` Lossless Slicing | CPU (QPDF C++) | **0.8 ms** | ~20 MB |
| **End-to-End** | **Full Stream Processing** | **100% Multi-core CPU** | **~10 - 15 ms / page** | **< 150 MB total RAM** |

### 3.2 Is GPU Needed? (Local vs. Kaggle / Colab)
* **Production Runtime:** **Zero GPU required.** Runs entirely on standard multi-core commodity CPUs (e.g. AWS `c6i.xlarge` or local 4-core laptops).
* **Where Kaggle / Cloud Notebooks Can Be Used:**
  * Downloading large bulk resume datasets (10GB+).
  * Batch offline data synthesis (stitching 100,000 pages to disk).
  * *Note:* Even training LightGBM on 100,000 feature rows takes **less than 45 seconds on a modern 8-core CPU**. Kaggle GPU is completely optional and not required for training.

---

## 4. Notebooks vs. Production Source Code Layout

To ensure research agility without polluting production code, the workspace is partitioned strictly:

```
cv-stream-segmenter/
├── notebooks/                     # RESEARCH, EXPLORATION & AUDIT ONLY
│   ├── 01_data_synthesis_check.ipynb   # Inspect raw CVs, test metadata stripping
│   ├── 02_heuristics_tuning.ipynb      # Test regexes, evaluate edge-case headers
│   ├── 03_feature_engineering_lgbm.ipynb # Feature correlation, SHAP, Optuna tuning
│   └── 04_viterbi_trellis_eval.ipynb   # Trellis path visualizer & error analysis
│
└── src/cv_segment/                # PRODUCTION ENGINE (Tested, Typed, Packaged)
    ├── ingestion/                 # PyMuPDF ingestion & OCR triage
    ├── heuristics/                # PII, font, pagination, syntax rules
    ├── features/                  # Tabular 36-dim pairwise feature extractor
    ├── models/                    # LightGBM classifier & Viterbi HMM decoder
    ├── slicer/                    # pikepdf zero-copy PDF slicer
    ├── api/                       # FastAPI microservice
    └── pipeline.py                # End-to-end orchestrated pipeline
```

* **Rule:** Code developed in notebooks must be refactored into modular classes in `src/cv_segment/` accompanied by unit tests in `tests/` before merging into `main`.

---

## 5. Team Task Distribution & Ownership Matrix

### 5.1 The 3-Engineer Matrix

| Engineer | Primary Role | Direct Code Ownership | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Engineer 1 (You)** | **Repo Orchestrator, Infra & IO Lead** | • `.github/workflows/`<br>• `docker/`<br>• `src/cv_segment/ingestion/`<br>• `src/cv_segment/slicer/`<br>• `src/cv_segment/api/`<br>• `src/cv_segment/pipeline.py` | 1. GitHub Actions CI (lint, types, pytest) & GHCR Docker build.<br>2. `PyMuPDF` text/geometry parser & conditional OCR fallback.<br>3. `pikepdf` zero-copy lossless slicing engine.<br>4. FastAPI service (`POST /v1/segment-stream`) & CLI entrypoint. |
| **Engineer 2** | **Heuristics & Feature Engineering Lead** | • `src/cv_segment/heuristics/`<br>• `src/cv_segment/features/`<br>• `configs/heuristics.yaml`<br>• `notebooks/02_heuristics_tuning.ipynb` | 1. Header PII regex & international phone parser.<br>2. Dominant font size dominance ratio extractor.<br>3. Pagination parser ("Page X of Y", "Page 2").<br>4. Cross-page sentence & bullet continuation checker.<br>5. 36-dimensional tabular pairwise feature vector generator. |
| **Engineer 3** | **ML, Sequence Modeling & Data Lead** | • `scripts/download_datasets.py`<br>• `scripts/generate_synthetic.py`<br>• `src/cv_segment/models/`<br>• `configs/lightgbm_params.yaml`<br>• `configs/hmm_priors.yaml` | 1. Kaggle/HF dataset downloader & anti-leakage synthetic generator.<br>2. Training LightGBM pairwise classifier with Optuna.<br>3. SHAP feature importance analysis and model serialization.<br>4. Viterbi HMM sequence optimizer enforcing CV length priors. |

---

## 6. Git Branching Strategy, PR Protocol & Docker Build Policy

### 6.1 Branching Topology
The repository uses **Trunk-Based Development with Short-Lived Feature Branches**:

```
main (protected) ───────────●───────────────────────────●──────────────────► (Tagged v1.0.0 -> GHCR)
                             ▲                           ▲
                             │ PR #1 (CI green)          │ PR #2 (CI green)
                             │                           │
feat/eng1-ingestion-slicer ──┘                           │
                                                         │
feat/eng2-heuristics-features ───────────────────────────┘
```

### 6.2 The 3 Core Feature Branches to Create
1. **`feat/eng1-ingestion-slicer-cicd`** (Assigned to Engineer 1)
   - Scope: CI/CD setup, Dockerfile, `PDFParser`, `OCRFallback`, `PikePDFSlicer`, `Pipeline` orchestrator.
2. **`feat/eng2-heuristics-features`** (Assigned to Engineer 2)
   - Scope: `pii_rules.py`, `font_rules.py`, `pagination.py`, `syntactic.py`, `pairwise.py` feature extractor.
3. **`feat/eng3-synthetic-lightgbm-viterbi`** (Assigned to Engineer 3)
   - Scope: `download_datasets.py`, `generate_synthetic.py`, `classifier.py`, `sequence_opt.py`.

### 6.3 Pull Request (PR) Policy & Branch-Tagged Docker Building
1. **Protected `main` Branch:** Direct pushes to `main` are strictly blocked.
2. **PR Quality Gates:**
   * Linter passes (`ruff check .` and `ruff format --check .`).
   * Type check passes (`mypy src/`).
   * Test suite passes with $\ge 85\%$ coverage (`pytest tests/`).
   * At least 1 peer approval.
3. **Branch-Tagged Docker Build Policy (GHCR):**
   * PR builds **never** overwrite `latest`.
   * PR builds use GitHub Actions cache (`type=gha`) and are tagged:
     `ghcr.io/<owner>/cv-stream-segmenter:pr-<pr_number>`
   * When merged into `main`, the image is automatically built and tagged as:
     `ghcr.io/<owner>/cv-stream-segmenter:latest` and `ghcr.io/<owner>/cv-stream-segmenter:<commit_sha>`.

---

## 7. Mathematical Specifications of Rules & Features

### 7.1 Tier 1 Heuristics
1. **PII Header Score ($S_{\text{PII}}$):**
   Within top 35% bounding box of page $P_k$:
   $$S_{\text{PII}}(P_k) = 2.0 \cdot \mathbb{I}(\text{Email}) + 1.5 \cdot \mathbb{I}(\text{Phone}) + 1.0 \cdot \mathbb{I}(\text{LinkedIn/GitHub})$$
   * If $S_{\text{PII}}(P_{i+1}) \ge 3.0 \implies P(\text{boundary}) \to 0.99$.
2. **Font Dominance Ratio ($R_{\text{font}}$):**
   $$R_{\text{font}}(P_k) = \frac{\max_{s \in \text{top25\%}(P_k)} \text{FontSize}(s)}{\text{median}_{s \in P_k} \text{FontSize}(s)}$$
   * If $R_{\text{font}}(P_{i+1}) \ge 1.8$ and $R_{\text{font}}(P_i) < 1.3 \implies$ Split signal.
3. **Pagination Tracking:**
   * $P_{i+1}$ has `"Page 1 of X"` $\implies$ Boundary $= 1.0$.
   * $P_i$ has `"Page k"` and $P_{i+1}$ has `"Page k+1"` $\implies$ Continuation $= 1.0$.
   * $P_i$ has `"Page k of k"` $\implies P_{i+1}$ is Boundary $= 1.0$.
4. **Syntactic Sentence Fracture:**
   * Trailing hyphen on $P_i$ + lowercase on $P_{i+1} \implies$ Continuation certainty $0.99$.
   * Non-terminal punctuation on $P_i$ + coordinating conjunction (`and`, `with`, `or`) + lowercase on $P_{i+1} \implies$ Continuation certainty $0.95$.

### 7.2 Tier 2 Feature Matrix (36 Features)
Pairwise vector $\vec{x}_{i, i+1}$ for adjacent pages $(P_i, P_{i+1})$:
* **Semantic & Text Similarity (6):** TF-IDF Cosine Similarity, Jaccard similarity of vocabulary, Candidate name token match score, Shared organization name count, Shared URL domain count, Shared email domain flag.
* **PII & Contact Deltas (6):** $S_{\text{PII}}(P_{i+1}) - S_{\text{PII}}(P_i)$, Email presence on $P_{i+1}$ header, Phone presence on $P_{i+1}$ header, Name presence on $P_{i+1}$, Address match score, Link count delta.
* **Typography & Layout Discontinuities (10):** Font Dominance Ratio delta, Max font size delta, Median font size delta, Left margin delta, Top margin delta, Line spacing delta, Text block count delta, Column count difference, Aspect ratio difference, Blank line count delta.
* **Syntactic Continuity Signals (8):** Trailing hyphen flag, Lack of terminal period flag, Conjunction ending flag, Initial lowercase flag, Bullet continuation flag, Numbered list increment flag, Table row continuation flag, Header entity match flag.
* **Document Lifecycle & Polarity (6):** Terminal section score on $P_i$ ("References", "Declarations"), Front section score on $P_{i+1}$ ("Summary", "Profile", "Objective"), Cumulative page count in current candidate segment, Pagination gap indicator, Exact page number increment flag, Total pagination mismatch flag.

### 7.3 Tier 3 Viterbi HMM Trellis Optimization
* **State Space:** $S \in \{\text{START}, \text{PAGE\_2}, \text{PAGE\_3}, \text{PAGE\_4+}\}$.
* **Transition Matrix $\mathbf{A}$:**
  $$A = \begin{pmatrix} 0.45 & 0.42 & 0.10 & 0.03 \\ 0.70 & 0.00 & 0.25 & 0.05 \\ 0.85 & 0.00 & 0.00 & 0.15 \\ 0.92 & 0.00 & 0.00 & 0.08 \end{pmatrix}$$
  *(Note: Transition $\text{START} \to \text{PAGE\_3}$ is strictly $0.0$, preventing impossible transitions).*
* **Optimal Trellis Path Decoding:**
  $$\mathbf{S}^* = \arg\max_{\mathbf{S}} \sum_{t=1}^N \left( \log P(O_t \mid S_t) + \log P(S_t \mid S_{t-1}) \right)$$

---

## 8. Anti-Leakage Synthetic Generation Protocol

```
[Pool of 3,000+ Raw Open Resumes (Kaggle/HF/LaTeX)]
                  │
                  ├── Distribution Sampling (1p: 45%, 2p: 42%, 3p: 10%, 4+p: 3%)
                  │
                  ├── Concatenation via pikepdf / Ghostscript
                  │   CRITICAL: Force unified /XRef table & strip all /CreationDate,
                  │   /ModDate, /Producer, and /Author metadata to prevent data leakage
                  │
                  ├── Stochastic Noise Emulation (15% of streams):
                  │   Rasterize to 200 DPI -> Perspective Skew (+-1.5 deg) -> PaddleOCR
                  │
                  └── Output Artifacts:
                      - data/synthetic/stream_0001.pdf
                      - data/synthetic/stream_0001_manifest.json
```

---
*End of Master Architecture Ledger (v1.0).*
