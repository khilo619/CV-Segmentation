# Project Backlog: CV Stream Segmentation Engine

> **Document Status:** Active Engineering Backlog  
> **Framework:** Agile / Scrum (5 Sprints $\times$ 2 Weeks)  
> **Team Size:** 3 Engineers (Lead Orchestrator/DevOps, Feature/Rule Lead, ML/Modeling Lead)  
> **Location:** `docs/PROJECT_BACKLOG.md`

---

## 1. Executive Summary & Backlog Overview

### 1.1 Business Context to Technical Translation
* **Raw Business Need:** "Our recruitment platform exports hundreds of job applicant resumes into a single 500-page monolithic PDF. Downstream resume parsers fail with payload timeouts or merge candidates together into fake 'PII chimeras'. We need an automated, fast, on-premise engine to accurately slice the stream into individual candidate CVs."
* **Engineering Solution:** A high-throughput, CPU-optimized cascaded engine combining **PyMuPDF fast digital ingestion**, **deterministic heuristic gating (PII/typography/pagination/syntax)**, **LightGBM pairwise gradient boosting on 36 features**, **Viterbi HMM global sequence optimization**, and **pikepdf zero-copy lossless slicing**.

### 1.2 Definition of Ready (DoR)
A backlog item is ready for sprint planning if:
1. User story is clearly articulated with business rationale.
2. Technical acceptance criteria are defined with testable conditions.
3. Dependencies and external libraries are identified.
4. Story points (Fibonacci: 1, 2, 3, 5, 8) and priority are assigned.

### 1.3 Definition of Done (DoD)
A story is considered complete when:
1. Code is implemented with strict static type hints (`mypy src/ --strict`).
2. Unit and integration tests are written and passing (`pytest tests/`).
3. Code coverage is maintained $\ge 70\%$ (progressing to $\ge 85\%$).
4. Code passes linter and formatter (`ruff check .` and `ruff format --check .`).
5. PR is reviewed, approved by at least 1 peer, and merged into `main`.

---

## 2. Epics Hierarchy

* **EPIC-1: High-Throughput Digital Ingestion & Conditional OCR Triage (Lead: Eng 1)**
* **EPIC-2: Open-Source Data Acquisition & Anti-Leakage Synthetic Generation (Lead: Eng 3)**
* **EPIC-3: Production Heuristic Gating Engine (Tier 1) (Lead: Eng 2)**
* **EPIC-4: 36-Dimensional Feature Engineering & LightGBM Classifier (Tier 2) (Lead: Eng 2 & 3)**
* **EPIC-5: Global Sequence Trellis Optimization (Tier 3 - Viterbi HMM) (Lead: Eng 3 & 1)**
* **EPIC-6: Zero-Copy Lossless PDF Slicing & Production Containerization (Lead: Eng 1)**

---

## 3. Detailed User Stories & Engineering Tasks

### EPIC-1: Ingestion & Conditional OCR

#### Story 1.1: Native PyMuPDF Digital Text & Geometry Extraction
* **User Persona:** As an ML pipeline, I need structured text spans, bounding boxes, and font sizes extracted from digital PDFs in $<5$ ms per page so that downstream models receive rich layout features without rasterization overhead.
* **Acceptance Criteria:**
  - `PDFStreamParser.parse_pdf()` returns a list of `PagePayload` objects.
  - Extracts text spans with $(x_0, y_0, x_1, y_1)$ bounding box, font name, and font size.
  - Correctly calculates printable character count and detects embedded image objects.
  - Execution time is $<4$ ms per page on standard CPU.
* **Story Points:** 3  
* **Priority:** P0 (Blocker)  
* **Assignee:** Engineer 1  

#### Story 1.2: Conditional OCR Triage Subsystem
* **User Persona:** As a system administrator, I want OCR to be triggered strictly on scanned or flatbed resume pages so that processing digital PDFs does not suffer heavy OCR latency penalties.
* **Acceptance Criteria:**
  - Evaluates text character density threshold ($\text{chars} < 80$ AND has embedded raster image).
  - Skips OCR on born-digital pages ($\ge 95\%$ of stream).
  - Routes scanned pages to `PaddleOCR` / `Tesseract` to populate synthetic `TextSpan` objects.
* **Story Points:** 3  
* **Priority:** P1  
* **Assignee:** Engineer 1  

---

### EPIC-2: Data Strategy & Synthetic Stream Generation

#### Story 2.1: Open-Source Resume Corpus Ingestion
* **User Persona:** As an ML engineer, I need a diverse pool of 3,000+ public resumes from Kaggle, Hugging Face, and GitHub LaTeX repositories so that the model generalizes to varied formatting templates.
* **Acceptance Criteria:**
  - `download_datasets.py` script automatically fetches open-source datasets.
  - Categorizes resumes by length (1-page, 2-page, 3-page, 4+ page) and industry domain.
* **Story Points:** 2  
* **Priority:** P0  
* **Assignee:** Engineer 3  

#### Story 2.2: Anti-Leakage Synthetic Stream Generator
* **User Persona:** As an ML engineer, I need programmatically concatenated multi-candidate PDF streams with ground-truth boundary manifests so that we have infinite labeled training data with zero manual annotation.
* **Acceptance Criteria:**
  - Generates multi-document streams of variable length (20 to 200 pages).
  - Enforces realistic CV length distributions: 45% (1-page), 42% (2-page), 10% (3-page), 3% (4+ page).
  - **Anti-Leakage Guarantee:** Re-saves PDFs with unified cross-reference tables and strips `/CreationDate`, `/ModDate`, and `/Producer` to prevent models from learning PDF metadata shortcuts.
  - Outputs paired `ground_truth_manifest.json` with exact split cut indices.
* **Story Points:** 5  
* **Priority:** P0 (Blocker)  
* **Assignee:** Engineer 3  

---

### EPIC-3: Production Heuristic Gating Engine (Tier 1)

#### Story 3.1: Header PII Density Signal
* **User Persona:** As a boundary detector, I want to identify candidate contact credentials in page headers so that new candidate start pages are recognized with high confidence.
* **Acceptance Criteria:**
  - Extracts email (`EMAIL_REGEX`), international phone (`phonenumbers`), LinkedIn, and GitHub links in the top 35% vertical zone of each page.
  - Returns weighted score $S_{\text{PII}}$. When $S_{\text{PII}} \ge 3.0$, emits high boundary confidence ($P \ge 0.98$).
* **Story Points:** 3  
* **Priority:** P0  
* **Assignee:** Engineer 2  

#### Story 3.2: Typography Hierarchy & Dominance Ratio Analyzer
* **User Persona:** As a boundary detector, I want to detect prominent candidate name titles so that typographic hierarchy discontinuities indicate candidate transitions.
* **Acceptance Criteria:**
  - Computes Dominance Ratio $R_{\text{font}} = \max(\text{top } 25\%) / \text{median}(\text{all spans})$.
  - Detects spikes where $R_{\text{font}}(P_{i+1}) \ge 1.8$ and $R_{\text{font}}(P_i) < 1.3$.
* **Story Points:** 2  
* **Priority:** P1  
* **Assignee:** Engineer 2  

#### Story 3.3: Explicit Header/Footer Pagination Tracking
* **User Persona:** As a boundary detector, I want to parse explicit page numbers ("Page 1 of 2", "Page 2", "1 / 3") in top/bottom 10% margins so that explicit pagination provides 100% deterministic ground truth.
* **Acceptance Criteria:**
  - Matches `"Page X of Y"` and solitary margin digits.
  - Emits boundary when next page is `"Page 1"` or previous page was `"Page k of k"`.
  - Emits continuation when next page is strictly $k+1$.
* **Story Points:** 3  
* **Priority:** P0  
* **Assignee:** Engineer 2  

#### Story 3.4: Cross-Page Syntactic Sentence Fracture Analyzer
* **User Persona:** As a boundary detector, I want to check for trailing hyphens and open coordinating conjunctions so that mid-sentence physical page splits are never falsely broken.
* **Acceptance Criteria:**
  - Detects trailing hyphens (`-`) on $P_i$ followed by lowercase string on $P_{i+1}$.
  - Detects non-terminal punctuation with trailing prepositions/conjunctions (`and`, `with`, `or`, `to`).
  - Emits high continuation certainty ($P \le 0.02$) short-circuiting downstream ML.
* **Story Points:** 3  
* **Priority:** P1  
* **Assignee:** Engineer 2  

---

### EPIC-4: 36-Dimensional Feature Engineering & LightGBM (Tier 2)

#### Story 4.1: Pairwise Tabular Feature Vector Extractor
* **User Persona:** As an ML classifier, I need a normalized 36-dimensional feature vector comparing adjacent pages $(P_i, P_{i+1})$ across semantic, PII, typography, syntactic, and lifecycle signals.
* **Acceptance Criteria:**
  - `PairwiseFeatureExtractor.extract_features()` produces a dictionary of 36 float features.
  - Handles edge cases (empty pages, non-Latin text, missing bounding boxes) with safe fallbacks.
  - Feature extraction completes in $<2$ ms per adjacent page pair on CPU.
* **Story Points:** 5  
* **Priority:** P0  
* **Assignee:** Engineer 2 & 3  

#### Story 4.2: LightGBM Pairwise Classifier Training & Tuning
* **User Persona:** As an ML engineer, I want to train a LightGBM binary classifier on 50,000 synthetic page transitions so that ambiguous layouts are resolved with $>96\%$ F1 accuracy.
* **Acceptance Criteria:**
  - Trains gradient-boosted decision trees using histogram binning and GOSS.
  - Hyperparameters optimized via Optuna (`max_depth=5`, `num_leaves=31`, `n_estimators=150`).
  - Evaluates SHAP feature importance to prune redundant features.
  - Serializes trained booster to `models/lightgbm_pairwise.pkl`.
* **Story Points:** 5  
* **Priority:** P0  
* **Assignee:** Engineer 3  

---

### EPIC-5: Structured Sequence Modeling & Viterbi Trellis (Tier 3)

#### Story 5.1: Hidden Markov Model Transition Priors Formulation
* **User Persona:** As a sequence decoder, I need empirical state transition probabilities $P(S_t \mid S_{t-1})$ over states $\{\text{START}, \text{PAGE\_2}, \text{PAGE\_3}, \text{PAGE\_4+}\}$ so that real-world resume length distributions are strictly enforced.
* **Acceptance Criteria:**
  - Implements state transition matrix $\mathbf{A}$ derived from real candidate distribution statistics.
  - Enforces structural impossibility constraints: $P(\text{START} \to \text{PAGE\_3}) = 0.00$.
  - Eliminates orphaned continuation pages.
* **Story Points:** 3  
* **Priority:** P0  
* **Assignee:** Engineer 3  

#### Story 5.2: Viterbi Trellis Dynamic Programming Implementation
* **User Persona:** As a pipeline orchestrator, I want to decode the globally optimal sequence of document boundaries across an entire stream so that local classifier noise is smoothed into valid document slices.
* **Acceptance Criteria:**
  - `ViterbiSequenceOptimizer.decode()` runs forward log-space trellis induction and backtracking.
  - Converts local emission probabilities into global 1-indexed split cut points.
  - Execution time is $<1$ ms for an entire 500-page stream on CPU.
* **Story Points:** 3  
* **Priority:** P0  
* **Assignee:** Engineer 3 & 1  

---

### EPIC-6: Zero-Copy Slicing & Enterprise Delivery

#### Story 6.1: Lossless Zero-Copy PDF Slicing Engine
* **User Persona:** As an operations engineer, I want to slice monolithic PDF streams into individual candidate PDFs without decompressing or re-encoding objects so that vector quality and embedded fonts are 100% preserved.
* **Acceptance Criteria:**
  - `PikePDFSlicer.slice_stream()` takes source PDF and 1-indexed boundary cut points.
  - Slices a 500-page PDF into 200 files in $<400$ ms via zero-copy QPDF cross-reference re-indexing.
  - Generates `manifest.json` containing candidate page ranges and processing metadata.
* **Story Points:** 3  
* **Priority:** P0  
* **Assignee:** Engineer 1  

#### Story 6.2: FastAPI Microservice & Container Packaging
* **User Persona:** As an enterprise client, I want an HTTP API endpoint (`POST /v1/segment-stream`) and an optimized Docker image on GHCR so that the engine deploys seamlessly in cloud or on-premise environments.
* **Acceptance Criteria:**
  - FastAPI endpoint accepts multipart PDF stream uploads and returns segmentation JSON.
  - Health check probe (`GET /v1/health`) for Kubernetes/container orchestration.
  - Multi-stage Docker image builds automatically in GitHub Actions and publishes to `ghcr.io`.
  - Container runs as non-root user `10001` with footprint $<350$ MB.
* **Story Points:** 5  
* **Priority:** P0  
* **Assignee:** Engineer 1  

---

## 4. Sprint Execution Roadmap (5 Sprints)

| Sprint | Focus Area | Primary Deliverables | Target Velocity |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | Scaffolding, Data & Ingestion | • Repo setup, CI/CD with GHCR, Dockerfile<br>• PyMuPDF ingestion parser & schemas<br>• Dataset fetcher & anti-leakage synthetic generator | 16 Story Points |
| **Sprint 2** | Heuristics & Feature Pipeline | • Tier 1 rules (PII, font dominance, pagination, syntax)<br>• 36-dimensional tabular pairwise feature vector extractor<br>• Conditional OCR triage benchmark | 16 Story Points |
| **Sprint 3** | Classical ML & Sequence Modeling | • LightGBM training on 50,000 synthetic pairs<br>• Optuna tuning and SHAP feature importance audit<br>• Viterbi HMM dynamic programming decoder | 16 Story Points |
| **Sprint 4** | Lossless Slicing & Pipeline Assembly | • pikepdf zero-copy PDF slicing engine<br>• End-to-end `SegmentationPipeline` assembly<br>• Full integration test suite & benchmark CLI | 14 Story Points |
| **Sprint 5** | Production Hardening & Gold Standard Eval | • FastAPI microservice packaging<br>• Evaluation against 25 real-world enterprise ATS batches<br>• Performance & latency optimization ($<15$ ms/page) | 12 Story Points |

---
*End of Agile Project Backlog.*
