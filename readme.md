# Track A: Narrative Consistency Validation System

A comprehensive end-to-end machine learning pipeline for validating the consistency of character backstories against actual novel content using semantic analysis, NLI inference, and ensemble modeling.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Models & Ensemble](#models--ensemble)
- [Output Format](#output-format)
- [Results](#results)
- [Track A Requirements](#track-a-requirements)

---

## 🎯 Overview

This project validates whether character backstories in novels are *consistent* with the actual narrative content or *contradict* it. It combines:

- *Semantic similarity analysis* (sentence embeddings)
- *Natural Language Inference (NLI)* models
- *Multiple machine learning classifiers* (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression)
- *Deep learning transformer* (DeBERTa-v3)
- *Pathway framework* for document retrieval and vector indexing
- *Ensemble voting* to combine predictions

Each prediction comes with detailed evidence showing:
- Which passages from the novel were matched
- NLI analysis (ENTAILMENT, CONTRADICTION, or NEUTRAL)
- Confidence scores
- Reasoning based on evidence

---

## 🔍 Problem Statement

Given a character backstory claim, determine if it:

1. *Consistent (Label = 1)*: The claim aligns with facts in the novel
2. *Contradict (Label = 0)*: The claim contradicts or fabricates events not in the novel

*Dataset:*
- *Train:* 80 labeled examples (40 consistent, 40 contradict)
- *Test:* ~20 unlabeled examples to predict
- *Novels:* 2 full-length classic novels (~40k+ lines each)

---

## 🏗️ Architecture

```bash
┌─────────────────────────────────────────────────────┐
│         Input: Character Backstories                 │
└────────────────────┬────────────────────────────────┘
                     ↓
        ┌────────────────────────────┐
        │  Feature Extraction Layer  │
        ├────────────────────────────┤
        │ • Semantic Similarity      │
        │   (all-MiniLM-L6-v2)      │
        │ • NLI Inference            │
        │   (DeBERTa-v3-MNLI)       │
        └────────────────┬───────────┘
                         ↓
        ┌────────────────────────────────────────┐
        │    ML Models (Parallel)                 │
        ├────────────────────────────────────────┤
        │ • XGBoost (40% weight)                 │
        │ • LightGBM (15% weight)                │
        │ • CatBoost (15% weight)                │
        │ • Random Forest (10% weight)           │
        │ • Logistic Regression (5% weight)      │
        │         ↓                              │
        │ Deep Learning Model (40% weight)       │
        │ • DeBERTa Transformer                  │
        └────────────────┬─────────────────────┘
                         ↓
        ┌──────────────────────────────┐
        │  Weighted Ensemble (0-1.0)   │
        │  Threshold: > 0.5 = consistent
        └──────────────────┬───────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │  Pathway Document Retrieval          │
        ├──────────────────────────────────────┤
        │ • Extract backstory claims           │
        │ • Find relevant novel passages       │
        │ • NLI analysis on each passage       │
        │ • Generate reasoning                 │
        └──────────────────┬──────────────────┘
                           ↓
        ┌──────────────────────────────┐
        │  Output: Predictions + Evidence
        ├──────────────────────────────┤
        │ • ID & Character             │
        │ • Prediction (0 or 1)        │
        │ • Confidence Score           │
        │ • Backstory Claims           │
        │ • Evidence Passages          │
        │ • NLI Results                │
        │ • Reasoning                  │
        └──────────────────────────────┘
```

---

## ✨ Features

### 1. *Semantic Feature Extraction*
- *Semantic Similarity*: Compares embedding vectors of backstory claims with novel passages
- *NLI Scores*: Determines logical relationships (entailment, contradiction, neutral)
- *Context Extraction*: Finds character mentions with surrounding context

### 2. *Multi-Model Ensemble*
- 5 traditional ML classifiers trained on semantic features
- 1 transformer model reading full text directly
- Weighted voting for robust predictions

### 3. *Pathway-Based Evidence Retrieval*
- Chunks novel into 1000-character segments (200-char overlap)
- Creates dense embeddings for all chunks
- Semantic search for relevant passages
- Evidence scoring with confidence metrics

### 4. *Detailed Evidence Output*
Each prediction includes:
- Up to 5 evidence passages with line numbers
- NLI classification for each passage
- Confidence scores (0.0-1.0)
- Contradiction/entailment counts
- Human-readable reasoning

### 5. *Cross-Validation Analysis*
- 5-fold stratified cross-validation
- Overfitting detection
- Per-model performance metrics (Accuracy, F1-Score)

---

## 🛠️ Technology Stack

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| *pandas* | Latest | Data manipulation |
| *numpy* | Latest | Numerical computing |
| *scikit-learn* | Latest | ML utilities, cross-validation |
| *pytorch* | Latest | Deep learning framework |

### NLP & Embeddings
| Library | Model | Purpose |
|---------|-------|---------|
| *sentence-transformers* | all-MiniLM-L6-v2 | Semantic embeddings |
| *transformers* | DeBERTa-v3-base-MNLI | NLI classification |
| *transformers* | DeBERTa-v3-small | Binary classification |

### Machine Learning Models
| Model | Framework | Hyperparameters |
|-------|-----------|-----------------|
| XGBoost | xgboost | 200 trees, depth=5, lr=0.05 |
| LightGBM | lightgbm | 200 trees, depth=5, lr=0.05 |
| CatBoost | catboost | 200 iterations, depth=5, lr=0.05 |
| Random Forest | sklearn | 200 trees, max_depth=10 |
| Logistic Regression | sklearn | max_iter=1000 |
| DeBERTa | transformers | Pre-trained, frozen |

### Data Framework
| Tool | Purpose |
|------|---------|
| *Pathway* | Document ingestion, chunking, vector indexing |

---

## 📁 Project Structure

```bash
KDSH/
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── solution.ipynb                      # Main notebook (23 cells)
│
├── data/
│   ├── train.csv                       # 80 training examples (labeled)
│   ├── test.csv                        # ~20 test examples (to predict)
│   └── novels/
│       ├── In search of the castaways.txt    # Full novel text
│       └── The Count of Monte Cristo.txt     # Full novel text
│
├── predictions.csv                     # Final submission (ID, label)
├── test_predictions_with_evidence.csv  # Test predictions + evidence
└── train_predictions_with_evidence.csv # Train predictions + evidence
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- ~4GB disk space for models
- GPU recommended (CUDA 11.8+) for transformer inference

### Steps

1. *Clone/Navigate to project directory:*
```bash
git clone https://github.com/Aditya-KumarJha/KDSH.git
```
```bash
cd KDSH
```


2. *Create virtual environment:*
```bash
python -m venv venv
```
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. *Install dependencies:*
```bash
pip install -r requirements.txt
```

4. *Verify installation:*
```bash
python -c "import pathway, sentence_transformers, torch; print('✓ All imports successful')"
```

---

## 🚀 Usage

### Running the Pipeline

1. *Open the notebook:*
bash
jupyter notebook solution.ipynb


2. *Run all cells in order* (top to bottom):
   - Cell 1: Title/Overview (markdown)
   - Cell 2: Import all libraries
   - Cell 3: Load novels and data
   - Cell 4: Define helper functions
   - Cell 5: Load NLI & embedding models
   - Cell 6: Extract semantic features (train & test)
   - Cell 7: Train ML models
   - Cell 8: Cross-validation analysis
   - Cell 9: Load transformer model
   - Cell 10: Get transformer predictions (test)
   - Cell 11: Ensemble predictions
   - Cell 12: Markdown - Pathway introduction
   - Cell 13: Create Pathway document store
   - Cell 14: Define retrieval functions
   - Cell 15: Generate test predictions with evidence
   - Cell 16: Generate train predictions with evidence
   - Cell 17: Define CSV saving function
   - Cell 18: Save all results
   - Cell 19: Display sample results

3. *Outputs generated:*
   - predictions.csv - Simple submission format
   - test_predictions_with_evidence.csv - Full evidence
   - train_predictions_with_evidence.csv - Full evidence

---

## 🔄 Pipeline Workflow

### Stage 1: Data Loading (Cell 3)

Load novels → Parse into lines
Load train.csv → 80 examples with labels
Load test.csv → ~20 examples (no labels)


### Stage 2: Preprocessing (Cells 4-5)

For each example:
├─ Extract character contexts from novel
├─ Create full context (book + character + content)
├─ Get embedding of content
├─ Get embeddings of character contexts
└─ Compute NLI scores


### Stage 3: Feature Extraction (Cell 5)

For each example, compute:
├─ max_similarity → Max cosine similarity to character contexts
├─ mean_similarity → Mean cosine similarity
├─ context_count → Number of contexts found
├─ entailment → NLI entailment score
├─ contradiction → NLI contradiction score
└─ neutral → NLI neutral score


*Result:* 6 numerical features per example

### Stage 4: Model Training (Cell 6)

Train 5 ML models on X_train (80 examples × 6 features)
Using y_train (80 labels: 0 or 1)


### Stage 5: Cross-Validation (Cell 7)

5-fold stratified CV on training data
├─ Check Accuracy per fold
├─ Check F1-Score per fold
└─ Calculate overfitting gap


### Stage 6: Transformer Predictions (Cells 9-10)

For each test example:
├─ Tokenize full_context
├─ Pass through DeBERTa-v3-small
└─ Get probability of class 1 (consistent)


### Stage 7: Ensemble (Cell 11)

final_score = (
    0.40 * transformer_pred +
    0.15 * xgb_pred +
    0.15 * lgbm_pred +
    0.15 * catboost_pred +
    0.10 * rf_pred +
    0.05 * lr_pred
)

prediction = 1 if final_score > 0.5 else 0


### Stage 8: Evidence Retrieval (Cells 13-16)

For each prediction:
├─ Extract backstory claims (split by sentences)
├─ For each claim:
│  ├─ Search Pathway vector store for relevant passages
│  ├─ Get top 3 similar passages
│  └─ Run NLI on each passage
└─ Aggregate evidence and generate reasoning


### Stage 9: Output (Cells 17-19)

Save:
├─ predictions.csv → ID, Label (submission format)
├─ test_predictions_with_evidence.csv → Full details
└─ train_predictions_with_evidence.csv → Full details


---

## 🤖 Models & Ensemble

### ML Models (Feature-Based)

*Input:* 6 semantic features per example

| Model | Trees | Depth | Learning Rate | Notes |
|-------|-------|-------|-----------------|-------|
| XGBoost | 200 | 5 | 0.05 | Gradient boosting |
| LightGBM | 200 | 5 | 0.05 | Fast, leaf-based |
| CatBoost | 200 | 5 | 0.05 | Handles categories |
| Random Forest | 200 | 10 | N/A | Bootstrap aggregation |
| Logistic Regression | N/A | N/A | N/A | Linear classifier |

*Training:* Each model learns patterns in the 6 features to predict 0 (contradict) or 1 (consistent)

### Transformer Model (Text-Based)

*Model:* DeBERTa-v3-small (pre-trained, fine-tuned on NLI)

*Input:* Full concatenated text

"Book: <book_name>
Character: <character_name>
Caption: <optional_caption>
Content: <backstory_claim>"


*Output:* Probability of class 1 (consistent)

### Ensemble Strategy

*Weighted Voting:*

Final Score = ∑(Weight_i × Prediction_i)

Weights:
- Transformer: 40% (most comprehensive, reads full text)
- XGBoost: 15% (strong gradient boosting)
- LightGBM: 15% (fast, accurate)
- CatBoost: 15% (handles categorical patterns)
- Random Forest: 10% (robust baseline)
- Logistic Regression: 5% (linear patterns)


*Decision Threshold:*
- If Score > 0.5 → Predict 1 (consistent)
- If Score ≤ 0.5 → Predict 0 (contradict)

---

## 📄 Output Format

### predictions.csv (Submission Format)
csv
id,label
95,1
136,1
59,1
...


### test_predictions_with_evidence.csv

*Columns:*
1. id - Example ID
2. book_name - Novel name
3. character - Character name
4. prediction - 0 (contradict) or 1 (consistent)
5. confidence - 0.0-1.0 probability score
6. backstory_claims - Extracted claims (pipe-separated)
7. evidence_summary - Up to 5 evidence passages with NLI analysis
8. reasoning - Human-readable explanation
9. contradictions - Count of contradicting passages
10. entailments - Count of supporting passages

*Example Row:*

ID: 95
Book: The Count of Monte Cristo
Character: Noirtier
Prediction: 1 (consistent)
Confidence: 0.563

Backstory Claims:
"Learning that Villefort meant to denounce him..."

Evidence:
--- Evidence 1 ---
Claim: [claim text]
Passage (Lines 37239-37258): [novel text excerpt]
NLI: contradiction (score: 0.852)

--- Evidence 2 ---
Claim: [claim text]
Passage (Lines 28108-28134): [novel text excerpt]
NLI: neutral (score: 0.905)

Reasoning: Found 1 contradictions vs 0 supporting evidences. 
The backstory contradicts established narrative facts.

Contradictions: 1
Entailments: 0


---

## 📊 Results

### Cross-Validation Performance (Training Data)

Expected results on 5-fold CV:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| XGBoost | ~0.65 | ~0.65 |
| LightGBM | ~0.63 | ~0.63 |
| CatBoost | ~0.60 | ~0.60 |
| Random Forest | ~0.55 | ~0.55 |
| Logistic Regression | ~0.50 | ~0.50 |
| *Ensemble* | *~0.68* | *~0.68* |

*Note:* Limited training data (80 examples) → Moderate performance expected

### Train Accuracy
- Ensemble achieves ~70-75% on train set
- Moderate overfitting detected (gap ~10-15%)

### Key Observations
1. *Transformer model* contributes most (40% weight) due to full-text understanding
2. *Semantic features* help identify relevant passages
3. *NLI analysis* is most reliable indicator of contradiction vs consistency
4. *Ensemble voting* reduces individual model bias

---

## 🎯 Track A Requirements

This project satisfies Track A requirements:

### ✅ Requirement 1: Pathway Framework Usage
- *Implementation:* Cells 13-16
- *Details:*
  - Novel chunking: 1000-char chunks with 200-char overlap
  - Vector indexing: All chunks embedded with all-MiniLM-L6-v2
  - Semantic retrieval: Top-K passage matching using cosine similarity
  - Metadata tracking: Line numbers for each passage

### ✅ Requirement 2: Evidence-Based Predictions
- *Implementation:* All test/train predictions include evidence
- *Details:*
  - Up to 5 passages retrieved per claim
  - NLI analysis on each passage
  - Contradiction/entailment counts
  - Reasoning based on evidence summary

### ✅ Requirement 3: Source Location Tracking
- *Implementation:* evidence_summary column
- *Details:*
  - Each passage shows "Lines X-Y" reference
  - Original novel text excerpt included
  - Similarity scores provided

### ✅ Requirement 4: Detailed Reasoning
- *Implementation:* reasoning column
- *Details:*
  - Count of contradictions found
  - Count of entailments found
  - Natural language explanation
  - Final conclusion about consistency

---

## 📝 Data Format Details

### train.csv Columns

id          - Example identifier (integer)
book_name   - Novel name (string)
char        - Character name (string)
caption     - Optional context (string or NaN)
content     - Backstory claim (string)
label       - Ground truth (0=contradict, 1=consistent)


### Example Training Data
csv
id,book_name,char,caption,content,label
46,In Search of the Castaways,Thalcave,,Thalcave's people faded as colonists...,consistent
137,The Count of Monte Cristo,Faria,The Origin of His Connection...,Suspected again in 1815...,contradict


---

## 🔧 Configuration & Hyperparameters

### Model Hyperparameters (Cell 6)
python
# XGBoost
n_estimators=200      # Number of boosting rounds
max_depth=5           # Tree depth
learning_rate=0.05    # Learning rate
eval_metric='logloss' # Loss function

# LightGBM
n_estimators=200
max_depth=5
learning_rate=0.05
verbose=-1            # Suppress output

# CatBoost
iterations=200
depth=5
learning_rate=0.05
verbose=0             # No verbose output

# Random Forest
n_estimators=200
max_depth=10          # Deeper than boosting models
random_state=42

# Logistic Regression
max_iter=1000         # Maximum iterations
random_state=42


### Pathway Configuration (Cell 13)
python
chunk_size = 1000          # Characters per chunk
overlap = 200              # Overlap between chunks
top_k = 5                  # Top passages to retrieve
nli_max_length = 512       # Max tokens for NLI input
transformer_batch_size = 8 # Batch size for inference


### Ensemble Weights (Cell 11)
python
weights = {
    'transformer': 0.4,
    'xgb': 0.15,
    'lgbm': 0.15,
    'catboost': 0.15,
    'rf': 0.1,
    'lr': 0.05
}
# Sum = 1.0 (normalized)


---

## 🐛 Troubleshooting

### Common Issues

*Issue:* ModuleNotFoundError: No module named 'pathway'
- *Solution:* pip install pathway

*Issue:* CUDA out of memory on transformer inference
- *Solution:* Reduce batch_size in Cell 10 (default: 8 → try 4 or 2)

*Issue:* Slow feature extraction (Cell 5)
- *Solution:* Normal for first run (~5-10 min). Models download ~1GB on first use.

*Issue:* Low accuracy on cross-validation
- *Solution:* This is expected with limited training data (80 examples). Ensemble helps mitigate.

*Issue:* NLI model very slow
- *Solution:* Model runs on CPU by default. GPU would speed up 5-10x. Set device=0 if CUDA available.

---

## 📚 References

### Papers & Models
- *Sentence Transformers:* https://www.sbert.net/
- *DeBERTa:* He et al., 2021 - Decoding-enhanced BERT
- *MNLI Dataset:* Williams et al., 2018 - Entailment NLI task
- *XGBoost:* Chen & Guestrin, 2016 - Gradient Boosting
- *LightGBM:* Ke et al., 2017 - Light Gradient Boosting
- *CatBoost:* Dorogush et al., 2018 - Categorical Boosting
- *Pathway:* https://pathway.com/ - Data processing framework

### Datasets
- Training examples: 80 hand-annotated backstories
- Novels: Classic literature (public domain)
- Test set: ~20 unlabeled examples

---

## 📄 License & Attribution

This project uses:
- Public domain novels (Project Gutenberg)
- Open-source ML libraries (Apache 2.0, MIT licenses)
- Pre-trained models (HuggingFace Hub)

---

## 👥 Contact & Support

*Project:* Track A - Narrative Consistency Validation
*Repository:* https://github.com/Aditya-KumarJha/KDSH
*Last Updated:* January 2026

For issues or questions, open a GitHub issue or review the notebook comments.

---

## 🎓 Learning Outcomes

This project demonstrates:

1. *NLP Techniques*
   - Semantic embeddings (sentence-transformers)
   - Natural language inference (MNLI)
   - Text preprocessing and chunking

2. *Machine Learning*
   - Multiple algorithm implementations
   - Ensemble methods
   - Cross-validation and overfitting analysis
   - Feature engineering

3. *Deep Learning*
   - Pre-trained transformer models
   - Fine-tuned inference
   - Token handling and batching

4. *Data Engineering*
   - Vector indexing and semantic search
   - Evidence retrieval pipelines
   - Large document processing

5. *Software Engineering*
   - Modular notebook design
   - Clear documentation
   - Reproducible results
   - Output formatting