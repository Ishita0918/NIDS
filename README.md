# NIDS
# 🛡️ Hybrid Metaheuristic NIDS — GA + PSO + Deep Learning

> **Accepted at ICEFCN-2026** — International Conference on Emerging Techniques and Future Threats in Cryptography & Network Security
> 
> Amity School of Engineering and Technology, Amity University Mumbai

A fully automated **Network Intrusion Detection System (NIDS)** that integrates **Genetic Algorithms (GA)** and **Particle Swarm Optimization (PSO)** with Deep Learning to build an intelligent, zero-manual-tuning security pipeline — achieving **82% accuracy** and **96% attack precision** on the NSL-KDD benchmark dataset.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Application](#-web-application)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Authors](#-authors)
- [License](#-license)
- [Citation](#-citation)

---

## 🔍 Overview

Traditional ML optimization for intrusion detection faces three core challenges:

- **Local minima trapping** — gradient-based methods get stuck in poor solutions
- **High-dimensional search** — 41 features and hundreds of hyperparameter combos make exhaustive search impractical
- **Manual dependency** — expert knowledge is needed for architecture design and tuning

This project solves all three using a **five-stage fully automated metaheuristic pipeline** — no manual configuration required at any stage.

---

## 🔁 Pipeline Architecture

```
Raw NSL-KDD Data (41 features)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Preprocessing     │  Label encoding, MinMax scaling, binary labels
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: Feature Selection │  Genetic Algorithm → 18 of 41 features selected
│         (GA — PyGAD)        │  Fitness: RF accuracy − feature count penalty
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Architecture NAS  │  PSO → optimal 3-layer [67, 47, 71] network
│         (PSO — PySwarms)    │  10 particles × 5 iterations
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4: Hyperparameter    │  GA → LR=0.005, Batch=512, RMSprop optimizer
│    Optimization (GA)        │  6 configs/gen × 5 generations
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 5: Final Training    │  Adam, class weights, early stopping, threshold=0.4
│      (TensorFlow/Keras)     │  Result: 82% accuracy, 96% attack precision
└─────────────────────────────┘
```

| Stage | Method | Key Result |
|-------|--------|------------|
| Data Preprocessing | Label Encoding + MinMaxScaler | Clean 41-feature dataset |
| Feature Selection | Genetic Algorithm (PyGAD) | **18 of 41 features** retained (56% reduction) |
| Architecture Search | PSO (PySwarms) | **3-layer [67, 47, 71]** network, dropout=0.13 |
| Hyperparameter Opt | Genetic Algorithm (PyGAD) | **LR=0.005**, Batch=512, RMSprop |
| Final Training | Deep Learning (Keras) | **82% accuracy, 96% attack precision** |

---

## 📊 Results

### Performance Metrics (NSL-KDD Test Set — 22,544 samples)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Overall Accuracy | **82.00%** | Correct predictions across all test samples |
| Attack Precision | **0.96** | 96% of attack alerts are correct |
| Attack Recall | 0.63 | 63% of all attacks detected |
| Normal Recall | 0.96 | 96% of normal traffic correctly identified |
| Macro F1-Score | **0.82** | Balanced performance across both classes |

### Accuracy Progression Through the Pipeline

| Pipeline Stage | Test Accuracy | Improvement |
|----------------|--------------|-------------|
| Baseline (41 features, no opt) | ~75.0% | — |
| After GA Feature Selection | ~77.0% | +2.0% |
| After PSO Architecture Search | ~78.9% | +1.9% |
| After GA Hyperparameter Opt | ~79.3% | +0.4% |
| **Final Model** | **82.0%** | **+7.0% total** |

> 💡 The entire 7-point improvement was achieved through **automated metaheuristic optimization** with **zero manual tuning**.

---

## 🗂️ Project Structure

```
CONFERENCE 1/
│
├── 📄 preprocess.py              # Stage 1: Data cleaning, encoding, normalization
├── 📄 feature_selection.py       # Stage 2: GA-based feature selection
├── 📄 nas_pso.py                 # Stage 3: PSO neural architecture search
├── 📄 hyperparameter_opt.py      # Stage 4: GA hyperparameter optimization
├── 📄 final_train.py             # Stage 5: Final model training & evaluation
│
├── 📄 predict.py                 # CLI prediction on new data
├── 📄 app.py                     # Streamlit interactive web UI
├── 📄 database.py                # SQLite auth & scan history management
│
├── 📦 hybrid_metaheuristic_model.keras   # Saved trained model [see note below]
│
├── 🗃️ KDDTrain+.txt              # NSL-KDD training set [see note below]
├── 🗃️ KDDTest+.txt               # NSL-KDD test set [see note below]
│
└── 🔢 *.npy                      # Auto-generated intermediate arrays
    ├── best_architecture.npy
    ├── best_hyperparams.npy
    ├── selected_features.npy
    ├── X_train.npy / X_test.npy
    ├── X_train_selected.npy / X_test_selected.npy
    └── y_train.npy / y_test.npy
```

> ⚠️ **Note:** The `.keras` model file, `.npy` arrays, and `.txt` dataset files are **not included** in this repository (see `.gitignore`). See [Installation](#-installation) for how to obtain them.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the NSL-KDD dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) and place the following files in the project root:

```
KDDTrain+.txt
KDDTest+.txt
```

### 4. (Optional) Download the pre-trained model

Download `hybrid_metaheuristic_model.keras` from **[Google Drive — link here]** and place it in the project root to skip training and use the app directly.

---

## 🚀 Usage

### Run the full pipeline (step by step)

```bash
python preprocess.py           # Stage 1: Preprocess data
python feature_selection.py    # Stage 2: GA feature selection
python nas_pso.py              # Stage 3: PSO architecture search
python hyperparameter_opt.py   # Stage 4: GA hyperparameter tuning
python final_train.py          # Stage 5: Train and evaluate final model
```

Each stage saves its output as `.npy` files, which are automatically loaded by the next stage.

### Run CLI prediction

```bash
python predict.py
```

### Launch the web application

```bash
streamlit run app.py
```

---

## 🌐 Web Application

The Streamlit web app provides three interactive modes:

| Mode | Description |
|------|-------------|
| **📂 File Upload** | Upload any NSL-KDD `.txt` or `.csv` file — get a donut chart of traffic distribution and risk score histogram |
| **✍️ Manual Input** | Enter connection parameters via dropdowns and sliders — real-time gauge meter shows threat level (0–100%) |
| **📡 Live Simulation** | Simulates live network traffic monitoring — connections processed one-by-one with animated risk score chart |

User authentication and scan history are managed through SQLite (`nids.db`).

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Deep learning model construction and training |
| `PyGAD` | Genetic Algorithm for feature selection and hyperparameter optimization |
| `PySwarms` | Particle Swarm Optimization for neural architecture search |
| `Scikit-learn` | Preprocessing, Random Forest evaluation, metrics |
| `Pandas / NumPy` | Data manipulation and numerical computing |
| `Streamlit` | Interactive web UI |
| `Plotly` | Interactive charts and visualizations |
| `SQLite` | User authentication and scan history |

---

## 🗄️ Dataset

**NSL-KDD** — Network Security Lab KDD Dataset  
Source: [Canadian Institute for Cybersecurity, University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)

| Property | Value |
|----------|-------|
| Training Samples | 125,973 connection records |
| Test Samples | 22,544 connection records |
| Total Features | 41 per connection |
| Selected Features | 18 (after GA selection) |
| Classification | Binary — Normal (0) / Attack (1) |
| Attack Types | DoS, Probe, R2L, U2R |

---

## 👩‍💻 Authors

**Ishita Leonard Dsouza**  
Amity School of Engineering and Technology, Amity University Mumbai  
📧 dsouza.leonard@s.amity.edu

**Dr. Swetta Kukreja**  
Supervisor & Mentor, Amity University Mumbai

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{dsouza2026hybrid,
  title     = {Integration of Metaheuristic Algorithms with Machine Learning and Deep Learning 
               Models for Network Intrusion Detection: A Hybrid GA-PSO Framework Applied to NSL-KDD},
  author    = {Ishita Leonard Dsouza and Kukreja, Swetta},
  booktitle = {International Conference on Emerging Techniques and Future Threats 
               in Cryptography \& Network Security (ICEFCN-2026)},
  year      = {2026},
  institution = {Amity University Mumbai}
}
```

---

<p align="center">
  Made with ❤️ at Amity University Mumbai &nbsp;|&nbsp; ICEFCN-2026
</p>
