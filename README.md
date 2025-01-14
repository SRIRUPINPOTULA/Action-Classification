# CSE 515 - Multimedia and Web Databases Project

This repository contains the implementation for **CSE 515 Multimedia and Web Databases**. The project is divided into three phases, each focused on a different aspect of multimedia video analysis. Each phase is located in its respective folder with the corresponding code, data, and outputs.

---

## Table of Contents

- [Phases](#phases)
  - [Phase 1: Feature Extraction and Similarity Measures](#phase-1-feature-extraction-and-similarity-measures)
  - [Phase 2: Dimensionality Reduction and Latent Semantics](#phase-2-dimensionality-reduction-and-latent-semantics)
  - [Phase 3: Indexing, Classification, and Relevance Feedback](#phase-3-indexing-classification-and-relevance-feedback)

## Project Overview

In this project, we explore the multimedia video data and build models for video classification, feature extraction, similarity search, and dimensionality reduction. The tasks include:

- **Phase 1:** Extract video features, build vector models, and calculate similarity/distance measures.
- **Phase 2:** Experiment with dimensionality reduction techniques (PCA, SVD, LDA, k-means) and create latent semantics from the video features.
- **Phase 3:** Implement indexing and classification methods and improve video retrieval through relevance feedback.

---

## Repository Structure

The repository is organized into three folders, each corresponding to one phase of the project:

```
/CSE515_Project
│
├── Phase1
│   ├── Code/
│   ├── Data/
│   ├── Outputs/
│   └── README.md
│
├── Phase2
│   ├── Code/
│   ├── Data/
│   ├── Outputs/
│   └── README.md
│
└── Phase3
    ├── Code/
    ├── Data/
    ├── Outputs/
    └── README.md
```

---

## Setup and Installation

### Prerequisites
Before running the code, ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries (listed in each phase's `requirements.txt` file):
  - `torch`, `torchvision` (for deep learning models)
  - `opencv-python`, `pyav` (for video processing)
  - `numpy`, `scipy` (for mathematical operations)
  - `scikit-learn` (for classification and dimensionality reduction)

To install the necessary packages, you can use `pip`:

```bash
pip install -r requirements.txt
```

### Setup

Each phase has its own folder, and within each folder:

1. Clone or download the repository.
2. Navigate to the relevant phase folder.
3. Follow the instructions in each phase’s README to run the tasks and generate the outputs.

---

## Usage Instructions

### Phase 1: Feature Extraction and Similarity Measures
- Navigate to the `Phase1` folder.
- The code for this phase is located in the `Code/` directory.
- Run the scripts to extract features from videos and compute similarity/distance measures between them.

Example:

```
python `filename.py`
```

### Phase 2: Dimensionality Reduction and Latent Semantics
- Navigate to the `Phase2` folder.
- The code for this phase is located in the `Code/` directory.
- Implement dimensionality reduction techniques and create latent semantic representations of videos.
- You can specify the dimensionality reduction technique (PCA, SVD, LDA, etc.) and the number of latent semantics.

Example:

```
python `filename.py`
```

### Phase 3: Indexing, Classification, and Relevance Feedback
- Navigate to the `Phase3` folder.
- The code for this phase is located in the `Code/` directory.
- Implement indexing (using Locality Sensitive Hashing) and classification models (k-NN, SVM).
- You can also perform relevance feedback to improve search results.

Example:

```
python `filename.py`
```

---

## Deliverables

Each phase should produce the following deliverables:

1. **Code:** Properly commented and modular code.
2. **Outputs:** Outputs for sample inputs, including intermediate results (e.g., feature vectors, classification results, etc.).
3. **Report:** A report for each phase, including:
   - Problem description
   - Methodology (how and why you approached the tasks)
   - Results and insights
   - Input and output formats

The final deliverables should be zipped or tarred and submitted via the Canvas digital dropbox.

---

## Phases

### Phase 1: Feature Extraction and Similarity Measures
- **Objective:** Implement programs to extract video features, perform similarity calculations, and visualize results.
- **Key Tasks:** 
  - Task 0: Download data (HMDB51) and extract video features using pre-trained models (R3D18, HOG, HOF, etc.).
  - Task 1: Implement similarity/distance measures for comparing feature vectors.

### Phase 2: Dimensionality Reduction and Latent Semantics
- **Objective:** Apply dimensionality reduction techniques to improve feature representations and explore latent semantics.
- **Key Tasks:** 
  - Task 0: Compute feature vectors for the target videos.
  - Task 1: Use dimensionality reduction techniques (PCA, SVD, LDA, k-means).
  - Task 2: Implement classification and prediction of video labels.

### Phase 3: Indexing, Classification, and Relevance Feedback
- **Objective:** Implement indexing using Locality Sensitive Hashing (LSH) and improve video search using classification and relevance feedback.
- **Key Tasks:**
  - Task 0: Compute the inherent dimensionality of video labels.
  - Task 1: Implement spectral clustering and k-means clustering for videos.
  - Task 2: Implement k-NN and SVM classifiers for video classification.
  - Task 3: Implement LSH for indexing and relevance feedback-based video search.

---

## Acknowledgments

- This project uses the **HMDB51 dataset** and pre-trained video feature extraction models like **R3D-18** from PyTorch.
- We also rely on popular libraries such as **OpenCV**, **PyTorch**, and **scikit-learn** for video processing, machine learning, and dimensionality reduction.

