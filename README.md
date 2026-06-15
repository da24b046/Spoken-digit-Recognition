# Spoken-digit-Recognition

A machine learning project for recognizing spoken digits (0-9) using Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM). This implementation uses audio feature extraction and statistical modeling to classify spoken digit utterances.

## Project Overview

This project implements a spoken digit recognition system with two complementary approaches:
- **HMM (Hidden Markov Model)**: Sequential modeling with vector quantization for discrete observations
- **GMM (Gaussian Mixture Model)**: Probabilistic modeling of continuous feature distributions

The system processes audio recordings, extracts acoustic features (MFCCs), and trains separate models for each digit (0-9).

## Project Structure

```
Spoken-digit-Recognition/
├── README.md                  # This file
├── Report.pdf                 # Project report with methodology and results
├── feature_extraction.py       # Audio feature extraction utilities
├── gmm.py                      # Gaussian Mixture Model implementation
├── hmm.py                      # Hidden Markov Model and Vector Quantizer implementation
├── prediction.py               # Model training and evaluation
└── recordings/                 # Directory for audio data (speaker 1-6)
    ├── train_recordings/       # Training data (speakers 1-3, 5)
    ├── val_recordings/         # Validation data (speaker 4)
    └── test_recordings/        # Test data (speaker 6)
```

## Features

### Feature Extraction (`feature_extraction.py`)
- **MFCC Features**: 13 Mel-Frequency Cepstral Coefficients per frame
- **Log-Energy**: Energy representation of the signal
- **Delta Features**: First-order derivatives capturing temporal dynamics
- **Total Features**: 27-dimensional vectors (13 MFCC + 1 log-energy + 13 delta)
- **Preprocessing**: Automatic padding for short audio files

### HMM Approach (`hmm.py`)
- **HMM Class**: Implements Baum-Welch algorithm for training
  - Configurable number of states (default: 3 states per digit)
  - Log-space forward-backward algorithm for numerical stability
  - E-step: Computes forward/backward probabilities and state posteriors
  - M-step: Updates transition and emission probabilities
  
- **VectorQuantiser Class**: K-means based feature discretization
  - Converts continuous MFCC features to discrete symbols (64 symbols)
  - Enables HMM to work with emission probabilities over discrete observations

### GMM Approach (`gmm.py`)
- **GaussianMixtureModel Class**: Implements EM algorithm
  - Configurable number of components (default: 4 components per digit)
  - Log-space computations for stability
  - E-step: Computes responsibilities (soft cluster assignments)
  - M-step: Updates mixture weights, means, and covariances
  - Regularization to prevent singular covariance matrices

### Training & Evaluation (`prediction.py`)
- **Data Preparation**: 
  - Normalizes features using global mean and standard deviation
  - Splits data by speaker (train: 1-3,5 | validation: 4 | test: 6)
  
- **Model Training**:
  - 10 HMM models (one per digit) with 3 states and 64 VQ symbols
  - 10 GMM models (one per digit) with 4 components
  
- **Evaluation Metrics**:
  - Overall accuracy
  - Per-digit accuracy breakdown
  - Confusion matrices (normalized by true label)
  - Visualization of results

## Dependencies

```
numpy              # Numerical computations
librosa            # Audio processing and feature extraction
scikit-learn       # Metrics and evaluation
matplotlib         # Visualization and plotting
```

## Usage

### 1. Prepare Audio Data
Place WAV files in the `recordings/` directory following the naming convention:
```
{digit}_{speaker_id}.wav
```
Example: `3_speaker1.wav`, `7_speaker4.wav`

### 2. Extract Features
```bash
python feature_extraction.py
```
This script:
- Processes all audio files in `recordings/`
- Extracts features from each file
- Splits data by speaker into train/val/test directories

### 3. Train Models and Evaluate
```bash
python prediction.py
```
This script:
- Trains 10 HMM models (one per digit)
- Trains 10 GMM models (one per digit)
- Evaluates on validation and test sets
- Generates confusion matrices and saves visualizations

### Output
- `confusionmatrix_hmm.png`: HMM validation and test confusion matrices
- `confusionmatrix_gmm.png`: GMM validation and test confusion matrices
- Console output with per-digit accuracy statistics

## Model Details

### HMM Configuration
- **States**: 3 per digit (left-to-right topology)
- **Observations**: 64 discrete symbols from VQ
- **Training**: Baum-Welch algorithm with convergence tolerance 1e-4
- **Scoring**: Normalized log-likelihood per observation

### GMM Configuration
- **Components**: 4 per digit
- **Features**: 27-dimensional continuous vectors
- **Training**: EM algorithm with convergence tolerance 1e-4
- **Scoring**: Log-likelihood normalized by sequence length

## Results

The models are evaluated on:
- **Validation Set**: Speaker 4 (unseen during training)
- **Test Set**: Speaker 6 (completely unseen speaker)

Confusion matrices show per-digit classification accuracy and common confusion patterns.

## Key Implementation Details

- **Log-space computations**: Prevents numerical underflow in probability calculations
- **Numerical stability**: Clipping, regularization, and pinv for matrix operations
- **Convergence checks**: Early stopping based on log-likelihood improvement
- **Data normalization**: Global z-score normalization across all training features

## References

The methodology and detailed results are documented in the accompanying Report.pdf.

## Author

Created as a machine learning project for spoken digit recognition.
