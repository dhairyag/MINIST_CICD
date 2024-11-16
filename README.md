# MNIST Digit Classification with PyTorch

A lightweight Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Project Overview

This project implements a simple CNN architecture to classify handwritten digits from the MNIST dataset. The model achieves >95% accuracy while maintaining a small parameter count (<25,000 parameters).

## Technical Architecture

### Model Architecture (SimpleCNN)
- Input: 28x28 grayscale images
- Architecture:
  - Conv1: 1 → 8 channels (3x3 kernel, padding=1)
  - BatchNorm + ReLU + MaxPool2d + Dropout(0.1)
  - Conv2: 8 → 16 channels (3x3 kernel, padding=1)
  - BatchNorm + ReLU + MaxPool2d + Dropout(0.1)
  - Fully Connected: 784 → 24 → 10
  - Output: 10 classes (digits 0-9)

### Key Features
- Automated ML pipeline using GitHub Actions
- Continuous integration with automated testing
- Model parameter efficiency (<25K parameters)
- Standardized data preprocessing
- CPU and GPU compatibility


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- tqdm

## Installation

### Clone the repository
```bash
git clone git@github.com:dhairyag/MINIST_CICD.git
cd MINIST_CICD
```

### Install dependencies
```bash
pip install -r requirements.txt
```


## Usage

### Training
```bash
python train.py
```
The trained model will be saved in the `models/` directory with a timestamp.

### Testing
```bash
python -m pytest tests/
```


## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Sets up Python 3.8
2. Installs dependencies
3. Trains the model
4. Runs the test suite

## Model Performance

- Accuracy: >95% on MNIST test set
- Parameters: <25,000
- Training: Single epoch with Adam optimizer
- Learning Rate: 0.001
- Batch Size: 64

## Data Preprocessing

- Normalization: Mean=0.1307, Std=0.3081
- Input: 28x28 grayscale images
- Data augmentation: None
