# Parkinson's Disease Detection using CNN-LSTM

## Overview
This project implements a deep learning model to detect Parkinson's disease based on vocal features from patients. The model combines Convolutional Neural Networks (CNN) for feature extraction and Bidirectional Long Short-Term Memory (Bi-LSTM) networks for sequence pattern recognition. Data preprocessing includes normalization and dimensionality reduction using UMAP. Model evaluation is performed using accuracy, classification reports, confusion matrices, and ROC curves.

## Dataset
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data). It contains multiple vocal features of patients, with a `status` column indicating whether a patient has Parkinson's disease (1) or is healthy (0).

## Features Used
- **Vocal attributes** extracted from patients.
- **Normalization** via StandardScaler.
- **Dimensionality reduction** using UMAP.
- **Feature visualization** with KDE plots and UMAP projections.

## Model Architecture
The deep learning model consists of:
- **Convolutional layers (CNN)** for feature extraction.
- **Batch normalization and dropout** for regularization.
- **Global Average Pooling** for dimensionality reduction.
- **Bidirectional LSTM layers** for sequential pattern recognition.
- **Dense layers** for final classification.
- **Sigmoid activation function** for binary classification.

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas seaborn tensorflow shap umap-learn scikit-learn matplotlib
```

## Usage
1. **Run the script**:
```bash
python parkinsons_detection.py
```
2. The trained model will be saved as `manual_tuned_parkinsons_model.h5`.
3. The script will output accuracy, classification report, confusion matrix, and visualizations.

## Evaluation Metrics
- **Accuracy**
- **Classification Report (Precision, Recall, F1-Score)**
- **Confusion Matrix**
- **ROC Curve with Threshold Annotations**

## Model Performance
The model's performance is evaluated using the test dataset, with detailed visualizations to assess classification accuracy and feature distribution.

## Visualizations
- **Feature Distribution Plots** (Kernel Density Estimation for `PPE` feature)
- **UMAP Projection of Features**
- **Confusion Matrix Heatmap**
- **ROC Curve with Thresholds**

## Output
After execution, the script provides:
- Trained CNN-LSTM model saved as `manual_tuned_parkinsons_model.h5`
- Performance evaluation metrics
- Visual plots for feature analysis and classification performance

## License
This project is open-source and free to use for research and educational purposes.

## Author
[SANJAI.S]

